# ============================================================
# predict_lora_ka_loda.py
#
# Full future forecast using the trained LoRA Kronos model.
#
# Experiment:
#   - Reliance data only
#   - Last 400 bars before 2021-01-01 = model context
#   - 2021-01-01 onward = completely unseen future
#   - Generate predictions for the ENTIRE future period
#   - Compare prediction against actual data
#   - Save Parquet + interactive Plotly HTML
#
# IMPORTANT:
#   We do NOT modify model/kronos.py.
#
#   Kronos' built-in auto_regressive_inference() correctly
#   generates all pred_len tokens, but its final decoder only
#   decodes the last max_context tokens. This script contains
#   our own equivalent inference function that decodes ALL
#   generated future tokens.
# ============================================================


import os
import sys
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go

from tqdm import trange


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

# Make ~/Kronos importable.
sys.path.insert(
    0,
    PROJECT_ROOT,
)


# ============================================================
# KRONOS IMPORTS
# ============================================================

from model import Kronos, KronosTokenizer
from model.kronos import (
    calc_time_stamps,
    sample_from_logits,
)


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Pretrained Kronos
# ------------------------------------------------------------

MODEL_PATH = (
    "/home/soq/__shutupandbendover/"
    "het-uchiha/weights/Kronos-base"
)

TOKENIZER_PATH = (
    "/home/soq/__shutupandbendover/"
    "het-uchiha/weights/Kronos-Tokenizer-base"
)


# ------------------------------------------------------------
# Reliance dataset
# ------------------------------------------------------------

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "het",
    "data",
    "RIL_19-26.parquet",
)


# ------------------------------------------------------------
# Our trained LoRA checkpoint
# ------------------------------------------------------------

LORA_PATH = os.path.join(
    PROJECT_ROOT,
    "het",
    "checkpoints_ours",
    "lora_best.pt",
)


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "het",
    "results",
)

FORECAST_PATH = os.path.join(
    RESULTS_DIR,
    "lora_full_forecast.parquet",
)

PLOT_PATH = os.path.join(
    RESULTS_DIR,
    "lora_full_forecast.html",
)


# ------------------------------------------------------------
# Experiment definition
# ------------------------------------------------------------

CUTOFF_DATE = "2021-01-01"

CONTEXT_LENGTH = 400

# Kronos context limit.
MAX_CONTEXT = 512

# Same clipping used during training.
CLIP = 5


# ------------------------------------------------------------
# Generation parameters
# ------------------------------------------------------------

TEMPERATURE = 1.0

TOP_K = 0

TOP_P = 0.9

SAMPLE_COUNT = 1


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


# ============================================================
# LoRA IMPLEMENTATION
# ============================================================

class LoRALinear(nn.Module):
    """
    Frozen base Linear layer + trainable low-rank update.

    This MUST match the LoRA layer used during training.

    y = base(x) + scaling * dropout(x) @ A.T @ B.T

    B is initialized to zero during training, meaning the
    initial LoRA model is identical to the pretrained model.
    """

    def __init__(
        self,
        base_linear,
        rank=8,
        alpha=16,
        dropout=0.05,
    ):
        super().__init__()

        self.base = base_linear

        # Freeze original pretrained layer.
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.rank = rank
        self.alpha = alpha

        self.scaling = (
            alpha / rank
        )

        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base_linear.in_features,
            )
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                base_linear.out_features,
                rank,
            )
        )

        # Same initialization used during training.
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

        self.dropout = nn.Dropout(
            dropout
        )

    def forward(self, x):

        base_output = self.base(x)

        lora_output = self.dropout(x)

        lora_output = (
            lora_output
            @ self.lora_A.t()
        )

        lora_output = (
            lora_output
            @ self.lora_B.t()
        )

        return (
            base_output
            + self.scaling * lora_output
        )


# ============================================================
# LoRA INJECTION
# ============================================================

def inject_lora(model):
    """
    Inject the exact LoRA configuration used for training.

    Targets:
        q_proj
        k_proj
        v_proj
        out_proj

    Rank:
        8

    Alpha:
        16

    Dropout:
        0.05
    """

    rank = 8
    alpha = 16
    dropout = 0.05

    target_names = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
    )

    # Freeze EVERYTHING first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Replace target attention projections.
    for block in model.transformer:

        attention = block.self_attn

        for name in target_names:

            original_layer = getattr(
                attention,
                name,
            )

            lora_layer = LoRALinear(
                original_layer,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

            setattr(
                attention,
                name,
                lora_layer,
            )

    return model


# ============================================================
# LOAD DATA
# ============================================================

def load_reliance_data():

    print(
        "\nLoading Reliance dataset:"
    )

    print(
        DATA_PATH
    )

    df = pd.read_parquet(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Datetime index
    # --------------------------------------------------------

    df.index = pd.to_datetime(
        df.index
    )

    # --------------------------------------------------------
    # Handle MultiIndex columns
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):
        df.columns = (
            df.columns
            .get_level_values(0)
        )

    # --------------------------------------------------------
    # Normalize column names
    # --------------------------------------------------------

    df.columns = [
        str(column).lower()
        for column in df.columns
    ]

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: "
            + str(missing)
        )

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    if "amount" not in df.columns:

        print(
            "Amount column missing. "
            "Creating amount = volume × OHLC mean."
        )

        df["amount"] = (
            df["volume"]
            * df[
                [
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            ].mean(axis=1)
        )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_index()

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    if df.index.has_duplicates:

        print(
            "Warning: duplicate timestamps found."
        )

        df = df[
            ~df.index.duplicated(
                keep="last"
            )
        ]

    return df


# ============================================================
# SELECT 400-BAR CONTEXT
# ============================================================

def build_context_and_future(df):

    # Everything strictly before 2021-01-01.
    before_cutoff = df[
        df.index < CUTOFF_DATE
    ]

    if len(before_cutoff) < CONTEXT_LENGTH:

        raise RuntimeError(
            f"Only {len(before_cutoff)} bars exist "
            f"before {CUTOFF_DATE}, but we need "
            f"{CONTEXT_LENGTH}."
        )

    # EXACTLY the final 400 bars before cutoff.
    context = (
        before_cutoff
        .tail(CONTEXT_LENGTH)
        .copy()
    )

    # Everything after the context.
    #
    # This includes the entire future beginning
    # 2021-01-01.
    future_actual = df[
        df.index > context.index[-1]
    ].copy()

    if future_actual.empty:

        raise RuntimeError(
            "No future data found after "
            "the 400-bar context."
        )

    return (
        context,
        future_actual,
    )


# ============================================================
# LOAD LoRA CHECKPOINT
# ============================================================

def load_lora_weights(model):

    if not os.path.exists(
        LORA_PATH
    ):

        raise FileNotFoundError(
            "LoRA checkpoint not found:\n"
            + LORA_PATH
        )

    print(
        "\nLoading LoRA checkpoint:"
    )

    print(
        LORA_PATH
    )

    checkpoint = torch.load(
        LORA_PATH,
        map_location=DEVICE,
    )

    if (
        "lora_state_dict"
        not in checkpoint
    ):

        raise RuntimeError(
            "Checkpoint does not contain "
            "'lora_state_dict'."
        )

    lora_state_dict = (
        checkpoint[
            "lora_state_dict"
        ]
    )

    missing, unexpected = (
        model.load_state_dict(
            lora_state_dict,
            strict=False,
        )
    )

    # Missing base parameters are EXPECTED,
    # because the checkpoint contains only
    # LoRA parameters.
    #
    # Unexpected parameters are NOT expected.
    if unexpected:

        raise RuntimeError(
            "Unexpected keys found in "
            "LoRA checkpoint:\n"
            + "\n".join(
                unexpected
            )
        )

    print(
        "Best epoch:",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "Best validation loss:",
        checkpoint.get(
            "val_loss",
            "unknown",
        ),
    )

    print(
        "LoRA tensors loaded:",
        len(lora_state_dict),
    )


# ============================================================
# FULL AUTOREGRESSIVE INFERENCE
# ============================================================

def generate_full_forecast(
    tokenizer,
    model,
    x,
    x_stamp,
    y_stamp,
    max_context,
    pred_len,
    clip=5,
    temperature=1.0,
    top_k=0,
    top_p=0.9,
    sample_count=1,
    verbose=True,
):
    """
    Autoregressive inference based directly on Kronos'
    auto_regressive_inference().

    The important difference:

    Original Kronos:
        Generates pred_len tokens
        BUT decodes only the final max_context tokens.

    This version:
        Generates pred_len tokens
        AND decodes ALL pred_len generated tokens.

    Therefore a 1392-bar forecast remains 1392 bars.
    """

    with torch.no_grad():

        # ----------------------------------------------------
        # Clip input
        # ----------------------------------------------------

        x = torch.clip(
            x,
            -clip,
            clip,
        )

        device = x.device

        # ----------------------------------------------------
        # Repeat for sample_count
        # ----------------------------------------------------

        x = (
            x.unsqueeze(1)
            .repeat(
                1,
                sample_count,
                1,
                1,
            )
            .reshape(
                -1,
                x.size(1),
                x.size(2),
            )
            .to(device)
        )

        x_stamp = (
            x_stamp.unsqueeze(1)
            .repeat(
                1,
                sample_count,
                1,
                1,
            )
            .reshape(
                -1,
                x_stamp.size(1),
                x_stamp.size(2),
            )
            .to(device)
        )

        y_stamp = (
            y_stamp.unsqueeze(1)
            .repeat(
                1,
                sample_count,
                1,
                1,
            )
            .reshape(
                -1,
                y_stamp.size(1),
                y_stamp.size(2),
            )
            .to(device)
        )

        # ----------------------------------------------------
        # Encode historical data
        # ----------------------------------------------------

        x_token = tokenizer.encode(
            x,
            half=True,
        )

        initial_seq_len = x.size(1)

        batch_size = (
            x_token[0].size(0)
        )

        # All timestamps:
        #
        # historical timestamps
        # +
        # future timestamps
        #
        full_stamp = torch.cat(
            [
                x_stamp,
                y_stamp,
            ],
            dim=1,
        )

        # ----------------------------------------------------
        # Allocate ALL future tokens
        # ----------------------------------------------------

        generated_pre = (
            x_token[0]
            .new_empty(
                batch_size,
                pred_len,
            )
        )

        generated_post = (
            x_token[1]
            .new_empty(
                batch_size,
                pred_len,
            )
        )

        # ----------------------------------------------------
        # Rolling context buffers
        # ----------------------------------------------------

        pre_buffer = (
            x_token[0]
            .new_zeros(
                batch_size,
                max_context,
            )
        )

        post_buffer = (
            x_token[1]
            .new_zeros(
                batch_size,
                max_context,
            )
        )

        buffer_len = min(
            initial_seq_len,
            max_context,
        )

        if buffer_len > 0:

            start_idx = max(
                0,
                initial_seq_len
                - max_context,
            )

            pre_buffer[
                :,
                :buffer_len,
            ] = x_token[0][
                :,
                start_idx:
                start_idx + buffer_len,
            ]

            post_buffer[
                :,
                :buffer_len,
            ] = x_token[1][
                :,
                start_idx:
                start_idx + buffer_len,
            ]

        # ----------------------------------------------------
        # Generation loop
        # ----------------------------------------------------

        if verbose:

            iterator = trange(
                pred_len,
                desc="Generating",
            )

        else:

            iterator = range(
                pred_len
            )

        for i in iterator:

            current_seq_len = (
                initial_seq_len + i
            )

            window_len = min(
                current_seq_len,
                max_context,
            )

            # ------------------------------------------------
            # Select current token context
            # ------------------------------------------------

            if (
                current_seq_len
                <= max_context
            ):

                input_s1 = (
                    pre_buffer[
                        :,
                        :window_len,
                    ]
                )

                input_s2 = (
                    post_buffer[
                        :,
                        :window_len,
                    ]
                )

            else:

                input_s1 = pre_buffer

                input_s2 = post_buffer

            # ------------------------------------------------
            # Select matching timestamp context
            # ------------------------------------------------

            context_end = (
                current_seq_len
            )

            context_start = max(
                0,
                context_end
                - max_context,
            )

            current_stamp = (
                full_stamp[
                    :,
                    context_start:
                    context_end,
                    :,
                ].contiguous()
            )

            # ------------------------------------------------
            # Predict S1
            # ------------------------------------------------

            s1_logits, context = (
                model.decode_s1(
                    input_s1,
                    input_s2,
                    current_stamp,
                )
            )

            s1_logits = (
                s1_logits[
                    :,
                    -1,
                    :,
                ]
            )

            sample_pre = (
                sample_from_logits(
                    s1_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_logits=True,
                )
            )

            # ------------------------------------------------
            # Predict S2 conditioned on S1
            # ------------------------------------------------

            s2_logits = (
                model.decode_s2(
                    context,
                    sample_pre,
                )
            )

            s2_logits = (
                s2_logits[
                    :,
                    -1,
                    :,
                ]
            )

            sample_post = (
                sample_from_logits(
                    s2_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sample_logits=True,
                )
            )

            # ------------------------------------------------
            # Store generated tokens
            # ------------------------------------------------

            generated_pre[
                :,
                i,
            ] = (
                sample_pre
                .squeeze(-1)
            )

            generated_post[
                :,
                i,
            ] = (
                sample_post
                .squeeze(-1)
            )

            # ------------------------------------------------
            # Update rolling context
            # ------------------------------------------------

            if (
                current_seq_len
                < max_context
            ):

                pre_buffer[
                    :,
                    current_seq_len,
                ] = (
                    sample_pre
                    .squeeze(-1)
                )

                post_buffer[
                    :,
                    current_seq_len,
                ] = (
                    sample_post
                    .squeeze(-1)
                )

            else:

                pre_buffer.copy_(
                    torch.roll(
                        pre_buffer,
                        shifts=-1,
                        dims=1,
                    )
                )

                post_buffer.copy_(
                    torch.roll(
                        post_buffer,
                        shifts=-1,
                        dims=1,
                    )
                )

                pre_buffer[
                    :,
                    -1,
                ] = (
                    sample_pre
                    .squeeze(-1)
                )

                post_buffer[
                    :,
                    -1,
                ] = (
                    sample_post
                    .squeeze(-1)
                )

        # ====================================================
        # IMPORTANT
        # ====================================================
        #
        # DO NOT concatenate x_token here and then take
        # the last max_context tokens.
        #
        # We specifically want ONLY the generated future
        # tokens, all pred_len of them.
        # ====================================================

        z = tokenizer.decode(
            [
                generated_pre,
                generated_post,
            ],
            half=True,
        )

        # ----------------------------------------------------
        # Shape:
        #
        # [batch * sample_count, pred_len, features]
        # ----------------------------------------------------

        z = z.reshape(
            -1,
            sample_count,
            z.size(1),
            z.size(2),
        )

        # Average multiple samples.
        preds = torch.mean(
            z,
            dim=1,
        )

        return preds.cpu().numpy()


# ============================================================
# PLOTLY GRAPH
# ============================================================

def create_plot(
    context,
    future_actual,
    prediction,
):

    fig = go.Figure()

    # --------------------------------------------------------
    # Historical 400 bars
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=context.index,
            y=context["close"],
            mode="lines",
            name="Historical Close",
            line=dict(
                width=1.5,
            ),
        )
    )

    # --------------------------------------------------------
    # Actual future
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=future_actual.index,
            y=future_actual["close"],
            mode="lines",
            name="Actual Future",
            line=dict(
                width=1.5,
            ),
        )
    )

    # --------------------------------------------------------
    # LoRA prediction
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=prediction.index,
            y=prediction["close"],
            mode="lines",
            name="LoRA Prediction",
            line=dict(
                width=2,
            ),
        )
    )

    # --------------------------------------------------------
    # Forecast boundary
    # --------------------------------------------------------

    forecast_start = (
        future_actual.index[0]
    )

    fig.add_vline(
        x=forecast_start,
        line_dash="dash",
        annotation_text="Forecast starts",
        annotation_position="top left",
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(
        title=(
            "RELIANCE.NS - Kronos LoRA "
            "Full Future Forecast"
        ),
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "KRONOS LoRA FULL FUTURE FORECAST"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # 1. LOAD PRETRAINED MODEL
    # ========================================================

    print(
        "\n[1/6] Loading pretrained Kronos..."
    )

    model = (
        Kronos
        .from_pretrained(
            MODEL_PATH
        )
        .to(DEVICE)
    )

    # ========================================================
    # 2. LOAD TOKENIZER
    # ========================================================

    print(
        "[2/6] Loading tokenizer..."
    )

    tokenizer = (
        KronosTokenizer
        .from_pretrained(
            TOKENIZER_PATH
        )
        .to(DEVICE)
    )

    tokenizer.eval()

    # ========================================================
    # 3. INJECT LoRA + LOAD CHECKPOINT
    # ========================================================

    print(
        "[3/6] Injecting LoRA..."
    )

    model = inject_lora(
        model
    )

    model = model.to(
        DEVICE
    )

    load_lora_weights(
        model
    )

    model.eval()

    # ========================================================
    # 4. LOAD RELIANCE DATA
    # ========================================================

    print(
        "\n[4/6] Loading Reliance data..."
    )

    df = load_reliance_data()

    # ========================================================
    # BUILD EXPERIMENT WINDOWS
    # ========================================================

    context, future_actual = (
        build_context_and_future(
            df
        )
    )

    print(
        "\n" + "-" * 72
    )

    print(
        "400-BAR HISTORICAL CONTEXT"
    )

    print(
        "-" * 72
    )

    print(
        f"Start : {context.index[0]}"
    )

    print(
        f"End   : {context.index[-1]}"
    )

    print(
        f"Bars  : {len(context)}"
    )

    print(
        "\n" + "-" * 72
    )

    print(
        "ACTUAL FUTURE"
    )

    print(
        "-" * 72
    )

    print(
        f"Start : {future_actual.index[0]}"
    )

    print(
        f"End   : {future_actual.index[-1]}"
    )

    print(
        f"Bars  : {len(future_actual)}"
    )

    # ========================================================
    # VERIFY EXPERIMENT
    # ========================================================

    assert (
        len(context)
        == CONTEXT_LENGTH
    )

    assert (
        context.index[-1]
        < pd.Timestamp(
            CUTOFF_DATE
        )
    )

    assert (
        future_actual.index[0]
        >= pd.Timestamp(
            CUTOFF_DATE
        )
    )

    pred_len = len(
        future_actual
    )

    print(
        f"\nForecast horizon: "
        f"{pred_len} bars"
    )

    # ========================================================
    # 5. PREPARE INPUT
    # ========================================================

    print(
        "\n[5/6] Preparing model input..."
    )

    feature_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]

    # --------------------------------------------------------
    # Raw 400-bar data
    # --------------------------------------------------------

    x = context[
        feature_columns
    ].values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Per-window normalization
    #
    # EXACTLY the same idea used during training.
    # --------------------------------------------------------

    x_mean = np.mean(
        x,
        axis=0,
    )

    x_std = np.std(
        x,
        axis=0,
    )

    x_norm = (
        x - x_mean
    ) / (
        x_std + 1e-5
    )

    x_norm = np.clip(
        x_norm,
        -CLIP,
        CLIP,
    )

    # --------------------------------------------------------
    # Time stamps
    #
    # calc_time_stamps() expects a Series because the
    # Kronos implementation uses `.dt`.
    # --------------------------------------------------------

    x_timestamp = pd.Series(
        context.index
    )

    y_timestamp = pd.Series(
        future_actual.index
    )

    x_stamp_df = (
        calc_time_stamps(
            x_timestamp
        )
    )

    y_stamp_df = (
        calc_time_stamps(
            y_timestamp
        )
    )

    # --------------------------------------------------------
    # Add batch dimension
    # --------------------------------------------------------

    x_norm = x_norm[
        np.newaxis,
        :,
    ]

    x_stamp = (
        x_stamp_df.values
        .astype(np.float32)
        [
            np.newaxis,
            :,
        ]
    )

    y_stamp = (
        y_stamp_df.values
        .astype(np.float32)
        [
            np.newaxis,
            :,
        ]
    )

    print(
        f"Input shape: {x_norm.shape}"
    )

    print(
        f"Future timestamp shape: "
        f"{y_stamp.shape}"
    )

    print(
        "\nThe model receives ONLY:"
    )

    print(
        f"  {len(context)} historical bars"
    )

    print(
        "The actual future is NOT passed "
        "into the model."
    )

    # ========================================================
    # GENERATE
    # ========================================================

    print(
        "\nStarting full future generation..."
    )

    print(
        f"Generating {pred_len} future bars..."
    )

    normalized_predictions = (
        generate_full_forecast(
            tokenizer=tokenizer,
            model=model,
            x=torch.from_numpy(
                x_norm
            ).to(DEVICE),
            x_stamp=torch.from_numpy(
                x_stamp
            ).to(DEVICE),
            y_stamp=torch.from_numpy(
                y_stamp
            ).to(DEVICE),
            max_context=MAX_CONTEXT,
            pred_len=pred_len,
            clip=CLIP,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            sample_count=SAMPLE_COUNT,
            verbose=True,
        )
    )

    # ========================================================
    # VERIFY OUTPUT
    # ========================================================

    print(
        "\nGenerated normalized shape:",
        normalized_predictions.shape,
    )

    if (
        normalized_predictions.shape[1]
        != pred_len
    ):

        raise RuntimeError(
            "CRITICAL: Generated prediction "
            f"length is "
            f"{normalized_predictions.shape[1]}, "
            f"but expected {pred_len}."
        )

    # ========================================================
    # DENORMALIZE
    # ========================================================

    predictions = (
        normalized_predictions.squeeze(0)
        * (x_std + 1e-5)
        + x_mean
    )

    # ========================================================
    # CREATE PREDICTION DATAFRAME
    # ========================================================

    prediction = pd.DataFrame(
        predictions,
        columns=feature_columns,
        index=future_actual.index,
    )

    # ========================================================
    # FINAL SANITY CHECK
    # ========================================================

    print(
        "\nPrediction shape:",
        prediction.shape,
    )

    if len(prediction) != len(
        future_actual
    ):

        raise RuntimeError(
            "Prediction and actual future "
            "lengths do not match."
        )

    if prediction.isnull().values.any():

        raise RuntimeError(
            "Prediction contains NaN values."
        )

    # ========================================================
    # BUILD COMPARISON DATAFRAME
    # ========================================================

    comparison = pd.DataFrame(
        index=future_actual.index
    )

    comparison[
        "actual_open"
    ] = future_actual["open"]

    comparison[
        "actual_high"
    ] = future_actual["high"]

    comparison[
        "actual_low"
    ] = future_actual["low"]

    comparison[
        "actual_close"
    ] = future_actual["close"]

    comparison[
        "actual_volume"
    ] = future_actual["volume"]

    comparison[
        "predicted_open"
    ] = prediction["open"]

    comparison[
        "predicted_high"
    ] = prediction["high"]

    comparison[
        "predicted_low"
    ] = prediction["low"]

    comparison[
        "predicted_close"
    ] = prediction["close"]

    comparison[
        "predicted_volume"
    ] = prediction["volume"]

    # ========================================================
    # SAVE PARQUET
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    comparison.to_parquet(
        FORECAST_PATH
    )

    print(
        "\nForecast comparison saved:"
    )

    print(
        FORECAST_PATH
    )

    # ========================================================
    # 6. CREATE PLOTLY GRAPH
    # ========================================================

    print(
        "\n[6/6] Creating Plotly graph..."
    )

    fig = create_plot(
        context=context,
        future_actual=future_actual,
        prediction=prediction,
    )

    fig.write_html(
        PLOT_PATH,
        include_plotlyjs=True,
    )

    print(
        "\nPlotly graph saved:"
    )

    print(
        PLOT_PATH
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "FORECAST COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Historical context : "
        f"{len(context)} bars"
    )

    print(
        f"Prediction         : "
        f"{len(prediction)} bars"
    )

    print(
        f"Actual future      : "
        f"{len(future_actual)} bars"
    )

    print(
        f"\nContext:"
    )

    print(
        f"  {context.index[0]}"
        f" -> "
        f"{context.index[-1]}"
    )

    print(
        f"\nForecast:"
    )

    print(
        f"  {prediction.index[0]}"
        f" -> "
        f"{prediction.index[-1]}"
    )

    print(
        f"\nActual:"
    )

    print(
        f"  {future_actual.index[0]}"
        f" -> "
        f"{future_actual.index[-1]}"
    )

    print(
        "\nOutput files:"
    )

    print(
        f"  Parquet: {FORECAST_PATH}"
    )

    print(
        f"  Plotly : {PLOT_PATH}"
    )

    # --------------------------------------------------------
    # Sample predictions
    # --------------------------------------------------------

    print(
        "\nFirst 5 predicted closes:"
    )

    print(
        prediction[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ].head()
    )

    print(
        "\nLast 5 predicted closes:"
    )

    print(
        prediction[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ].tail()
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "DONE"
    )

    print(
        "=" * 72
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()