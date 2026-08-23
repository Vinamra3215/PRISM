import os
import sys
import math
import random
import gc

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. KRONOS IMPORT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, PROJECT_ROOT)

from model import Kronos, KronosTokenizer, KronosPredictor
from model.kronos import calc_time_stamps


# ============================================================
# 2. CONFIGURATION
# ============================================================

DATA_PATH = os.path.expanduser(
    "~/Kronos/het/data/NIFTY50_5Y_OHLCV.parquet"
)

MODEL_PATH = (
    "/home/soq/__shutupandbendover/"
    "het-uchiha/weights/Kronos-base"
)

TOKENIZER_PATH = (
    "/home/soq/__shutupandbendover/"
    "het-uchiha/weights/Kronos-Tokenizer-base"
)


# ------------------------------------------------------------
# Rolling experiment
# ------------------------------------------------------------

# The experiment STARTS here.
# There is intentionally NO fixed end date.
#
# Therefore all available data from 2022-01-01 through the
# final row of the parquet file is used.

START_DATE = "2022-01-01"
END_DATE = None

WINDOW_SIZE = 40
TRAIN_SIZE = 32
VAL_SIZE = 8


# Safety check for the experiment definition.
assert TRAIN_SIZE + VAL_SIZE == WINDOW_SIZE


# ------------------------------------------------------------
# LoRA
# ------------------------------------------------------------

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
)


# ------------------------------------------------------------
# LoRA training
# ------------------------------------------------------------

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01

MAX_STEPS = 1000

VAL_EVERY = 10
PATIENCE = 5

GRAD_CLIP_NORM = 1.0

CLIP_VALUE = 5.0


# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

PRED_LEN = 1

TEMPERATURE = 1.0
TOP_P = 0.9
SAMPLE_COUNT = 1


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

CSV_OUTPUT = (
    "nifty50_rolling_lora_predictions.csv"
)

HTML_OUTPUT = (
    "nifty50_rolling_lora_predictions.html"
)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

if torch.cuda.is_available():

    DEVICE = "cuda:0"

else:

    DEVICE = "cpu"


# ------------------------------------------------------------
# Data columns
# ------------------------------------------------------------

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
]

VOLUME_COLUMN = "volume"
AMOUNT_COLUMN = "amount"


# ============================================================
# 3. LoRA LINEAR LAYER
# ============================================================

class LoRALinear(nn.Module):
    """
    Frozen pretrained Linear layer plus a trainable low-rank
    LoRA update.

        output = base(x) + scaling * B(A(dropout(x)))

    B is initialized to zero, so before training the LoRA
    wrapped model produces exactly the pretrained model output.
    """

    def __init__(
        self,
        base_linear,
        rank,
        alpha,
        dropout,
    ):

        super().__init__()

        if not isinstance(
            base_linear,
            nn.Linear,
        ):

            raise TypeError(
                "LoRALinear requires nn.Linear"
            )

        self.base = base_linear

        # Freeze pretrained parameters.
        for parameter in self.base.parameters():

            parameter.requires_grad = False

        self.rank = rank
        self.alpha = alpha

        self.scaling = (
            alpha / rank
        )

        # IMPORTANT:
        # Create LoRA parameters on the same device as
        # the pretrained Kronos layer.
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base_linear.in_features,
                device=base_linear.weight.device,
            )
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                base_linear.out_features,
                rank,
                device=base_linear.weight.device,
            )
        )

        # Standard LoRA initialization.
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

        # B remains zero initialized.

        self.dropout = (
            nn.Dropout(dropout)
            if dropout > 0
            else nn.Identity()
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
# 4. INJECT LoRA
# ============================================================

def inject_lora(model):
    """
    Freeze the complete pretrained Kronos model and replace
    q/k/v/out attention projections with fresh LoRA layers.

    Returns only the trainable LoRA parameters.
    """

    # Freeze absolutely everything first.
    for parameter in model.parameters():

        parameter.requires_grad = False

    trainable_parameters = []

    replaced_modules = []

    for block_index, block in enumerate(
        model.transformer
    ):

        attention = block.self_attn

        for target_name in LORA_TARGETS:

            original = getattr(
                attention,
                target_name,
            )

            if not isinstance(
                original,
                nn.Linear,
            ):

                raise TypeError(
                    f"Expected "
                    f"transformer.{block_index}"
                    f".self_attn.{target_name} "
                    f"to be nn.Linear, "
                    f"got {type(original)}"
                )

            lora_layer = LoRALinear(
                original,
                rank=LORA_R,
                alpha=LORA_ALPHA,
                dropout=LORA_DROPOUT,
            )

            setattr(
                attention,
                target_name,
                lora_layer,
            )

            trainable_parameters.extend(
                [
                    lora_layer.lora_A,
                    lora_layer.lora_B,
                ]
            )

            replaced_modules.append(
                f"transformer.{block_index}"
                f".self_attn.{target_name}"
            )

    return trainable_parameters


# ============================================================
# 5. LOAD NIFTY DATA
# ============================================================

def load_nifty():

    print(
        "\nLoading NIFTY 50 data:"
    )

    print(
        DATA_PATH
    )

    if not os.path.exists(
        DATA_PATH
    ):

        raise FileNotFoundError(
            f"\nData file not found:\n"
            f"{DATA_PATH}"
        )

    df = pd.read_parquet(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Handle MultiIndex columns
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        df.columns = [
            str(column[0]).strip().lower()
            for column in df.columns
        ]

    else:

        df.columns = [
            str(column).strip().lower()
            for column in df.columns
        ]

    # --------------------------------------------------------
    # Handle timestamp column
    # --------------------------------------------------------

    if "timestamps" not in df.columns:

        if "date" in df.columns:

            df.rename(
                columns={
                    "date": "timestamps"
                },
                inplace=True,
            )

        elif "datetime" in df.columns:

            df.rename(
                columns={
                    "datetime": "timestamps"
                },
                inplace=True,
            )

        else:

            # Date is stored as the DataFrame index.
            df = df.reset_index()

            df.columns = [
                str(column).strip().lower()
                for column in df.columns
            ]

            if "date" in df.columns:

                df.rename(
                    columns={
                        "date": "timestamps"
                    },
                    inplace=True,
                )

            elif "datetime" in df.columns:

                df.rename(
                    columns={
                        "datetime": "timestamps"
                    },
                    inplace=True,
                )

            elif "index" in df.columns:

                df.rename(
                    columns={
                        "index": "timestamps"
                    },
                    inplace=True,
                )

    if "timestamps" not in df.columns:

        raise RuntimeError(
            "Could not identify the date/timestamp "
            "column.\n"
            f"Available columns: "
            f"{df.columns.tolist()}"
        )

    # --------------------------------------------------------
    # Convert timestamps
    # --------------------------------------------------------

    df["timestamps"] = pd.to_datetime(
        df["timestamps"],
        errors="coerce",
    )

    # Remove timezone if present.
    if df["timestamps"].dt.tz is not None:

        df["timestamps"] = (
            df["timestamps"]
            .dt.tz_localize(None)
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "timestamps",
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
            "\nMissing required columns: "
            + str(missing)
            + "\nAvailable columns:\n"
            + str(df.columns.tolist())
        )

    # --------------------------------------------------------
    # Keep required columns
    # --------------------------------------------------------

    df = df[
        required_columns
    ].copy()

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df.dropna(
        subset=required_columns,
        inplace=True,
    )

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df.sort_values(
        "timestamps",
        inplace=True,
    )

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    df.drop_duplicates(
        subset="timestamps",
        keep="last",
        inplace=True,
    )

    df.reset_index(
        drop=True,
        inplace=True,
    )

    # --------------------------------------------------------
    # Create amount
    #
    # Kronos tokenizer uses six OHLCV+amount features.
    # --------------------------------------------------------

    df[AMOUNT_COLUMN] = (
        df[VOLUME_COLUMN]
        * df[PRICE_COLUMNS].mean(
            axis=1
        )
    )

    # --------------------------------------------------------
    # START from 2022-01-01.
    #
    # There is deliberately NO 2022-12-31 cutoff.
    # Everything available after 2022 is retained.
    # --------------------------------------------------------

    df = df[
        df["timestamps"]
        >= pd.Timestamp(START_DATE)
    ].copy()

    # Optional end date.
    # Currently None, meaning use the final available row.

    if END_DATE is not None:

        df = df[
            df["timestamps"]
            <= pd.Timestamp(END_DATE)
        ].copy()

    df.reset_index(
        drop=True,
        inplace=True,
    )

    # --------------------------------------------------------
    # Validate minimum data
    # --------------------------------------------------------

    if len(df) <= WINDOW_SIZE:

        raise RuntimeError(
            f"Only {len(df)} rows available "
            f"after {START_DATE}. "
            f"Need more than {WINDOW_SIZE}."
        )

    # --------------------------------------------------------
    # Print final data range
    # --------------------------------------------------------

    print(
        "\nExperiment data:"
    )

    print(
        f"Start: "
        f"{df['timestamps'].iloc[0]}"
    )

    print(
        f"End:   "
        f"{df['timestamps'].iloc[-1]}"
    )

    print(
        f"Rows:  "
        f"{len(df)}"
    )

    return df


# ============================================================
# 6. NORMALIZATION
# ============================================================

def normalize_window(
    window_df,
    mean,
    std,
):
    """
    Normalize a dataframe using externally supplied statistics.

    This is important for validation:
    validation data must NOT determine its own normalization
    statistics.
    """

    values = window_df[
        PRICE_COLUMNS
        + [
            VOLUME_COLUMN,
            AMOUNT_COLUMN,
        ]
    ].values.astype(
        np.float32
    )

    normalized = (
        values - mean
    ) / (
        std + 1e-5
    )

    normalized = np.clip(
        normalized,
        -CLIP_VALUE,
        CLIP_VALUE,
    )

    timestamps = pd.Series(
        pd.to_datetime(
            window_df["timestamps"]
        )
    ).reset_index(
        drop=True
    )

    stamp_df = calc_time_stamps(
        timestamps
    )

    x_tensor = torch.from_numpy(
        normalized[
            np.newaxis,
            :,
            :,
        ]
    ).to(DEVICE)

    stamp_tensor = torch.from_numpy(
        stamp_df.values.astype(
            np.float32
        )[
            np.newaxis,
            :,
            :,
        ]
    ).to(DEVICE)

    return (
        x_tensor,
        stamp_tensor,
    )


# ============================================================
# 7. PREPARE TRAIN / VALIDATION TENSORS
# ============================================================

def prepare_train_val_tensors(
    train_df,
    val_df,
):

    columns = (
        PRICE_COLUMNS
        + [
            VOLUME_COLUMN,
            AMOUNT_COLUMN,
        ]
    )

    # --------------------------------------------------------
    # Compute normalization statistics ONLY from training data.
    # --------------------------------------------------------

    train_values = train_df[
        columns
    ].values.astype(
        np.float32
    )

    train_mean = train_values.mean(
        axis=0
    )

    train_std = train_values.std(
        axis=0
    )

    # --------------------------------------------------------
    # Normalize train using train statistics.
    # --------------------------------------------------------

    train_x, train_stamp = (
        normalize_window(
            train_df,
            train_mean,
            train_std,
        )
    )

    # --------------------------------------------------------
    # Normalize validation using TRAIN statistics.
    # --------------------------------------------------------

    val_x, val_stamp = (
        normalize_window(
            val_df,
            train_mean,
            train_std,
        )
    )

    return (
        train_x,
        train_stamp,
        val_x,
        val_stamp,
    )


# ============================================================
# 8. TEACHER-FORCED LOSS
# ============================================================

def teacher_forced_loss(
    model,
    s1_ids,
    s2_ids,
    stamp_tensor,
):

    s1_logits, s2_logits = model(
        s1_ids,
        s2_ids,
        stamp=stamp_tensor,
        use_teacher_forcing=True,
        s1_targets=s1_ids,
    )

    s1_loss = F.cross_entropy(
        s1_logits[:, :-1, :].reshape(
            -1,
            s1_logits.size(-1),
        ),
        s1_ids[:, 1:].reshape(
            -1
        ),
    )

    s2_loss = F.cross_entropy(
        s2_logits[:, :-1, :].reshape(
            -1,
            s2_logits.size(-1),
        ),
        s2_ids[:, 1:].reshape(
            -1
        ),
    )

    return (
        s1_loss + s2_loss
    )


# ============================================================
# 9. TOKENIZE TRAIN / VALIDATION
# ============================================================

def tokenize_train_val(
    model,
    tokenizer,
    train_df,
    val_df,
):

    (
        train_x,
        train_stamp,
        val_x,
        val_stamp,
    ) = prepare_train_val_tensors(
        train_df,
        val_df,
    )

    with torch.no_grad():

        train_tokens = tokenizer.encode(
            train_x
        )

        val_tokens = tokenizer.encode(
            val_x
        )

        train_s1, train_s2 = (
            model.embedding.split_token(
                train_tokens,
                model.embedding.s2_bits,
            )
        )

        val_s1, val_s2 = (
            model.embedding.split_token(
                val_tokens,
                model.embedding.s2_bits,
            )
        )

    return (
        train_s1,
        train_s2,
        train_stamp,
        val_s1,
        val_s2,
        val_stamp,
    )


# ============================================================
# 10. TRAIN ONE FRESH LoRA MODEL
# ============================================================

def train_one_window(
    model,
    tokenizer,
    train_df,
    val_df,
    window_number,
):

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"WINDOW {window_number}"
    )

    print(
        f"Train: "
        f"{train_df['timestamps'].iloc[0].date()}"
        f" → "
        f"{train_df['timestamps'].iloc[-1].date()}"
    )

    print(
        f"Val:   "
        f"{val_df['timestamps'].iloc[0].date()}"
        f" → "
        f"{val_df['timestamps'].iloc[-1].date()}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    (
        train_s1,
        train_s2,
        train_stamp,
        val_s1,
        val_s2,
        val_stamp,
    ) = tokenize_train_val(
        model,
        tokenizer,
        train_df,
        val_df,
    )

    # --------------------------------------------------------
    # Fresh LoRA
    # --------------------------------------------------------

    lora_parameters = inject_lora(
        model
    )

    trainable_count = sum(
        parameter.numel()
        for parameter
        in lora_parameters
    )

    print(
        f"Trainable LoRA parameters: "
        f"{trainable_count:,}"
    )

    # --------------------------------------------------------
    # Optimizer sees ONLY LoRA parameters.
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        lora_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_loss = float(
        "inf"
    )

    best_state = None

    checks_without_improvement = 0

    best_step = 0

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for step in range(
        1,
        MAX_STEPS + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        train_loss = teacher_forced_loss(
            model,
            train_s1,
            train_s2,
            train_stamp,
        )

        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            lora_parameters,
            GRAD_CLIP_NORM,
        )

        optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if (
            step == 1
            or step % VAL_EVERY == 0
            or step == MAX_STEPS
        ):

            model.eval()

            with torch.no_grad():

                val_loss = (
                    teacher_forced_loss(
                        model,
                        val_s1,
                        val_s2,
                        val_stamp,
                    )
                )

            train_value = (
                train_loss.item()
            )

            val_value = (
                val_loss.item()
            )

            print(
                f"step {step:4d} | "
                f"train {train_value:.5f} | "
                f"val {val_value:.5f}"
            )

            # ------------------------------------------------
            # New best
            # ------------------------------------------------

            if (
                val_value
                < best_val_loss
            ):

                best_val_loss = (
                    val_value
                )

                best_step = step

                checks_without_improvement = 0

                # Save ONLY trainable LoRA parameters.
                best_state = {
                    name:
                    parameter.detach()
                    .cpu()
                    .clone()

                    for name, parameter
                    in model.named_parameters()

                    if parameter.requires_grad
                }

                print(
                    "  -> new best"
                )

            else:

                checks_without_improvement += 1

                print(
                    f"  -> no improvement "
                    f"({checks_without_improvement}/"
                    f"{PATIENCE})"
                )

                if (
                    checks_without_improvement
                    >= PATIENCE
                ):

                    print(
                        f"  -> early stopping "
                        f"at step {step}"
                    )

                    break

    # --------------------------------------------------------
    # Ensure a best checkpoint exists.
    # --------------------------------------------------------

    if best_state is None:

        raise RuntimeError(
            "No best LoRA state was produced."
        )

    # --------------------------------------------------------
    # Restore best LoRA parameters.
    # --------------------------------------------------------

    current_parameters = dict(
        model.named_parameters()
    )

    for name, saved_value in (
        best_state.items()
    ):

        if name not in current_parameters:

            raise RuntimeError(
                f"Best LoRA parameter "
                f"'{name}' is missing "
                f"from the current model."
            )

        current_parameters[
            name
        ].data.copy_(
            saved_value.to(
                DEVICE
            )
        )

    model.eval()

    print(
        f"Best validation loss: "
        f"{best_val_loss:.5f}"
    )

    print(
        f"Best step: "
        f"{best_step}"
    )

    return model


# ============================================================
# 11. ONE-DAY PREDICTION
# ============================================================

def predict_one_day(
    model,
    tokenizer,
    context_df,
    prediction_timestamp,
):

    predictor = KronosPredictor(
        model,
        tokenizer,
        device=DEVICE,
        max_context=512,
    )

    # --------------------------------------------------------
    # The model receives ONLY the 40 historical bars.
    # --------------------------------------------------------

    x_df = context_df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    x_timestamp = context_df[
        "timestamps"
    ].copy()

    # --------------------------------------------------------
    # Future timestamp only.
    # No actual future price is passed.
    # --------------------------------------------------------

    y_timestamp = pd.Series(
        [
            prediction_timestamp
        ]
    )

    prediction = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=TEMPERATURE,
        top_p=TOP_P,
        sample_count=SAMPLE_COUNT,
    )

    return prediction.iloc[0]


# ============================================================
# 12. METRICS
# ============================================================

def calculate_metrics(
    results,
):

    metrics = {}

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:

        actual = results[
            f"actual_{column}"
        ].values

        predicted = results[
            f"pred_{column}"
        ].values

        mae = np.mean(
            np.abs(
                actual - predicted
            )
        )

        rmse = np.sqrt(
            np.mean(
                (
                    actual
                    - predicted
                ) ** 2
            )
        )

        metrics[column] = {
            "MAE": mae,
            "RMSE": rmse,
        }

    # --------------------------------------------------------
    # One-day directional accuracy.
    #
    # For every prediction:
    #
    # predicted direction =
    # predicted close - previous actual close
    #
    # actual direction =
    # actual close - previous actual close
    # --------------------------------------------------------

    if (
        "previous_actual_close"
        in results.columns
    ):

        previous_close = results[
            "previous_actual_close"
        ].values

        actual_close = results[
            "actual_close"
        ].values

        predicted_close = results[
            "pred_close"
        ].values

        actual_direction = np.sign(
            actual_close
            - previous_close
        )

        predicted_direction = np.sign(
            predicted_close
            - previous_close
        )

        directional_accuracy = (
            actual_direction
            == predicted_direction
        ).mean() * 100

    else:

        directional_accuracy = np.nan

    return (
        metrics,
        directional_accuracy,
    )


# ============================================================
# 13. CREATE PLOTLY HTML
# ============================================================

def create_html(
    results,
):

    fig = go.Figure()

    # --------------------------------------------------------
    # Actual close
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=results[
                "timestamp"
            ],
            y=results[
                "actual_close"
            ],
            mode="lines",
            name="Actual Close",
        )
    )

    # --------------------------------------------------------
    # Predicted close
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=results[
                "timestamp"
            ],
            y=results[
                "pred_close"
            ],
            mode="lines",
            name="LoRA Predicted Close",
        )
    )

    fig.update_layout(
        title=(
            "NIFTY 50 - Rolling LoRA "
            "One-Day-Ahead Forecast"
        ),
        xaxis_title="Date",
        yaxis_title="NIFTY 50",
        hovermode="x unified",
        template="plotly_white",
    )

    fig.write_html(
        HTML_OUTPUT,
        include_plotlyjs=True,
        full_html=True,
    )


# ============================================================
# 14. MAIN ROLLING EXPERIMENT
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "NIFTY 50 ROLLING LoRA EXPERIMENT"
    )

    print(
        "=" * 70
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Start date: {START_DATE}"
    )

    if END_DATE is None:

        print(
            "End date: FINAL AVAILABLE DATE"
        )

    else:

        print(
            f"End date: {END_DATE}"
        )

    print(
        f"Window: {WINDOW_SIZE} trading days"
    )

    print(
        f"Training: {TRAIN_SIZE} trading days"
    )

    print(
        f"Validation: {VAL_SIZE} trading days"
    )

    print(
        "Prediction horizon: 1 trading day"
    )

    print(
        "Fresh Kronos + Fresh LoRA: YES"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_nifty()

    # --------------------------------------------------------
    # Every prediction requires 40 previous observations.
    #
    # Therefore:
    #
    # first 40 rows -> first prediction is row 41
    #
    # total predictions = N - 40
    # --------------------------------------------------------

    total_predictions = (
        len(df)
        - WINDOW_SIZE
    )

    print(
        f"\nTotal predictions: "
        f"{total_predictions}"
    )

    if total_predictions <= 0:

        raise RuntimeError(
            "Not enough data for a single "
            "40-day rolling prediction."
        )

    # --------------------------------------------------------
    # Load tokenizer ONCE.
    #
    # It is frozen and contains no LoRA state.
    # --------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        KronosTokenizer
        .from_pretrained(
            TOKENIZER_PATH
        )
        .to(DEVICE)
    )

    tokenizer.eval()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = []

    # --------------------------------------------------------
    # Rolling loop
    #
    # prediction_index = 40:
    #
    # context = rows 0..39
    # prediction = row 40
    #
    # prediction_index = 41:
    #
    # context = rows 1..40
    # prediction = row 41
    #
    # etc.
    # --------------------------------------------------------

    for prediction_index in range(
        WINDOW_SIZE,
        len(df),
    ):

        window_number = (
            prediction_index
            - WINDOW_SIZE
            + 1
        )

        # ----------------------------------------------------
        # Exactly 40 historical trading days.
        # ----------------------------------------------------

        window = df.iloc[
            prediction_index
            - WINDOW_SIZE:
            prediction_index
        ].copy()

        if len(window) != WINDOW_SIZE:

            raise RuntimeError(
                f"Window {window_number}: "
                f"expected {WINDOW_SIZE} rows, "
                f"got {len(window)}."
            )

        # ----------------------------------------------------
        # First 32 = training.
        # Last 8 = validation.
        # ----------------------------------------------------

        train_df = window.iloc[
            :TRAIN_SIZE
        ].copy()

        val_df = window.iloc[
            TRAIN_SIZE:
        ].copy()

        if len(train_df) != TRAIN_SIZE:

            raise RuntimeError(
                f"Window {window_number}: "
                f"expected {TRAIN_SIZE} "
                f"training rows."
            )

        if len(val_df) != VAL_SIZE:

            raise RuntimeError(
                f"Window {window_number}: "
                f"expected {VAL_SIZE} "
                f"validation rows."
            )

        # ----------------------------------------------------
        # The 41st day is completely unseen during training.
        # ----------------------------------------------------

        actual_row = df.iloc[
            prediction_index
        ]

        prediction_timestamp = (
            actual_row["timestamps"]
        )

        previous_actual_close = float(
            df.iloc[
                prediction_index - 1
            ]["close"]
        )

        print(
            "\n\n"
            + "#" * 70
        )

        print(
            f"ROLLING PREDICTION "
            f"{window_number}/"
            f"{total_predictions}"
        )

        print(
            f"Context: "
            f"{window['timestamps'].iloc[0].date()}"
            f" → "
            f"{window['timestamps'].iloc[-1].date()}"
        )

        print(
            f"Train:   "
            f"{train_df['timestamps'].iloc[0].date()}"
            f" → "
            f"{train_df['timestamps'].iloc[-1].date()}"
        )

        print(
            f"Val:     "
            f"{val_df['timestamps'].iloc[0].date()}"
            f" → "
            f"{val_df['timestamps'].iloc[-1].date()}"
        )

        print(
            f"Predict: "
            f"{prediction_timestamp.date()}"
        )

        print(
            "#" * 70
        )

        # ====================================================
        # FRESH PRETRAINED KRONOS
        # ====================================================

        print(
            "\nLoading FRESH pretrained Kronos..."
        )

        model = (
            Kronos
            .from_pretrained(
                MODEL_PATH
            )
            .to(DEVICE)
        )

        # Make sure the pretrained model starts frozen.
        model.eval()

        # ====================================================
        # FRESH LoRA + TRAIN
        # ====================================================

        model = train_one_window(
            model,
            tokenizer,
            train_df,
            val_df,
            window_number,
        )

        # ====================================================
        # PREDICT NEXT DAY
        # ====================================================

        print(
            "\nPredicting next trading day..."
        )

        prediction = predict_one_day(
            model,
            tokenizer,
            window,
            prediction_timestamp,
        )

        # ====================================================
        # SAVE RESULT
        # ====================================================

        result = {
            "timestamp": prediction_timestamp,

            "previous_actual_close":
                previous_actual_close,

            "actual_open": float(
                actual_row["open"]
            ),

            "actual_high": float(
                actual_row["high"]
            ),

            "actual_low": float(
                actual_row["low"]
            ),

            "actual_close": float(
                actual_row["close"]
            ),

            "actual_volume": float(
                actual_row["volume"]
            ),

            "pred_open": float(
                prediction["open"]
            ),

            "pred_high": float(
                prediction["high"]
            ),

            "pred_low": float(
                prediction["low"]
            ),

            "pred_close": float(
                prediction["close"]
            ),

            "pred_volume": float(
                prediction["volume"]
            ),
        }

        results.append(
            result
        )

        print(
            "\nPrediction:"
        )

        print(
            f"Actual Close: "
            f"{result['actual_close']:.4f}"
        )

        print(
            f"Pred Close:   "
            f"{result['pred_close']:.4f}"
        )

        print(
            f"Error:        "
            f"{result['pred_close'] - result['actual_close']:.4f}"
        )

        # ----------------------------------------------------
        # Incrementally save results.
        #
        # If a long experiment is interrupted, predictions
        # completed so far are still available.
        # ----------------------------------------------------

        pd.DataFrame(
            results
        ).to_csv(
            CSV_OUTPUT,
            index=False,
        )

        # ====================================================
        # DELETE MODEL
        # ====================================================

        del model

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    # ========================================================
    # FINAL RESULTS DATAFRAME
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    if len(results_df) != total_predictions:

        raise RuntimeError(
            f"Prediction count mismatch: "
            f"expected {total_predictions}, "
            f"got {len(results_df)}."
        )

    # --------------------------------------------------------
    # Save final CSV
    # --------------------------------------------------------

    results_df.to_csv(
        CSV_OUTPUT,
        index=False,
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ROLLING FORECAST COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"Predictions: "
        f"{len(results_df)}"
    )

    print(
        f"Expected:    "
        f"{total_predictions}"
    )

    print(
        f"CSV: "
        f"{os.path.abspath(CSV_OUTPUT)}"
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics, directional_accuracy = (
        calculate_metrics(
            results_df
        )
    )

    print(
        "\nMETRICS"
    )

    print(
        "-" * 70
    )

    for column, values in (
        metrics.items()
    ):

        print(
            f"{column.upper():8s} | "
            f"MAE: {values['MAE']:.6f} | "
            f"RMSE: {values['RMSE']:.6f}"
        )

    print(
        "-" * 70
    )

    print(
        f"Close directional accuracy: "
        f"{directional_accuracy:.2f}%"
    )

    # ========================================================
    # HTML
    # ========================================================

    create_html(
        results_df
    )

    print(
        f"\nHTML: "
        f"{os.path.abspath(HTML_OUTPUT)}"
    )

    print(
        "\nDone."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()