#!/usr/bin/env python3

import os
import sys
import json
import time
import math
import random
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import optuna

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    "/home/soq/__shutupandbendover/yatin/nifty50 lora"
)

DATA_DIR = BASE_DIR / "NIFTY50_OHLCV"

RESULT_DIR = BASE_DIR / "company_lora_results"

PRED_DIR = RESULT_DIR / "predictions"
METRIC_DIR = RESULT_DIR / "metrics"
OPTUNA_DIR = RESULT_DIR / "optuna"
DASHBOARD_DIR = RESULT_DIR / "dashboards"

KRONOS_DIR = Path("/home/soq/Kronos")

TOKENIZER_PATH = Path(
    "/home/soq/__shutupandbendover/het-uchiha/weights/"
    "Kronos-Tokenizer-base"
)

MODEL_PATH = Path(
    "/home/soq/__shutupandbendover/het-uchiha/weights/"
    "Kronos-base"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

LOOKBACK = 118

FORECAST_HORIZON = 1

NUM_EPOCHS = 25

ROLLING_BATCH_SIZE = 8

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

OPTUNA_TRIALS = 3

MAX_CONTEXT = 512

TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 0.99

GRAD_CLIP = 3.0

LR_LOW = 1e-5
LR_HIGH = 5e-4

WEIGHT_DECAY_LOW = 0.0
WEIGHT_DECAY_HIGH = 1e-2

SEED = 42


# ============================================================
# SIX COMPANIES FOR DASHBOARDS
# ============================================================

PLOT_COMPANIES = {
    "RELIANCE",
    "TCS",
    "INFY",
    "HDFCBANK",
    "ICICIBANK",
    "SBIN",
}


# ============================================================
# KRONOS INPUT FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ============================================================
# SEED
# ============================================================

def seed_everything(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything()


# ============================================================
# IMPORT KRONOS
# ============================================================

sys.path.insert(
    0,
    str(KRONOS_DIR)
)

from model.kronos import (
    Kronos,
    KronosTokenizer,
    calc_time_stamps,
    sample_from_logits,
)


# ============================================================
# PRINT CONFIGURATION
# ============================================================

print("=" * 80)
print("NIFTY 50 KRONOS + LORA ROLLING FORECAST")
print("=" * 80)

print(f"DEVICE: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )

print()

print(
    f"Input features: {FEATURE_COLUMNS}"
)

print(
    f"Lookback: {LOOKBACK}"
)

print(
    f"Forecast Horizon: {FORECAST_HORIZON}"
)

print(
    f"Epochs: {NUM_EPOCHS}"
)

print(
    f"LoRA Rank: {LORA_RANK}"
)

print(
    f"LoRA Alpha: {LORA_ALPHA}"
)

print(
    f"LoRA Dropout: {LORA_DROPOUT}"
)

print(
    f"Rolling Batch Size: {ROLLING_BATCH_SIZE}"
)

print(
    f"Optuna Trials: {OPTUNA_TRIALS}"
)

print("=" * 80)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

for directory in [
    RESULT_DIR,
    PRED_DIR,
    METRIC_DIR,
    OPTUNA_DIR,
    DASHBOARD_DIR,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

print("\nLoading Kronos tokenizer...")

tokenizer = KronosTokenizer.from_pretrained(
    str(TOKENIZER_PATH)
)

tokenizer = tokenizer.to(
    DEVICE
)

tokenizer.eval()

print(
    "Tokenizer loaded."
)


# ============================================================
# LOAD KRONOS BASE
# ============================================================

print("\nLoading Kronos Base...")

base_model = Kronos.from_pretrained(
    str(MODEL_PATH)
)

base_model = base_model.to(
    DEVICE
)

# Freeze entire Base
for parameter in base_model.parameters():

    parameter.requires_grad = False


base_model.eval()

print(
    "Kronos Base loaded and frozen."
)


# ============================================================
# BATCHED LORA LINEAR
# ============================================================

class BatchedLoRALinear(nn.Module):

    def __init__(
        self,
        base_layer,
        max_batch,
        rank,
        alpha,
        dropout,
    ):

        super().__init__()

        self.base_layer = base_layer

        self.max_batch = max_batch

        self.rank = rank

        self.alpha = alpha

        self.scaling = alpha / rank

        self.dropout = nn.Dropout(
            dropout
        )

        in_features = (
            base_layer.in_features
        )

        out_features = (
            base_layer.out_features
        )

        device = (
            base_layer.weight.device
        )

        dtype = (
            base_layer.weight.dtype
        )

        self.lora_A = nn.Parameter(
            torch.empty(
                max_batch,
                rank,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        self.lora_B = nn.Parameter(
            torch.empty(
                max_batch,
                out_features,
                rank,
                device=device,
                dtype=dtype,
            )
        )

        self.reset_parameters()

        self.register_buffer(
            "active_adapter_indices",
            torch.arange(
                max_batch,
                device=device,
                dtype=torch.long,
            ),
            persistent=False,
        )

        # Base stays frozen
        for parameter in (
            self.base_layer.parameters()
        ):

            parameter.requires_grad = False


    def reset_parameters(self):

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

        nn.init.zeros_(
            self.lora_B
        )


    def reset_active_adapters(
        self,
        active_count,
    ):

        with torch.no_grad():

            nn.init.kaiming_uniform_(
                self.lora_A[
                    :active_count
                ],
                a=math.sqrt(5),
            )

            nn.init.zeros_(
                self.lora_B[
                    :active_count
                ]
            )


    def set_adapter_indices(
        self,
        indices,
    ):

        indices = torch.as_tensor(
            indices,
            device=self.lora_A.device,
            dtype=torch.long,
        )

        self.active_adapter_indices[
            :len(indices)
        ].copy_(indices)


    def forward(self, x):

        base_output = (
            self.base_layer(x)
        )

        batch_size = x.shape[0]

        indices = (
            self.active_adapter_indices[
                :batch_size
            ]
        )

        A = self.lora_A.index_select(
            0,
            indices,
        )

        B = self.lora_B.index_select(
            0,
            indices,
        )

        x_drop = self.dropout(x)

        low_rank = torch.einsum(
            "bti,bri->btr",
            x_drop,
            A,
        )

        delta = torch.einsum(
            "btr,bor->bto",
            low_rank,
            B,
        )

        return (
            base_output
            + self.scaling * delta
        )


# ============================================================
# INSERT LORA
# ============================================================

def insert_batched_lora(
    model,
):

    replaced = 0

    def recursive_replace(
        module
    ):

        nonlocal replaced

        for name, child in list(
            module.named_children()
        ):

            if isinstance(
                child,
                nn.Linear,
            ):

                wrapped = (
                    BatchedLoRALinear(
                        base_layer=child,
                        max_batch=ROLLING_BATCH_SIZE,
                        rank=LORA_RANK,
                        alpha=LORA_ALPHA,
                        dropout=LORA_DROPOUT,
                    )
                )

                setattr(
                    module,
                    name,
                    wrapped,
                )

                replaced += 1

            else:

                recursive_replace(
                    child
                )

    recursive_replace(
        model
    )

    return replaced


# ============================================================
# INSERT
# ============================================================

print(
    "\nCreating batched LoRA model..."
)

lora_model = base_model

num_lora_layers = (
    insert_batched_lora(
        lora_model
    )
)

lora_model = lora_model.to(
    DEVICE
)

print(
    f"LoRA inserted into "
    f"{num_lora_layers} Linear layers."
)


# ============================================================
# TRAINABLE PARAMETERS
# ============================================================

trainable_params = [
    parameter
    for parameter in lora_model.parameters()
    if parameter.requires_grad
]

total_params = sum(
    parameter.numel()
    for parameter in lora_model.parameters()
)

trainable_count = sum(
    parameter.numel()
    for parameter in trainable_params
)

print(
    f"Trainable parameters: "
    f"{trainable_count:,} / "
    f"{total_params:,} "
    f"({100.0 * trainable_count / total_params:.4f}%)"
)


# ============================================================
# FIND COMPANY FILE
# ============================================================

def find_company_file(
    company
):

    candidates = [

        DATA_DIR
        / f"{company}_OHLCV.parquet",

        DATA_DIR
        / f"{company}.parquet",

        DATA_DIR
        / f"{company.upper()}_OHLCV.parquet",

    ]

    for path in candidates:

        if path.exists():

            return path


    matches = list(
        DATA_DIR.glob(
            f"*{company}*.parquet"
        )
    )

    if matches:

        return matches[0]


    return None


# ============================================================
# LOAD COMPANY
# ============================================================

def load_company_data(
    company
):

    path = find_company_file(
        company
    )

    if path is None:

        raise FileNotFoundError(
            f"No parquet file found "
            f"for {company}"
        )


    df = pd.read_parquet(
        path
    )

    df.columns = [
        str(column)
        .strip()
        .lower()
        for column in df.columns
    ]


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    date_candidates = [
        "date",
        "datetime",
        "timestamp",
        "time",
    ]

    date_column = None

    for column in date_candidates:

        if column in df.columns:

            date_column = column

            break


    if date_column is None:

        raise ValueError(
            f"{company}: "
            f"No date column found. "
            f"Columns = "
            f"{list(df.columns)}"
        )


    df["date"] = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )


    # --------------------------------------------------------
    # OHLCV
    # --------------------------------------------------------

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{company}: "
            f"Missing columns: {missing}"
        )


    # --------------------------------------------------------
    # NUMERIC
    # --------------------------------------------------------

    for column in required:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )


    # --------------------------------------------------------
    # AMOUNT
    #
    # Kronos tokenizer expects 6 features.
    #
    # If amount exists, use it.
    #
    # Otherwise:
    #
    # amount = close * volume
    # --------------------------------------------------------

    if "amount" in df.columns:

        df["amount"] = pd.to_numeric(
            df["amount"],
            errors="coerce",
        )

    else:

        df["amount"] = (
            df["close"]
            * df["volume"]
        )


    # --------------------------------------------------------
    # SELECT EXACT SIX FEATURES
    # --------------------------------------------------------

    df = df[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]
    ].copy()


    # --------------------------------------------------------
    # CLEAN
    # --------------------------------------------------------

    df = df.dropna()

    df = (
        df.sort_values(
            "date"
        )
        .drop_duplicates(
            "date"
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    if len(df.columns) != 7:

        raise RuntimeError(
            f"{company}: "
            f"Unexpected number of columns."
        )


    return df


# ============================================================
# NORMALIZE
# ============================================================

def normalize_window(
    df_window
):

    values = (
        df_window[
            FEATURE_COLUMNS
        ]
        .astype(
            np.float32
        )
        .values
    )

    # Shape must be:
    #
    # [118, 6]

    if values.shape[1] != 6:

        raise RuntimeError(
            "Kronos input must have "
            f"6 features, got "
            f"{values.shape}"
        )


    mean = values.mean(
        axis=0
    )

    std = values.std(
        axis=0
    )

    std[
        std < 1e-6
    ] = 1.0


    normalized = (
        values - mean
    ) / std


    normalized = np.clip(
        normalized,
        -5.0,
        5.0,
    )


    return (
        normalized.astype(
            np.float32
        ),
        mean,
        std,
    )


# ============================================================
# DATAFRAME -> TENSOR
# ============================================================

def dataframe_to_tensor(
    normalized
):

    tensor = torch.tensor(
        normalized,
        dtype=torch.float32,
        device=DEVICE,
    )

    # [118, 6]
    #
    # becomes
    #
    # [1, 118, 6]

    tensor = tensor.unsqueeze(
        0
    )

    if tensor.shape[-1] != 6:

        raise RuntimeError(
            "Kronos tokenizer input "
            f"must have 6 features, "
            f"got {tensor.shape}"
        )

    return tensor


# ============================================================
# TIMESTAMP
# ============================================================

def create_stamp_tensor(
    timestamp_series
):

    timestamp_series = (
        pd.Series(
            pd.to_datetime(
                timestamp_series
            )
        )
        .reset_index(
            drop=True
        )
    )

    stamp_df = calc_time_stamps(
        timestamp_series
    )

    stamp = torch.tensor(
        stamp_df.values,
        dtype=torch.float32,
        device=DEVICE,
    )

    stamp = stamp.unsqueeze(
        0
    )

    return stamp


# ============================================================
# PREPARE ROLLING WINDOWS
# ============================================================

def prepare_rolling_windows(
    df
):

    windows = []

    total_windows = (
        len(df)
        - LOOKBACK
        - FORECAST_HORIZON
        + 1
    )

    if total_windows <= 0:

        return windows


    for i in range(
        total_windows
    ):

        start = i

        end = (
            i + LOOKBACK
        )

        target_index = end


        x_df = (
            df.iloc[
                start:end
            ]
            .copy()
        )


        target_df = (
            df.iloc[
                target_index:
                target_index
                + FORECAST_HORIZON
            ]
            .copy()
        )


        if len(target_df) < (
            FORECAST_HORIZON
        ):

            continue


        normalized, mean, std = (
            normalize_window(
                x_df
            )
        )


        x_tensor = (
            dataframe_to_tensor(
                normalized
            )
        )


        x_stamp = (
            create_stamp_tensor(
                x_df["date"]
            )
        )


        y_stamp = (
            create_stamp_tensor(
                target_df["date"]
            )
        )


        windows.append(
            {
                "x_tensor":
                    x_tensor,

                "x_stamp":
                    x_stamp,

                "y_stamp":
                    y_stamp,

                "mean":
                    mean,

                "std":
                    std,

                "input_df":
                    x_df,

                "target_df":
                    target_df,

                "target_date":
                    target_df.iloc[
                        0
                    ]["date"],

                "window_index":
                    i,
            }
        )


    return windows


# ============================================================
# TOKENIZE BATCH
# ============================================================

@torch.no_grad()
def tokenize_window_batch(
    window_batch
):

    x = torch.cat(
        [
            item["x_tensor"]
            for item in window_batch
        ],
        dim=0,
    )

    # IMPORTANT:
    #
    # x shape:
    # [batch, 118, 6]
    #
    # NOT DataFrame.

    if x.ndim != 3:

        raise RuntimeError(
            f"Unexpected tokenizer "
            f"input shape: {x.shape}"
        )


    if x.shape[-1] != 6:

        raise RuntimeError(
            f"Kronos requires "
            f"6 features. "
            f"Got {x.shape[-1]}"
        )


    token_s1, token_s2 = (
        tokenizer.encode(
            x,
            half=True,
        )
    )

    return (
        token_s1.to(DEVICE),
        token_s2.to(DEVICE),
    )


# ============================================================
# TRAIN ONE ROLLING BATCH
# ============================================================

def train_rolling_batch(
    window_batch,
    learning_rate,
    weight_decay,
):

    batch_size = len(
        window_batch
    )


    # --------------------------------------------------------
    # FRESH ADAPTERS
    # --------------------------------------------------------

    for module in (
        lora_model.modules()
    ):

        if isinstance(
            module,
            BatchedLoRALinear,
        ):

            module.set_adapter_indices(
                list(
                    range(
                        batch_size
                    )
                )
            )

            module.reset_active_adapters(
                batch_size
            )


    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    token_s1, token_s2 = (
        tokenize_window_batch(
            window_batch
        )
    )


    # --------------------------------------------------------
    # TEACHER FORCING
    # --------------------------------------------------------

    input_s1 = (
        token_s1[:, :-1]
    )

    input_s2 = (
        token_s2[:, :-1]
    )

    target_s1 = (
        token_s1[:, 1:]
    )

    target_s2 = (
        token_s2[:, 1:]
    )


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    x_stamp = torch.cat(
        [
            item["x_stamp"]
            for item in window_batch
        ],
        dim=0,
    )

    input_stamp = (
        x_stamp[:, :-1, :]
    )


    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=weight_decay,
    )


    losses = []

    lora_model.train()


    # --------------------------------------------------------
    # 25 EPOCHS
    # --------------------------------------------------------

    for epoch in range(
        NUM_EPOCHS
    ):

        optimizer.zero_grad(
            set_to_none=True
        )


        s1_logits, s2_logits = (
            lora_model(
                input_s1,
                input_s2,
                stamp=input_stamp,
                use_teacher_forcing=True,
                s1_targets=target_s1,
            )
        )


        loss, _, _ = (
            lora_model.head.compute_loss(
                s1_logits,
                s2_logits,
                target_s1,
                target_s2,
            )
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            trainable_params,
            max_norm=GRAD_CLIP,
        )


        optimizer.step()


        losses.append(
            float(
                loss.detach()
                .cpu()
            )
        )


    lora_model.eval()


    return {
        "loss":
            losses[-1],

        "loss_history":
            losses,
    }


# ============================================================
# FORECAST ONE WINDOW
# ============================================================

@torch.no_grad()
def forecast_one_window(
    window,
    adapter_index,
):

    # --------------------------------------------------------
    # CORRECT ADAPTER
    # --------------------------------------------------------

    for module in (
        lora_model.modules()
    ):

        if isinstance(
            module,
            BatchedLoRALinear,
        ):

            module.set_adapter_indices(
                [adapter_index]
            )


    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

    x_tensor = (
        window["x_tensor"]
    )


    token_s1, token_s2 = (
        tokenizer.encode(
            x_tensor,
            half=True,
        )
    )


    token_s1 = token_s1.to(
        DEVICE
    )

    token_s2 = token_s2.to(
        DEVICE
    )


    # --------------------------------------------------------
    # TIMESTAMP
    # --------------------------------------------------------

    x_stamp = (
        window["x_stamp"]
        .to(DEVICE)
    )

    y_stamp = (
        window["y_stamp"]
        .to(DEVICE)
    )


    full_stamp = torch.cat(
        [
            x_stamp,
            y_stamp,
        ],
        dim=1,
    )


    # --------------------------------------------------------
    # CONTEXT
    # --------------------------------------------------------

    initial_seq_len = (
        token_s1.shape[1]
    )


    context_start = max(
        0,
        initial_seq_len
        - MAX_CONTEXT,
    )


    input_s1 = (
        token_s1[
            :,
            context_start:
        ]
        .contiguous()
    )


    input_s2 = (
        token_s2[
            :,
            context_start:
        ]
        .contiguous()
    )


    current_stamp = (
        full_stamp[
            :,
            context_start:
            initial_seq_len,
            :
        ]
        .contiguous()
    )


    # --------------------------------------------------------
    # DECODE S1
    # --------------------------------------------------------

    s1_logits, context = (
        lora_model.decode_s1(
            input_s1,
            input_s2,
            current_stamp,
        )
    )


    s1_logits = (
        s1_logits[
            :,
            -1,
            :
        ]
    )


    sample_pre = (
        sample_from_logits(
            s1_logits,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            sample_logits=True,
        )
    )


    # --------------------------------------------------------
    # DECODE S2
    # --------------------------------------------------------

    s2_logits = (
        lora_model.decode_s2(
            context,
            sample_pre,
        )
    )


    s2_logits = (
        s2_logits[
            :,
            -1,
            :
        ]
    )


    sample_post = (
        sample_from_logits(
            s2_logits,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            sample_logits=True,
        )
    )


    # --------------------------------------------------------
    # APPEND TOKENS
    # --------------------------------------------------------

    full_s1 = torch.cat(
        [
            token_s1,
            sample_pre,
        ],
        dim=1,
    )


    full_s2 = torch.cat(
        [
            token_s2,
            sample_post,
        ],
        dim=1,
    )


    # --------------------------------------------------------
    # DECODE
    # --------------------------------------------------------

    final_context_start = max(
        0,
        full_s1.shape[1]
        - MAX_CONTEXT,
    )


    decode_s1_ids = (
        full_s1[
            :,
            final_context_start:
        ]
        .contiguous()
    )


    decode_s2_ids = (
        full_s2[
            :,
            final_context_start:
        ]
        .contiguous()
    )


    prediction = (
        tokenizer.decode(
            [
                decode_s1_ids,
                decode_s2_ids,
            ],
            half=True,
        )
    )


    prediction = (
        prediction
        .detach()
        .cpu()
        .numpy()
    )


    prediction = (
        prediction[
            0,
            -1,
            :
        ]
    )


    # --------------------------------------------------------
    # INVERSE NORMALIZATION
    # --------------------------------------------------------

    mean = (
        window["mean"]
    )

    std = (
        window["std"]
    )


    prediction = (
        prediction * std
        + mean
    )


    # We expect six outputs
    #
    # open
    # high
    # low
    # close
    # volume
    # amount

    if len(prediction) < 6:

        raise RuntimeError(
            "Kronos returned fewer "
            f"than 6 features: "
            f"{prediction.shape}"
        )


    return prediction.astype(
        np.float64
    )


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def optuna_objective(
    trial,
    windows,
):

    learning_rate = (
        trial.suggest_float(
            "learning_rate",
            LR_LOW,
            LR_HIGH,
            log=True,
        )
    )


    weight_decay = (
        trial.suggest_float(
            "weight_decay",
            WEIGHT_DECAY_LOW,
            WEIGHT_DECAY_HIGH,
        )
    )


    tuning_windows = (
        windows[
            :min(
                ROLLING_BATCH_SIZE,
                len(windows),
            )
        ]
    )


    result = train_rolling_batch(
        tuning_windows,
        learning_rate,
        weight_decay,
    )


    return result[
        "loss"
    ]


# ============================================================
# OPTUNA
# ============================================================

def tune_hyperparameters(
    company,
    windows,
):

    print(
        f"\n{company}: "
        f"Starting Optuna "
        f"({OPTUNA_TRIALS} trials)..."
    )


    study = optuna.create_study(
        direction="minimize",
        study_name=(
            f"{company}_LoRA"
        ),
    )


    study.optimize(
        lambda trial:
            optuna_objective(
                trial,
                windows,
            ),
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=False,
    )


    best_params = (
        study.best_params
    )


    with open(
        OPTUNA_DIR
        / f"{company}_best_params.json",
        "w",
    ) as f:

        json.dump(
            best_params,
            f,
            indent=4,
        )


    history = []

    for trial in study.trials:

        row = {
            "trial":
                trial.number,

            "value":
                trial.value,

            "state":
                str(trial.state),
        }

        row.update(
            trial.params
        )

        history.append(
            row
        )


    pd.DataFrame(
        history
    ).to_csv(
        OPTUNA_DIR
        / f"{company}_optuna_history.csv",
        index=False,
    )


    print(
        f"{company}: "
        f"Best Optuna params = "
        f"{best_params}"
    )


    return best_params


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual_close,
    predicted_close,
):

    actual_close = np.asarray(
        actual_close,
        dtype=np.float64,
    )

    predicted_close = np.asarray(
        predicted_close,
        dtype=np.float64,
    )


    if len(actual_close) < 2:

        return {}


    previous_actual = (
        actual_close[:-1]
    )


    actual_returns = (
        actual_close[1:]
        / previous_actual
        - 1.0
    )


    predicted_returns = (
        predicted_close[1:]
        / previous_actual
        - 1.0
    )


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mean_squared_error(
            actual_returns,
            predicted_returns,
        )
    )


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual_returns,
        predicted_returns,
    )


    # --------------------------------------------------------
    # MAPE
    # --------------------------------------------------------

    nonzero = (
        np.abs(
            actual_returns
        ) > 1e-12
    )


    if np.any(nonzero):

        mape = np.mean(
            np.abs(
                (
                    actual_returns[
                        nonzero
                    ]
                    - predicted_returns[
                        nonzero
                    ]
                )
                /
                actual_returns[
                    nonzero
                ]
            )
        )

    else:

        mape = np.nan


    # --------------------------------------------------------
    # R2
    # --------------------------------------------------------

    try:

        r2 = r2_score(
            actual_returns,
            predicted_returns,
        )

    except Exception:

        r2 = np.nan


    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    actual_direction = np.sign(
        actual_returns
    )

    predicted_direction = np.sign(
        predicted_returns
    )


    direction_accuracy = (
        np.mean(
            actual_direction
            == predicted_direction
        )
    )


    # --------------------------------------------------------
    # PEARSON
    # --------------------------------------------------------

    if (
        np.std(actual_returns)
        > 1e-12
        and
        np.std(predicted_returns)
        > 1e-12
    ):

        pearson = np.corrcoef(
            actual_returns,
            predicted_returns,
        )[0, 1]

    else:

        pearson = np.nan


    # --------------------------------------------------------
    # TOTAL RETURNS
    # --------------------------------------------------------

    actual_total_return = (
        actual_close[-1]
        /
        actual_close[0]
        - 1.0
    )


    predicted_total_return = (
        predicted_close[-1]
        /
        actual_close[0]
        - 1.0
    )


    return {

        "Predictions":
            int(len(actual_close)),

        "RMSE":
            float(
                rmse * 100.0
            ),

        "MAE":
            float(
                mae * 100.0
            ),

        "MAPE":
            float(
                mape * 100.0
            ),

        "R2":
            float(r2),

        "Direction Accuracy":
            float(
                direction_accuracy
                * 100.0
            ),

        "Actual Return":
            float(
                actual_total_return
                * 100.0
            ),

        "Predicted Return":
            float(
                predicted_total_return
                * 100.0
            ),

        "Pearson Correlation":
            float(pearson),
    }


# ============================================================
# SAVE PREDICTIONS
# ============================================================

def save_predictions(
    company,
    prediction_rows,
):

    prediction_df = pd.DataFrame(
        prediction_rows
    )


    path = (
        PRED_DIR
        / f"{company}_LoRA_predictions.csv"
    )


    prediction_df.to_csv(
        path,
        index=False,
    )


    return prediction_df


# ============================================================
# SAVE METRICS
# ============================================================

def save_company_metrics(
    company,
    metrics,
):

    path = (
        METRIC_DIR
        / f"{company}_LoRA_metrics.json"
    )


    with open(
        path,
        "w",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )


# ============================================================
# DASHBOARD
# ============================================================

def create_dashboard(
    company,
    prediction_df,
    metrics,
):

    dates = pd.to_datetime(
        prediction_df["date"]
    )


    fig = make_subplots(

        rows=4,

        cols=2,

        column_widths=[
            0.76,
            0.24,
        ],

        row_heights=[
            0.32,
            0.32,
            0.20,
            0.16,
        ],

        specs=[

            [
                {"type": "candlestick"},
                {"type": "table"},
            ],

            [
                {"type": "candlestick"},
                {"type": "table"},
            ],

            [
                {"type": "scatter"},
                {"type": "table"},
            ],

            [
                {"type": "table"},
                {"type": "table"},
            ],
        ],

        vertical_spacing=0.035,

        horizontal_spacing=0.025,
    )


    # ========================================================
    # ACTUAL
    # ========================================================

    fig.add_trace(

        go.Candlestick(

            x=dates,

            open=prediction_df[
                "actual_open"
            ],

            high=prediction_df[
                "actual_high"
            ],

            low=prediction_df[
                "actual_low"
            ],

            close=prediction_df[
                "actual_close"
            ],

            name="Actual OHLCV",
        ),

        row=1,

        col=1,
    )


    # ========================================================
    # ACTUAL INFO
    # ========================================================

    last = prediction_df.iloc[
        -1
    ]


    fig.add_trace(

        go.Table(

            header=dict(
                values=[
                    "Actual Latest"
                ]
            ),

            cells=dict(
                values=[

                    [
                        f"Open: "
                        f"{last['actual_open']:.2f}",

                        f"High: "
                        f"{last['actual_high']:.2f}",

                        f"Low: "
                        f"{last['actual_low']:.2f}",

                        f"Close: "
                        f"{last['actual_close']:.2f}",

                        f"Volume: "
                        f"{last['actual_volume']:,.0f}",
                    ]

                ]
            ),
        ),

        row=1,

        col=2,
    )


    # ========================================================
    # PREDICTED
    # ========================================================

    fig.add_trace(

        go.Candlestick(

            x=dates,

            open=prediction_df[
                "pred_open"
            ],

            high=prediction_df[
                "pred_high"
            ],

            low=prediction_df[
                "pred_low"
            ],

            close=prediction_df[
                "pred_close"
            ],

            name="Predicted OHLCV",
        ),

        row=2,

        col=1,
    )


    # ========================================================
    # PREDICTED INFO
    # ========================================================

    fig.add_trace(

        go.Table(

            header=dict(
                values=[
                    "Predicted Latest"
                ]
            ),

            cells=dict(
                values=[

                    [
                        f"Open: "
                        f"{last['pred_open']:.2f}",

                        f"High: "
                        f"{last['pred_high']:.2f}",

                        f"Low: "
                        f"{last['pred_low']:.2f}",

                        f"Close: "
                        f"{last['pred_close']:.2f}",

                        f"Volume: "
                        f"{last['pred_volume']:,.0f}",
                    ]

                ]
            ),
        ),

        row=2,

        col=2,
    )


    # ========================================================
    # RAW CLOSE ERROR
    # ========================================================

    raw_error = (
        prediction_df[
            "actual_close"
        ]
        -
        prediction_df[
            "pred_close"
        ]
    )


    fig.add_trace(

        go.Scatter(

            x=dates,

            y=raw_error,

            mode="lines",

            name=(
                "Actual Close - "
                "Predicted Close"
            ),
        ),

        row=3,

        col=1,
    )


    fig.add_trace(

        go.Table(

            header=dict(
                values=[
                    "Latest Raw Error"
                ]
            ),

            cells=dict(
                values=[
                    [
                        f"{raw_error.iloc[-1]:.4f}"
                    ]
                ]
            ),
        ),

        row=3,

        col=2,
    )


    # ========================================================
    # METRICS
    # ========================================================

    metric_names = [

        "RMSE (%)",

        "MAE (%)",

        "MAPE (%)",

        "R²",

        "Direction Accuracy (%)",

        "Actual Return (%)",

        "Predicted Return (%)",

        "Pearson Correlation",

    ]


    metric_values = [

        metrics.get(
            "RMSE",
            np.nan
        ),

        metrics.get(
            "MAE",
            np.nan
        ),

        metrics.get(
            "MAPE",
            np.nan
        ),

        metrics.get(
            "R2",
            np.nan
        ),

        metrics.get(
            "Direction Accuracy",
            np.nan
        ),

        metrics.get(
            "Actual Return",
            np.nan
        ),

        metrics.get(
            "Predicted Return",
            np.nan
        ),

        metrics.get(
            "Pearson Correlation",
            np.nan
        ),
    ]


    formatted = []

    for name, value in zip(
        metric_names,
        metric_values,
    ):

        if pd.isna(value):

            formatted.append(
                "NaN"
            )

        elif (
            name == "R²"
            or
            name == "Pearson Correlation"
        ):

            formatted.append(
                f"{value:.5f}"
            )

        else:

            formatted.append(
                f"{value:.4f}"
            )


    fig.add_trace(

        go.Table(

            header=dict(
                values=[
                    "Metric",
                    "Value",
                ]
            ),

            cells=dict(

                values=[
                    metric_names,
                    formatted,
                ]

            ),
        ),

        row=4,

        col=1,
    )


    # ========================================================
    # EXPERIMENT INFO
    # ========================================================

    fig.add_trace(

        go.Table(

            header=dict(
                values=[
                    "Experiment"
                ]
            ),

            cells=dict(

                values=[

                    [
                        "118-day rolling window",

                        "1-day forecast",

                        "25 epochs",

                        "LoRA rank = 8",

                        "LoRA alpha = 16",

                        "Dropout = 0.05",

                        "Frozen Kronos Base",

                        "Fresh LoRA per window",
                    ]

                ]

            ),
        ),

        row=4,

        col=2,
    )


    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(

        title=(
            f"{company} — "
            f"Kronos + LoRA "
            f"118-Day Rolling / "
            f"1-Day Forecast"
        ),

        height=1500,

        width=1700,

        xaxis_rangeslider_visible=False,

        xaxis2_rangeslider_visible=False,

        template="plotly_white",
    )


    fig.update_yaxes(
        title_text="Actual Price",
        row=1,
        col=1,
    )


    fig.update_yaxes(
        title_text="Predicted Price",
        row=2,
        col=1,
    )


    fig.update_yaxes(
        title_text=(
            "Actual Close - "
            "Predicted Close"
        ),
        row=3,
        col=1,
    )


    output_path = (
        DASHBOARD_DIR
        / f"{company}_LoRA_dashboard.html"
    )


    fig.write_html(
        output_path,
        include_plotlyjs=True,
    )


    print(
        f"{company}: Dashboard saved -> "
        f"{output_path}"
    )


# ============================================================
# PROCESS COMPANY
# ============================================================

def process_company(
    company
):

    start_time = time.time()


    print("\n")
    print("=" * 80)

    print(
        f"PROCESSING: {company}"
    )

    print("=" * 80)


    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = load_company_data(
        company
    )


    print(
        f"{company}: "
        f"{len(df):,} rows | "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )


    # --------------------------------------------------------
    # VERIFY SIX FEATURES
    # --------------------------------------------------------

    print(
        f"{company}: "
        f"Kronos features = "
        f"{FEATURE_COLUMNS}"
    )


    # --------------------------------------------------------
    # WINDOWS
    # --------------------------------------------------------

    windows = (
        prepare_rolling_windows(
            df
        )
    )


    print(
        f"{company}: "
        f"{len(windows):,} rolling windows"
    )


    if not windows:

        print(
            f"{company}: "
            f"Not enough data."
        )

        return None


    # --------------------------------------------------------
    # OPTUNA
    # --------------------------------------------------------

    best_params = (
        tune_hyperparameters(
            company,
            windows,
        )
    )


    learning_rate = (
        best_params[
            "learning_rate"
        ]
    )


    weight_decay = (
        best_params[
            "weight_decay"
        ]
    )


    # --------------------------------------------------------
    # ROLLING FORECAST
    # --------------------------------------------------------

    prediction_rows = []

    total_windows = (
        len(windows)
    )


    print(
        f"\n{company}: "
        f"Starting rolling LoRA training..."
    )


    rolling_start = time.time()


    for batch_start in range(
        0,
        total_windows,
        ROLLING_BATCH_SIZE,
    ):

        batch_end = min(
            batch_start
            + ROLLING_BATCH_SIZE,
            total_windows,
        )


        window_batch = (
            windows[
                batch_start:
                batch_end
            ]
        )


        print(
            f"\r{company}: "
            f"windows "
            f"{batch_start + 1:,}/"
            f"{total_windows:,}",
            end="",
            flush=True,
        )


        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        train_result = (
            train_rolling_batch(
                window_batch,
                learning_rate,
                weight_decay,
            )
        )


        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        for local_index, window in enumerate(
            window_batch
        ):

            prediction = (
                forecast_one_window(
                    window,
                    adapter_index=local_index,
                )
            )


            target = (
                window[
                    "target_df"
                ]
                .iloc[0]
            )


            prediction_rows.append(

                {

                    "date":
                        target["date"],

                    "actual_open":
                        float(
                            target["open"]
                        ),

                    "actual_high":
                        float(
                            target["high"]
                        ),

                    "actual_low":
                        float(
                            target["low"]
                        ),

                    "actual_close":
                        float(
                            target["close"]
                        ),

                    "actual_volume":
                        float(
                            target["volume"]
                        ),

                    "actual_amount":
                        float(
                            target["amount"]
                        ),

                    "pred_open":
                        float(
                            prediction[0]
                        ),

                    "pred_high":
                        float(
                            prediction[1]
                        ),

                    "pred_low":
                        float(
                            prediction[2]
                        ),

                    "pred_close":
                        float(
                            prediction[3]
                        ),

                    "pred_volume":
                        float(
                            prediction[4]
                        ),

                    "pred_amount":
                        float(
                            prediction[5]
                        ),

                    "window_index":
                        int(
                            window[
                                "window_index"
                            ]
                        ),

                    "training_loss":
                        float(
                            train_result[
                                "loss"
                            ]
                        ),
                }
            )


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    print()


    rolling_time = (
        time.time()
        - rolling_start
    )


    print(
        f"{company}: "
        f"Rolling training finished "
        f"in {rolling_time / 3600:.2f} hours"
    )


    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    prediction_df = (
        save_predictions(
            company,
            prediction_rows,
        )
    )


    prediction_df = (
        prediction_df
        .sort_values(
            "date"
        )
        .reset_index(
            drop=True
        )
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(

        prediction_df[
            "actual_close"
        ].values,

        prediction_df[
            "pred_close"
        ].values,
    )


    metrics[
        "Company"
    ] = company


    metrics[
        "Lookback"
    ] = LOOKBACK


    metrics[
        "Forecast Horizon"
    ] = FORECAST_HORIZON


    metrics[
        "Epochs"
    ] = NUM_EPOCHS


    metrics[
        "LoRA Rank"
    ] = LORA_RANK


    metrics[
        "LoRA Alpha"
    ] = LORA_ALPHA


    metrics[
        "LoRA Dropout"
    ] = LORA_DROPOUT


    save_company_metrics(
        company,
        metrics,
    )


    print("\n")
    print(
        f"{company} METRICS"
    )

    print("-" * 60)


    for key, value in metrics.items():

        if isinstance(
            value,
            float,
        ):

            print(
                f"{key:25s}: "
                f"{value:.6f}"
            )

        else:

            print(
                f"{key:25s}: "
                f"{value}"
            )


    # --------------------------------------------------------
    # DASHBOARD
    # --------------------------------------------------------

    if company in PLOT_COMPANIES:

        create_dashboard(
            company,
            prediction_df,
            metrics,
        )


    elapsed = (
        time.time()
        - start_time
    )


    print(
        f"\n{company}: "
        f"Total time = "
        f"{elapsed / 3600:.2f} hours"
    )


    return metrics


# ============================================================
# GET COMPANIES
# ============================================================

def get_all_companies():

    files = sorted(
        DATA_DIR.glob(
            "*.parquet"
        )
    )


    companies = []


    for file in files:

        name = file.stem


        if name.endswith(
            "_OHLCV"
        ):

            name = name[
                :-len("_OHLCV")
            ]


        companies.append(
            name.upper()
        )


    return sorted(
        list(
            set(
                companies
            )
        )
    )


# ============================================================
# SAVE ALL METRICS
# ============================================================

def save_all_metrics(
    all_metrics
):

    if not all_metrics:

        return


    df = pd.DataFrame(
        all_metrics
    )


    if "Company" in df.columns:

        columns = [
            "Company"
        ] + [
            column
            for column in df.columns
            if column != "Company"
        ]

        df = df[
            columns
        ]


    csv_path = (
        METRIC_DIR
        / "ALL_50_NIFTY_LoRA_metrics.csv"
    )


    json_path = (
        METRIC_DIR
        / "ALL_50_NIFTY_LoRA_metrics.json"
    )


    df.to_csv(
        csv_path,
        index=False,
    )


    with open(
        json_path,
        "w",
    ) as f:

        json.dump(
            all_metrics,
            f,
            indent=4,
        )


    matrix_metrics = [

        "RMSE",

        "MAE",

        "MAPE",

        "R2",

        "Direction Accuracy",

        "Actual Return",

        "Predicted Return",

        "Pearson Correlation",

    ]


    available = [

        metric
        for metric in matrix_metrics
        if metric in df.columns

    ]


    if available:

        matrix = (
            df.set_index(
                "Company"
            )[available]
        )


        matrix.to_csv(

            METRIC_DIR
            / "NIFTY50_LoRA_metrics_matrix.csv"

        )


        heatmap = go.Figure(

            data=go.Heatmap(

                z=matrix.values,

                x=matrix.columns,

                y=matrix.index,

                hoverongaps=False,

            )

        )


        heatmap.update_layout(

            title=(
                "NIFTY 50 "
                "Kronos + LoRA "
                "Metrics Matrix"
            ),

            height=1500,

            width=1500,

        )


        heatmap.write_html(

            METRIC_DIR
            / "NIFTY50_LoRA_metrics_heatmap.html"

        )


