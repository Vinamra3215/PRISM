#!/usr/bin/env python3
import sys
import json
import math
import time
import random
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import optuna

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, pearsonr
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ============================================================
# BLOCK 1 — KRONOS + LoRA ONLY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# YOUR ACTUAL DATA LOCATION
DATA_DIR = Path("/home/soq/NIFTY50_OHLCV")

RESULT_DIR = BASE_DIR / "company_lora_results"
PRED_DIR = RESULT_DIR / "predictions"
METRICS_DIR = RESULT_DIR / "metrics"
OPTUNA_DIR = RESULT_DIR / "optuna"
CHECKPOINT_DIR = RESULT_DIR / "checkpoints"
EMBED_DIR = RESULT_DIR / "embeddings"
DASHBOARD_DIR = RESULT_DIR / "dashboards"

# Kronos
KRONOS_PATH = str(Path(__file__).resolve().parent)
TOKENIZER_PATH = str(Path(__file__).resolve().parent / "weights" / "Kronos-Tokenizer-base")
MODEL_PATH = str(Path(__file__).resolve().parent / "weights" / "Kronos-base")

# ------------------------------------------------------------
# REQUIRED ROLLING SETUP
# ------------------------------------------------------------
LOOKBACK = 118
FORECAST_HORIZON = 1

# LoRA
LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# Training
NUM_EPOCHS = 25
BATCH_SIZE = 8
GRAD_CLIP = 3.0
MAX_CONTEXT = 512

# Optuna
# Kept small for practical runtime.
OPTUNA_TRIALS = 3
OPTUNA_EPOCHS = 5
LR_LOW = 1e-5
LR_HIGH = 5e-4
WEIGHT_DECAY_LOW = 0.0
WEIGHT_DECAY_HIGH = 0.01

# Early stopping
EARLY_STOP_PATIENCE = 15

# Forecasting
# We use deterministic argmax decoding for reproducible Block 1 results.
TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 0.99

# ------------------------------------------------------------
# BLOCK 3 CHRONOLOGY
# ------------------------------------------------------------
TRAIN_START = pd.Timestamp("2022-01-01")
TRAIN_END = pd.Timestamp("2022-12-31")

OPTUNA_VAL_START = pd.Timestamp("2023-01-01")
OPTUNA_VAL_END = pd.Timestamp("2023-07-31")

EARLY_STOP_START = pd.Timestamp("2023-08-01")
EARLY_STOP_END = pd.Timestamp("2023-12-31")

TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2099-12-31")

# Dashboards only — ALL 50 companies are still processed.
PLOT_COMPANIES = set()

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
    else "cpu"
)

for directory in [
    RESULT_DIR,
    PRED_DIR,
    METRICS_DIR,
    OPTUNA_DIR,
    CHECKPOINT_DIR,
    EMBED_DIR,
    DASHBOARD_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, KRONOS_PATH)

from model.kronos import (
    Kronos,
    KronosTokenizer,
)


# ============================================================
# SEED
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)


# ============================================================
# LoRA
# ============================================================

class LoRALinear(nn.Module):
    """
    Frozen original Linear layer + trainable low-rank LoRA update.

    output =
        original_linear(x)
        + (alpha / rank) * B(A(dropout(x)))
    """

    def __init__(
        self,
        base_layer,
        rank=8,
        alpha=16,
        dropout=0.05,
    ):
        super().__init__()

        if not isinstance(base_layer, nn.Linear):
            raise TypeError(
                "LoRALinear requires an nn.Linear layer."
            )

        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

        # Freeze Kronos original parameters.
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                base_layer.in_features,
            )
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                rank,
            )
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

    def forward(self, x):
        base_output = self.base(x)

        low_rank = self.dropout(x)
        low_rank = F.linear(
            low_rank,
            self.lora_A,
        )
        low_rank = F.linear(
            low_rank,
            self.lora_B,
        )

        return (
            base_output
            + self.scaling * low_rank
        )


def replace_linear_with_lora(module):
    """
    Replace every nn.Linear inside Kronos with LoRALinear.
    """

    for name, child in list(
        module.named_children()
    ):
        if isinstance(child, LoRALinear):
            continue

        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                LoRALinear(
                    child,
                    rank=LORA_RANK,
                    alpha=LORA_ALPHA,
                    dropout=LORA_DROPOUT,
                ),
            )
        else:
            replace_linear_with_lora(child)


def freeze_base_and_enable_lora(model):
    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True


def lora_parameters(model):
    return [
        parameter
        for name, parameter in model.named_parameters()
        if (
            (
                "lora_A" in name
                or "lora_B" in name
            )
            and parameter.requires_grad
        )
    ]


def reset_lora(model):
    """
    Reset only LoRA.
    Kronos Base remains frozen.
    """

    for module in model.modules():
        if isinstance(module, LoRALinear):
            nn.init.kaiming_uniform_(
                module.lora_A,
                a=math.sqrt(5),
            )
            nn.init.zeros_(
                module.lora_B
            )


def save_lora_checkpoint(
    model,
    path,
    company,
    best_epoch,
    best_val,
    learning_rate,
    weight_decay,
):
    checkpoint = {
        "company": company,
        "best_epoch": best_epoch,
        "best_val_return_rmse": best_val,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "lookback": LOOKBACK,
        "forecast_horizon": FORECAST_HORIZON,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_state": {
            name: parameter.detach().cpu()
            for name, parameter
            in model.named_parameters()
            if (
                "lora_A" in name
                or "lora_B" in name
            )
        },
    }

    torch.save(
        checkpoint,
        path,
    )


def load_lora_checkpoint(
    model,
    path,
):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    own_state = model.state_dict()

    for name, value in checkpoint[
        "lora_state"
    ].items():
        if name not in own_state:
            raise KeyError(
                f"Checkpoint parameter not found: {name}"
            )

        own_state[name].copy_(
            value.to(DEVICE)
        )

    model.load_state_dict(
        own_state,
        strict=False,
    )

    return checkpoint


# ============================================================
# DATA
# ============================================================

def company_path(company):
    return DATA_DIR / (
        f"{company}_OHLCV.parquet"
    )


def get_all_companies():
    files = sorted(
        DATA_DIR.glob(
            "*_OHLCV.parquet"
        )
    )

    companies = []

    for file in files:
        name = file.stem

        if name.endswith("_OHLCV"):
            name = name[
                :-len("_OHLCV")
            ]

        companies.append(
            name.upper()
        )

    return sorted(
        set(companies)
    )


def load_company_data(company):
    path = company_path(company)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}"
        )

    df = pd.read_parquet(
        path
    ).copy()

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    if "Date" not in df.columns:
        raise ValueError(
            f"{path} has no Date column."
        )

    df["date"] = (
        pd.to_datetime(
            df["Date"]
        )
        .dt.tz_localize(None)
    )

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

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
            f"{company}: missing {missing}"
        )

    df = df[
        [
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    for column in required:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = (
        df.dropna()
        .drop_duplicates(
            subset=["date"]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Kronos needs six features:
    # open, high, low, close, volume, amount.
    #
    # The files do not contain amount, so construct it as:
    # volume * mean(OHLC).
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

    return df


def calc_time_stamps(timestamps):
    timestamps = pd.Series(
        pd.to_datetime(timestamps)
    ).reset_index(drop=True)

    stamp = pd.DataFrame(
        {
            "minute":
                timestamps.dt.minute,

            "hour":
                timestamps.dt.hour,

            "weekday":
                timestamps.dt.weekday,

            "day":
                timestamps.dt.day,

            "month":
                timestamps.dt.month,
        }
    )

    return stamp.to_numpy(
        dtype=np.float32
    )


def make_window(
    df,
    target_index,
):
    """
    Exactly:

        previous 118 trading days
                    ↓
               predict day t

    Target/current day is NOT included in x.
    """

    x = df.iloc[
        target_index - LOOKBACK:
        target_index
    ].copy()

    target = df.iloc[
        target_index
    ]

    raw_values = x[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    mean = raw_values.mean(
        axis=0
    )

    std = raw_values.std(
        axis=0
    )

    normalized = (
        raw_values
        - mean
    ) / (
        std + 1e-5
    )

    normalized = np.clip(
        normalized,
        -5.0,
        5.0,
    )

    return {
        "x":
            normalized,

        "raw_x":
            x,

        "x_stamp":
            calc_time_stamps(
                x["date"]
            ),

        "timestamps":
            x["date"].reset_index(
                drop=True
            ),

        "target_date":
            pd.Timestamp(
                target["date"]
            ),

        "target":
            target.to_dict(),

        "previous_close":
            float(
                x.iloc[-1]["close"]
            ),

        "window_start":
            pd.Timestamp(
                x.iloc[0]["date"]
            ),

        "window_end":
            pd.Timestamp(
                x.iloc[-1]["date"]
            ),

        "mean":
            mean,

        "std":
            std,

        "window_index":
            target_index,
    }


def build_windows(df):
    windows = []

    # First valid target requires 118 previous rows.
    for target_index in range(
        LOOKBACK,
        len(df),
    ):
        windows.append(
            make_window(
                df,
                target_index,
            )
        )

    return windows


def select_windows(
    windows,
    start_date,
    end_date,
):
    return [
        window
        for window in windows
        if (
            start_date
            <= window["target_date"]
            <= end_date
        )
    ]


# ============================================================
# LOAD KRONOS + LoRA
# ============================================================

def load_lora_model():
    print(
        "Loading Kronos tokenizer..."
    )

    tokenizer = (
        KronosTokenizer
        .from_pretrained(
            TOKENIZER_PATH
        )
    )

    print(
        "Loading Kronos Base..."
    )

    model = (
        Kronos
        .from_pretrained(
            MODEL_PATH
        )
    )

    print(
        "Injecting LoRA..."
    )

    replace_linear_with_lora(
        model
    )

    freeze_base_and_enable_lora(
        model
    )

    model = model.to(
        DEVICE
    )

    tokenizer = tokenizer.to(
        DEVICE
    )

    model.eval()

    return (
        model,
        tokenizer,
    )


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize_batch(
    tokenizer,
    windows,
):
    x = np.stack(
        [
            window["x"]
            for window in windows
        ],
        axis=0,
    )

    x = torch.from_numpy(
        x
    ).to(DEVICE)

    token_s1, token_s2 = (
        tokenizer.encode(
            x,
            half=True,
        )
    )

    stamps = np.stack(
        [
            window["x_stamp"]
            for window in windows
        ],
        axis=0,
    )

    stamps = torch.from_numpy(
        stamps
    ).to(DEVICE)

    return (
        token_s1.long(),
        token_s2.long(),
        stamps,
    )


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(
    model,
    tokenizer,
    windows,
    optimizer,
):
    model.train()

    indices = np.random.permutation(
        len(windows)
    )

    losses = []

    for start in range(
        0,
        len(indices),
        BATCH_SIZE,
    ):
        batch_indices = indices[
            start:
            start + BATCH_SIZE
        ]

        batch = [
            windows[index]
            for index in batch_indices
        ]

        (
            token_s1,
            token_s2,
            stamps,
        ) = tokenize_batch(
            tokenizer,
            batch,
        )

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

        input_stamp = (
            stamps[:, :-1, :]
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            s1_logits,
            s2_logits,
        ) = model(
            input_s1,
            input_s2,
            stamp=input_stamp,
            use_teacher_forcing=True,
            s1_targets=target_s1,
        )

        loss, _, _ = (
            model.head.compute_loss(
                s1_logits,
                s2_logits,
                target_s1,
                target_s2,
            )
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            lora_parameters(model),
            max_norm=GRAD_CLIP,
        )

        optimizer.step()

        losses.append(
            float(
                loss.detach()
                .cpu()
            )
        )

    if not losses:
        return np.nan

    return float(
        np.mean(losses)
    )


# ============================================================
# ONE-DAY FORECAST
# ============================================================

@torch.no_grad()
def forecast_window(
    model,
    tokenizer,
    window,
):
    """
    Deterministic one-step Kronos forecast.

    Input:
        exactly 118 previous actual trading days.

    Output:
        exactly one next-day OHLCV+amount prediction.
    """

    model.eval()

    x = window[
        "raw_x"
    ][
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    mean = x.mean(
        axis=0
    )

    std = x.std(
        axis=0
    )

    x_normalized = (
        x - mean
    ) / (
        std + 1e-5
    )

    x_normalized = np.clip(
        x_normalized,
        -5.0,
        5.0,
    )

    x_tensor = (
        torch.from_numpy(
            x_normalized[
                None,
                :,
                :
            ]
        )
        .to(DEVICE)
    )

    stamp = (
        torch.from_numpy(
            window[
                "x_stamp"
            ][
                None,
                :,
                :
            ]
        )
        .to(DEVICE)
    )

    token_s1, token_s2 = (
        tokenizer.encode(
            x_tensor,
            half=True,
        )
    )

    # Predict s1 for next token.
    (
        s1_logits,
        context,
    ) = model.decode_s1(
        token_s1,
        token_s2,
        stamp=stamp,
    )

    next_s1 = torch.argmax(
        s1_logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    # Predict s2 conditioned on next s1.
    s2_logits = model.decode_s2(
        context,
        next_s1,
    )

    next_s2 = torch.argmax(
        s2_logits[:, -1, :],
        dim=-1,
        keepdim=True,
    )

    full_s1 = torch.cat(
        [
            token_s1,
            next_s1,
        ],
        dim=1,
    )

    full_s2 = torch.cat(
        [
            token_s2,
            next_s2,
        ],
        dim=1,
    )

    decoded = tokenizer.decode(
        [
            full_s1,
            full_s2,
        ],
        half=True,
    )

    predicted_normalized = (
        decoded[
            :,
            -1,
            :
        ]
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    prediction = (
        predicted_normalized
        * (std + 1e-5)
        + mean
    )

    return prediction.astype(
        np.float64
    )


# ============================================================
# VALIDATION
# ============================================================

def evaluate_windows(
    model,
    tokenizer,
    windows,
):
    if not windows:
        return (
            np.nan,
            [],
        )

    rows = []

    for window in windows:
        prediction = forecast_window(
            model,
            tokenizer,
            window,
        )

        actual_close = float(
            window[
                "target"
            ]["close"]
        )

        previous_close = float(
            window[
                "previous_close"
            ]
        )

        predicted_close = float(
            prediction[3]
        )

        rows.append(
            {
                "date":
                    window[
                        "target_date"
                    ],

                "actual_return":
                    actual_close
                    / previous_close
                    - 1.0,

                "pred_return":
                    predicted_close
                    / previous_close
                    - 1.0,

                "actual_close":
                    actual_close,

                "pred_close":
                    predicted_close,
            }
        )

    validation_df = pd.DataFrame(
        rows
    )

    rmse = math.sqrt(
        mean_squared_error(
            validation_df[
                "actual_return"
            ],
            validation_df[
                "pred_return"
            ],
        )
    )

    return (
        float(rmse),
        rows,
    )


# ============================================================
# OPTUNA
# ============================================================

def optuna_tune(
    model,
    tokenizer,
    train_windows,
    validation_windows,
    company,
):
    print()
    print(
        f"{company}: "
        f"Optuna "
        f"({OPTUNA_TRIALS} trials)"
    )

    study = optuna.create_study(
        direction="minimize",
        study_name=(
            f"{company}_Block1_LoRA"
        ),
    )

    initial_lora = {
        name:
            parameter
            .detach()
            .cpu()
            .clone()

        for name, parameter
        in model.named_parameters()

        if (
            "lora_A" in name
            or "lora_B" in name
        )
    }

    def objective(trial):
        # Every trial starts from the same LoRA initialization.
        state = model.state_dict()

        for name, value in (
            initial_lora.items()
        ):
            state[name].copy_(
                value.to(DEVICE)
            )

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

        optimizer = (
            torch.optim.AdamW(
                lora_parameters(model),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        )

        for _ in range(
            OPTUNA_EPOCHS
        ):
            train_one_epoch(
                model,
                tokenizer,
                train_windows,
                optimizer,
            )

        validation_rmse, _ = (
            evaluate_windows(
                model,
                tokenizer,
                validation_windows,
            )
        )

        print(
            f"  trial={trial.number} | "
            f"lr={learning_rate:.3e} | "
            f"wd={weight_decay:.6f} | "
            f"val_return_RMSE="
            f"{validation_rmse:.6f}"
        )

        return validation_rmse

    study.optimize(
        objective,
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=False,
    )

    best_parameters = (
        study.best_params
    )

    with open(
        OPTUNA_DIR
        / f"{company}_best_params.json",
        "w",
    ) as file:
        json.dump(
            best_parameters,
            file,
            indent=2,
        )

    study.trials_dataframe().to_csv(
        OPTUNA_DIR
        / f"{company}_optuna_history.csv",
        index=False,
    )

    print(
        f"{company}: "
        f"Best parameters = "
        f"{best_parameters}"
    )

    return best_parameters


# ============================================================
# EARLY STOPPING
# ============================================================

def train_with_early_stopping(
    model,
    tokenizer,
    train_windows,
    early_stop_windows,
    learning_rate,
    weight_decay,
    company,
):
    # Start final training from fresh LoRA.
    reset_lora(model)

    optimizer = (
        torch.optim.AdamW(
            lora_parameters(model),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    )

    best_rmse = float(
        "inf"
    )

    best_epoch = 0
    bad_epochs = 0

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"{company}_LoRA_best.pt"
    )

    history = []

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        train_loss = (
            train_one_epoch(
                model,
                tokenizer,
                train_windows,
                optimizer,
            )
        )

        validation_rmse, _ = (
            evaluate_windows(
                model,
                tokenizer,
                early_stop_windows,
            )
        )

        history.append(
            {
                "epoch":
                    epoch,

                "train_loss":
                    train_loss,

                "early_stop_return_rmse":
                    validation_rmse,
            }
        )

        print(
            f"{company}: "
            f"epoch {epoch:02d}/"
            f"{NUM_EPOCHS} | "
            f"train_loss="
            f"{train_loss:.6f} | "
            f"Aug-Dec_RMSE="
            f"{validation_rmse:.6f}"
        )

        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
            best_epoch = epoch
            bad_epochs = 0

            save_lora_checkpoint(
                model,
                checkpoint_path,
                company,
                best_epoch,
                best_rmse,
                learning_rate,
                weight_decay,
            )
        else:
            bad_epochs += 1

        if (
            bad_epochs
            >= EARLY_STOP_PATIENCE
        ):
            print(
                f"{company}: "
                f"Early stopping."
            )
            break

    pd.DataFrame(
        history
    ).to_csv(
        METRICS_DIR
        / f"{company}_training_history.csv",
        index=False,
    )

    load_lora_checkpoint(
        model,
        checkpoint_path,
    )

    return (
        checkpoint_path,
        best_epoch,
        best_rmse,
    )


# ============================================================
# EMBEDDINGS
# ============================================================

@torch.no_grad()
def extract_embedding(
    model,
    tokenizer,
    window,
):
    """
    Last hidden state from the 118-day context.

    Expected Block 3 embedding size:
        768
    """

    model.eval()

    x = torch.from_numpy(
        window["x"][None]
    ).to(DEVICE)

    token_s1, token_s2 = (
        tokenizer.encode(
            x,
            half=True,
        )
    )

    stamp = torch.from_numpy(
        window[
            "x_stamp"
        ][None]
    ).to(DEVICE)

    _, context = (
        model.decode_s1(
            token_s1,
            token_s2,
            stamp=stamp,
        )
    )

    embedding = (
        context[
            :,
            -1,
            :
        ]
        .squeeze(0)
        .float()
        .cpu()
        .numpy()
    )

    if embedding.shape[0] != 832:
        raise RuntimeError(
            "Expected an 832-d Kronos embedding from Kronos-base, "
            f"got {embedding.shape[0]}."
        )

    return embedding


def save_all_embeddings(
    model,
    tokenizer,
    windows,
    company,
):
    """
    Extract embeddings for every usable rolling day
    available from the supplied data.

    No model training occurs here.
    """

    rows = []

    total = len(
        windows
    )

    for number, window in enumerate(
        windows,
        1,
    ):
        embedding = (
            extract_embedding(
                model,
                tokenizer,
                window,
            )
        )

        rows.append(
            [
                window[
                    "target_date"
                ],
                *embedding.tolist(),
            ]
        )

        if (
            number % 50 == 0
            or number == total
        ):
            print(
                f"\r{company}: "
                f"embeddings "
                f"{number:,}/"
                f"{total:,}",
                end="",
                flush=True,
            )

    print()

    columns = [
        "date"
    ] + [
        f"embedding_{i}"
        for i in range(832)
    ]

    embedding_df = pd.DataFrame(
        rows,
        columns=columns,
    )

    output_path = (
        EMBED_DIR
        / f"{company}_kronos_lora_embeddings.parquet"
    )

    embedding_df.to_parquet(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# WALK-FORWARD ONE-DAY TEST
# ============================================================

def walk_forward_test(
    model,
    tokenizer,
    windows,
    company,
):
    """
    For every test day:

        use previous 118 actual trading days
                    ↓
                predict 1 day
                    ↓
                move forward 1 day
                    ↓
                repeat

    IMPORTANT:
        LoRA weights are NOT retrained during the test.
        The best checkpoint is fixed.
    """

    test_windows = [
        window
        for window in windows
        if (
            TEST_START
            <= window["target_date"]
            <= TEST_END
        )
    ]

    rows = []

    total = len(
        test_windows
    )

    for number, window in enumerate(
        test_windows,
        1,
    ):
        prediction = forecast_window(
            model,
            tokenizer,
            window,
        )

        target = (
            window["target"]
        )

        previous_close = float(
            window[
                "previous_close"
            ]
        )

        actual_close = float(
            target["close"]
        )

        predicted_close = float(
            prediction[3]
        )

        rows.append(
            {
                "company":
                    company,

                "date":
                    window[
                        "target_date"
                    ],

                "window_start":
                    window[
                        "window_start"
                    ],

                "window_end":
                    window[
                        "window_end"
                    ],

                "previous_close":
                    previous_close,

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
                    actual_close,

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
                    predicted_close,

                "pred_volume":
                    float(
                        prediction[4]
                    ),

                "pred_amount":
                    float(
                        prediction[5]
                    ),

                "actual_return":
                    actual_close
                    / previous_close
                    - 1.0,

                "pred_return":
                    predicted_close
                    / previous_close
                    - 1.0,
            }
        )

        if (
            number % 25 == 0
            or number == total
        ):
            print(
                f"\r{company}: "
                f"test forecast "
                f"{number:,}/"
                f"{total:,}",
                end="",
                flush=True,
            )

    print()

    return pd.DataFrame(
        rows
    )


# ============================================================
# COMPANY METRICS
# ============================================================

def calculate_company_metrics(
    prediction_df,
):
    if prediction_df.empty:
        return {}

    actual = (
        prediction_df[
            "actual_return"
        ].to_numpy()
    )

    predicted = (
        prediction_df[
            "pred_return"
        ].to_numpy()
    )

    metrics = {
        "N":
            len(prediction_df),

        "RMSE_return":
            math.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            ),

        "MAE_return":
            mean_absolute_error(
                actual,
                predicted,
            ),

        "R2_return":
            r2_score(
                actual,
                predicted,
            ),

        "Direction_Accuracy":
            float(
                np.mean(
                    np.sign(actual)
                    == np.sign(predicted)
                )
            ),
    }

    if len(actual) > 1:
        metrics[
            "Pearson_return"
        ] = float(
            pearsonr(
                actual,
                predicted,
            ).statistic
        )

    return metrics


# ============================================================
# CROSS-SECTIONAL BLOCK 1 METRICS
# ============================================================

def calculate_cross_sectional_metrics(
    all_predictions,
):
    df = all_predictions.copy()

    df["date"] = pd.to_datetime(
        df["date"]
    )

    daily_rows = []
    top10_hit = []
    quantile_rows = []

    for date, group in df.groupby(
        "date"
    ):
        group = group.dropna(
            subset=[
                "actual_return",
                "pred_return",
            ]
        ).copy()

        if len(group) < 3:
            continue

        rank_ic = spearmanr(
            group[
                "pred_return"
            ],
            group[
                "actual_return"
            ],
        ).statistic

        group = group.sort_values(
            "pred_return"
        )

        n = min(
            10,
            len(group) // 2,
        )

        if n > 0:
            bottom_return = (
                group.iloc[:n][
                    "actual_return"
                ].mean()
            )

            top_return = (
                group.iloc[-n:][
                    "actual_return"
                ].mean()
            )

            long_short = (
                top_return
                - bottom_return
            )

            top10_hit.append(
                float(
                    top_return > 0
                )
            )
        else:
            long_short = np.nan

        daily_rows.append(
            {
                "date":
                    date,

                "RankIC":
                    rank_ic,

                "LongShort":
                    long_short,

                "n_stocks":
                    len(group),
            }
        )

        # Five predicted-return quantiles.
        try:
            group["quantile"] = pd.qcut(
                group[
                    "pred_return"
                ],
                q=5,
                labels=False,
                duplicates="drop",
            )

            quantile_return = (
                group
                .groupby(
                    "quantile",
                    observed=True,
                )[
                    "actual_return"
                ]
                .mean()
            )

            if len(
                quantile_return
            ) >= 3:
                quantile_rows.append(
                    {
                        "date":
                            date,

                        "Q1":
                            float(
                                quantile_return.iloc[0]
                            ),

                        "Q5":
                            float(
                                quantile_return.iloc[-1]
                            ),
                    }
                )
        except Exception:
            pass

    daily_df = pd.DataFrame(
        daily_rows
    )

    quantile_df = pd.DataFrame(
        quantile_rows
    )

    if daily_df.empty:
        return (
            {},
            daily_df,
            quantile_df,
        )

    ic = daily_df[
        "RankIC"
    ].dropna()

    if (
        len(ic) > 1
        and ic.std(ddof=1) > 0
    ):
        icir = (
            ic.mean()
            / ic.std(ddof=1)
        )
    else:
        icir = np.nan

    if not quantile_df.empty:
        monotonicity = float(
            np.mean(
                quantile_df[
                    "Q5"
                ].to_numpy()
                >=
                quantile_df[
                    "Q1"
                ].to_numpy()
            )
        )
    else:
        monotonicity = np.nan

    metrics = {
        "Days":
            len(daily_df),

        "Mean_RankIC":
            float(ic.mean()),

        "ICIR":
            float(icir),

        "Mean_LongShort":
            float(
                daily_df[
                    "LongShort"
                ].mean()
            ),

        "Positive_IC_Days":
            float(
                (ic > 0).mean()
            ),

        "Cumulative_IC":
            float(
                ic.sum()
            ),

        "Top10_Positive_Return_HitRate":
            float(
                np.mean(top10_hit)
            )
            if top10_hit
            else np.nan,

        "Quantile_Monotonicity":
            monotonicity,
    }

    return (
        metrics,
        daily_df,
        quantile_df,
    )


# ============================================================
# DASHBOARD
# ============================================================

def create_dashboard(
    company,
    prediction_df,
):
    if prediction_df.empty:
        return

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=prediction_df[
                "date"
            ],
            y=prediction_df[
                "actual_close"
            ],
            mode="lines",
            name="Actual Close",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=prediction_df[
                "date"
            ],
            y=prediction_df[
                "pred_close"
            ],
            mode="lines",
            name="Predicted Close",
        )
    )

    figure.update_layout(
        title=(
            f"{company} — "
            f"Block 1 Kronos + LoRA | "
            f"118-day → 1-day walk-forward"
        ),
        xaxis_title="Date",
        yaxis_title="Close",
        template="plotly_white",
        height=700,
        width=1400,
    )

    figure.write_html(
        DASHBOARD_DIR
        / f"{company}_LoRA_dashboard.html",
        include_plotlyjs=True,
    )


# ============================================================
# PROCESS ONE COMPANY
# ============================================================

def process_company(
    company,
):
    start_time = time.time()

    print()
    print("=" * 90)
    print(
        f"PROCESSING "
        f"{company}"
    )
    print("=" * 90)

    df = load_company_data(
        company
    )

    print(
        f"{company}: "
        f"{len(df):,} rows | "
        f"{df['date'].min().date()} -> "
        f"{df['date'].max().date()}"
    )

    windows = build_windows(
        df
    )

    train_windows = select_windows(
        windows,
        TRAIN_START,
        TRAIN_END,
    )

    optuna_windows = select_windows(
        windows,
        OPTUNA_VAL_START,
        OPTUNA_VAL_END,
    )

    early_stop_windows = select_windows(
        windows,
        EARLY_STOP_START,
        EARLY_STOP_END,
    )

    print(
        f"{company}: "
        f"usable windows={len(windows):,} | "
        f"train={len(train_windows):,} | "
        f"Optuna={len(optuna_windows):,} | "
        f"early-stop={len(early_stop_windows):,}"
    )

    if not train_windows:
        print(
            f"{company}: "
            f"no training windows."
        )
        return None

    if not optuna_windows:
        print(
            f"{company}: "
            f"no Optuna validation windows."
        )
        return None

    if not early_stop_windows:
        print(
            f"{company}: "
            f"no early-stop windows."
        )
        return None

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    model, tokenizer = (
        load_lora_model()
    )

    print(
        f"{company}: "
        f"trainable LoRA parameters = "
        f"{sum(p.numel() for p in lora_parameters(model)):,}"
    )

    # --------------------------------------------------------
    # OPTUNA
    # --------------------------------------------------------

    best_parameters = (
        optuna_tune(
            model,
            tokenizer,
            train_windows,
            optuna_windows,
            company,
        )
    )

    learning_rate = float(
        best_parameters[
            "learning_rate"
        ]
    )

    weight_decay = float(
        best_parameters[
            "weight_decay"
        ]
    )

    # --------------------------------------------------------
    # FINAL TRAINING + EARLY STOP
    # --------------------------------------------------------

    (
        checkpoint_path,
        best_epoch,
        best_validation_rmse,
    ) = train_with_early_stopping(
        model,
        tokenizer,
        train_windows,
        early_stop_windows,
        learning_rate,
        weight_decay,
        company,
    )

    # --------------------------------------------------------
    # ALL USABLE EMBEDDINGS
    # --------------------------------------------------------

    embedding_path = (
        save_all_embeddings(
            model,
            tokenizer,
            windows,
            company,
        )
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------
    #
    # The checkpoint is now FIXED.
    # No training on test.
    #
    # For each test date:
    #
    #   previous 118 actual days
    #              ↓
    #        predict next day
    #              ↓
    #        shift one day
    #              ↓
    #             repeat
    #
    # --------------------------------------------------------

    prediction_df = (
        walk_forward_test(
            model,
            tokenizer,
            windows,
            company,
        )
    )

    prediction_path = (
        PRED_DIR
        / f"{company}_test_predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
    )

    # --------------------------------------------------------
    # PER-COMPANY METRICS
    # --------------------------------------------------------

    company_metrics = (
        calculate_company_metrics(
            prediction_df
        )
    )

    company_metrics.update(
        {
            "Company":
                company,

            "Lookback":
                LOOKBACK,

            "Forecast_Horizon":
                FORECAST_HORIZON,

            "LoRA_Rank":
                LORA_RANK,

            "LoRA_Alpha":
                LORA_ALPHA,

            "LoRA_Dropout":
                LORA_DROPOUT,

            "Best_Epoch":
                best_epoch,

            "Best_EarlyStop_Return_RMSE":
                best_validation_rmse,

            "Learning_Rate":
                learning_rate,

            "Weight_Decay":
                weight_decay,

            "Checkpoint":
                str(checkpoint_path),

            "Embedding_File":
                str(embedding_path),
        }
    )

    with open(
        METRICS_DIR
        / f"{company}_metrics.json",
        "w",
    ) as file:
        json.dump(
            company_metrics,
            file,
            indent=2,
            default=str,
        )

    if company in PLOT_COMPANIES:
        create_dashboard(
            company,
            prediction_df,
        )

    print()
    print(
        f"{company}: "
        f"TEST FINISHED"
    )
    print(
        f"  predictions = "
        f"{prediction_path}"
    )
    print(
        f"  embeddings  = "
        f"{embedding_path}"
    )
    print(
        f"  checkpoint  = "
        f"{checkpoint_path}"
    )

    for key, value in (
        company_metrics.items()
    ):
        if isinstance(
            value,
            (
                float,
                np.floating,
            ),
        ):
            print(
                f"  {key}: "
                f"{value:.8f}"
            )
        else:
            print(
                f"  {key}: "
                f"{value}"
            )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"{company}: "
        f"elapsed = "
        f"{elapsed / 3600:.2f} hours"
    )

    del model
    del tokenizer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        prediction_df,
        company_metrics,
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 90)
    print(
        "BLOCK 1 — "
        "KRONOS + LoRA ONLY"
    )
    print("=" * 90)

    print(
        "Device:",
        DEVICE,
    )

    print(
        "Data:",
        DATA_DIR,
    )

    print(
        "Lookback:",
        LOOKBACK,
        "trading days",
    )

    print(
        "Forecast:",
        FORECAST_HORIZON,
        "trading day",
    )

    print(
        "LoRA:",
        f"rank={LORA_RANK}, "
        f"alpha={LORA_ALPHA}, "
        f"dropout={LORA_DROPOUT}",
    )

    print(
        "Training:",
        "2022",
    )

    print(
        "Optuna:",
        "Jan-Jul 2023",
    )

    print(
        "Early stopping:",
        "Aug-Dec 2023",
    )

    print(
        "Final test:",
        "Jan 2024 -> latest",
    )

    print()

    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"DATA_DIR does not exist: "
            f"{DATA_DIR}"
        )

    companies = ["ADANIENT"]

    print(
        f"Companies found: "
        f"{len(companies)}"
    )

    print(
        ", ".join(companies)
    )

    all_predictions = []
    all_company_metrics = []

    for number, company in enumerate(
        companies,
        1,
    ):
        print()
        print(
            f"######## "
            f"COMPANY "
            f"{number}/"
            f"{len(companies)}: "
            f"{company} "
            f"########"
        )

        try:
            result = (
                process_company(
                    company
                )
            )

            if result is None:
                continue

            prediction_df, metrics = (
                result
            )

            all_predictions.append(
                prediction_df
            )

            all_company_metrics.append(
                metrics
            )

        except Exception as error:
            print()
            print(
                f"ERROR in "
                f"{company}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

            import traceback

            traceback.print_exc()

    if not all_predictions:
        raise RuntimeError(
            "No company completed."
        )

    # --------------------------------------------------------
    # COMBINE ALL 50 TEST PREDICTIONS
    # --------------------------------------------------------

    all_predictions_df = (
        pd.concat(
            all_predictions,
            ignore_index=True,
        )
    )

    all_predictions_df.to_csv(
        PRED_DIR
        / "ALL_50_companies_test_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        all_company_metrics
    ).to_csv(
        METRICS_DIR
        / "ALL_50_company_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # CROSS-SECTIONAL BLOCK 1 METRICS
    # --------------------------------------------------------

    (
        cross_sectional_metrics,
        daily_metrics,
        quantile_metrics,
    ) = calculate_cross_sectional_metrics(
        all_predictions_df
    )

    daily_metrics.to_csv(
        METRICS_DIR
        / "Block1_daily_cross_sectional_metrics.csv",
        index=False,
    )

    quantile_metrics.to_csv(
        METRICS_DIR
        / "Block1_quantile_returns.csv",
        index=False,
    )

    with open(
        METRICS_DIR
        / "Block1_cross_sectional_metrics.json",
        "w",
    ) as file:
        json.dump(
            cross_sectional_metrics,
            file,
            indent=2,
            default=str,
        )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 90)
    print(
        "BLOCK 1 COMPLETE"
    )
    print("=" * 90)

    print()
    print(
        "Cross-sectional metrics:"
    )

    for key, value in (
        cross_sectional_metrics.items()
    ):
        print(
            f"{key:40s}: "
            f"{value}"
        )

    print()
    print(
        "Predictions:",
        PRED_DIR,
    )

    print(
        "Metrics:",
        METRICS_DIR,
    )

    print(
        "Checkpoints:",
        CHECKPOINT_DIR,
    )

    print(
        "Embeddings:",
        EMBED_DIR,
    )

    print(
        "Dashboards:",
        DASHBOARD_DIR,
    )


if __name__ == "__main__":
    main()
