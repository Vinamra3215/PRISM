# =============================================================================
# NIFTY 50 — KRONOS BASE + 118-DAY ROLLING LoRA
# =============================================================================
#
# EXPERIMENT DESIGN
# -----------------------------------------------------------------------------
#
# For every prediction date:
#
#       Previous 118 trading days
#                 |
#                 v
#        Fresh Kronos Base
#                 |
#                 v
#          Fresh LoRA adapters
#                 |
#                 v
#        LoRA self-supervised
#             training
#                 |
#                 v
#        Predict NEXT 1 day
#                 |
#                 v
#          Save prediction
#                 |
#                 v
#        Move window forward 1 day
#                 |
#                 v
#        Fresh LoRA again
#
# Kronos Base weights are NEVER updated.
# LoRA adapters are RESET for every rolling window.
#
# ----------------------------------------------------------------------------- 
# EVALUATION
# -----------------------------------------------------------------------------
#
# Candlestick graphs:
#       Raw OHLCV
#       ONLY FOR VISUALIZATION
#
# All predictive metrics:
#       ONLY on DAILY CLOSE RETURNS
#
# Metrics:
#       RMSE
#       MAE
#       MAPE
#       R^2
#       Direction Accuracy
#       Return (% change)
#       Pearson Correlation
#
# ----------------------------------------------------------------------------- 
# OPTUNA
# -----------------------------------------------------------------------------
#
# Optuna tunes the LoRA + inference configuration on a chronological
# calibration section BEFORE the final rolling evaluation.
#
# Tuned:
#       LoRA rank
#       LoRA alpha
#       LoRA dropout
#       learning rate
#       epochs
#       temperature
#       top_k
#       top_p
#       sample_count
#
# The Optuna calibration data is never used as final evaluation data.
#
# =============================================================================

import os
import sys
import json
import random
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import optuna

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots


warnings.filterwarnings("ignore")


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path(
    "/home/soq/__shutupandbendover/yatin/lora tune"
)

KRONOS_DIR = Path(
    "/home/soq/Kronos"
)

DATA_PATH = BASE_DIR / "NIFTY50_5Y_OHLCV.parquet"

RESULTS_DIR = BASE_DIR / "nifty_lora_results"

TOKENIZER_PATH = Path(
    "/home/soq/__shutupandbendover/het-uchiha/weights/"
    "Kronos-Tokenizer-base"
)

MODEL_PATH = Path(
    "/home/soq/__shutupandbendover/het-uchiha/weights/"
    "Kronos-base"
)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

SEED = 42

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

WINDOW_SIZE = 118

PRED_LEN = 1

REQUESTED_START_DATE = pd.Timestamp(
    "2022-01-01"
)

MAX_CONTEXT = 512

# -------------------------------------------------------------------------
# Optuna
# -------------------------------------------------------------------------

OPTUNA_TRIALS = 8

OPTUNA_TIMEOUT = None

OPTUNA_STUDY_NAME = (
    "nifty50_kronos_lora_118day"
)

# -------------------------------------------------------------------------
# Final rolling training
#
# These are defaults only.
# Optuna-selected values replace them.
# -------------------------------------------------------------------------

DEFAULT_RANK = 8

DEFAULT_ALPHA = 16

DEFAULT_DROPOUT = 0.05

DEFAULT_LR = 1e-4

DEFAULT_EPOCHS = 5

# -------------------------------------------------------------------------
# Inference defaults
# -------------------------------------------------------------------------

DEFAULT_TEMPERATURE = 1.0

DEFAULT_TOP_K = 0

DEFAULT_TOP_P = 0.90

DEFAULT_SAMPLE_COUNT = 1

# -------------------------------------------------------------------------
# Optuna calibration
#
# First eligible 118-day window is divided chronologically:
#
#       calibration training = first 98 days
#       calibration validation = next 20 days
#
# Optuna predicts the validation section and evaluates CLOSE RETURNS.
# -------------------------------------------------------------------------

CALIBRATION_TRAIN_DAYS = 98

CALIBRATION_VALIDATION_DAYS = 20

# -------------------------------------------------------------------------
# LoRA target
#
# We adapt all Linear layers in Kronos Base.
# The underlying Base parameters remain frozen.
# -------------------------------------------------------------------------

LORA_TARGET_ALL_LINEAR = True


# =============================================================================
# SEED
# =============================================================================

def set_seed(seed=SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True

        torch.backends.cudnn.benchmark = False


# =============================================================================
# PRINT CONFIGURATION
# =============================================================================

def print_configuration():

    print()
    print("=" * 100)
    print("NIFTY 50 — KRONOS BASE + 118-DAY ROLLING LoRA")
    print("=" * 100)

    print()
    print("Device:")
    print(f"  {DEVICE}")

    print()
    print("Data:")
    print(f"  {DATA_PATH}")

    print()
    print("Kronos source:")
    print(f"  {KRONOS_DIR}")

    print()
    print("Tokenizer:")
    print(f"  {TOKENIZER_PATH}")

    print()
    print("Kronos Base:")
    print(f"  {MODEL_PATH}")

    print()
    print("Rolling window:")
    print(f"  {WINDOW_SIZE} trading days")

    print()
    print("Prediction:")
    print(f"  Next {PRED_LEN} trading day")

    print()
    print("Evaluation:")
    print("  DAILY CLOSE RETURNS ONLY")

    print()
    print("Metrics:")
    print("  RMSE")
    print("  MAE")
    print("  MAPE")
    print("  R^2")
    print("  Direction Accuracy")
    print("  Return (% change)")
    print("  Pearson Correlation")

    print()
    print("Optuna:")
    print(f"  Trials: {OPTUNA_TRIALS}")

    print()
    print("IMPORTANT:")
    print("  Fresh LoRA adapters are created/reset for every rolling window.")

    print()
    print("IMPORTANT:")
    print("  Kronos Base weights remain frozen.")

    print()
    print("IMPORTANT:")
    print("  Raw OHLCV is used ONLY for candlestick visualization.")

    print("=" * 100)


# =============================================================================
# KRONOS IMPORT
# =============================================================================

def import_kronos():

    print()
    print("=" * 100)
    print("IMPORTING KRONOS")
    print("=" * 100)

    if not KRONOS_DIR.exists():

        raise FileNotFoundError(
            f"Kronos directory does not exist:\n"
            f"{KRONOS_DIR}"
        )

    if str(KRONOS_DIR) not in sys.path:

        sys.path.insert(
            0,
            str(KRONOS_DIR)
        )

    try:

        from model.kronos import (
            Kronos,
            KronosTokenizer,
            KronosPredictor,
        )

        print("Kronos import successful.")

        print("  Kronos")
        print("  KronosTokenizer")
        print("  KronosPredictor")

        return (
            Kronos,
            KronosTokenizer,
            KronosPredictor,
        )

    except Exception as e:

        print()
        print("Kronos import FAILED.")

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error: {e}"
        )

        raise


# =============================================================================
# DATA LOADING
# =============================================================================

def normalize_column_name(column):

    text = str(column)

    text = text.lower()

    text = text.replace(
        " ",
        ""
    )

    text = text.replace(
        "_",
        ""
    )

    text = text.replace(
        "'",
        ""
    )

    text = text.replace(
        '"',
        ""
    )

    return text


def find_column(columns, candidates):

    normalized = {}

    for column in columns:

        normalized[
            normalize_column_name(column)
        ] = column

    for candidate in candidates:

        key = normalize_column_name(
            candidate
        )

        if key in normalized:

            return normalized[key]

    return None


def load_nifty_data():

    print()
    print("=" * 100)
    print("LOADING NIFTY 50 DATA")
    print("=" * 100)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"NIFTY data not found:\n{DATA_PATH}"
        )

    df = pd.read_parquet(DATA_PATH)

    print()
    print("Original shape:")
    print(df.shape)

    print()
    print("Original columns:")
    print(df.columns.tolist())

    # =====================================================================
    # HANDLE YFINANCE-STYLE MULTIINDEX
    # =====================================================================

    if isinstance(df.columns, pd.MultiIndex):

        print()
        print("MultiIndex columns detected.")

        # Example:
        # ('Close', '^NSEI')
        # ('High', '^NSEI')
        #
        # We only need the first level:
        # Close
        # High
        # Low
        # Open
        # Volume

        first_level = df.columns.get_level_values(0)

        df.columns = [
            str(col).strip().lower()
            for col in first_level
        ]

    else:

        df.columns = [
            str(col).strip().lower()
            for col in df.columns
        ]

    print()
    print("Columns after MultiIndex handling:")
    print(df.columns.tolist())

    # =====================================================================
    # FIND OHLCV
    # =====================================================================

    required_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            "Could not identify required NIFTY OHLCV columns.\n"
            f"Missing: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # =====================================================================
    # KEEP ONLY OHLCV
    # =====================================================================

    df = df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    # =====================================================================
    # DATE INDEX
    # =====================================================================

    if isinstance(df.index, pd.DatetimeIndex):

        df.index = pd.to_datetime(
            df.index
        )

    else:

        # Try common date column names
        date_column = None

        for candidate in [
            "date",
            "datetime",
            "timestamp",
        ]:

            if candidate in df.columns:

                date_column = candidate
                break

        if date_column is None:

            # In your file the date is expected to be the index.
            # If parquet loaded it as a generic index, try conversion.
            try:

                df.index = pd.to_datetime(
                    df.index
                )

            except Exception as e:

                raise ValueError(
                    "Could not identify the NIFTY date index."
                ) from e

        else:

            df[date_column] = pd.to_datetime(
                df[date_column]
            )

            df = df.set_index(
                date_column
            )

    # =====================================================================
    # NUMERIC CONVERSION
    # =====================================================================

    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # =====================================================================
    # CLEAN DATA
    # =====================================================================

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=required_columns
    )

    # Remove duplicate dates
    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    # Sort chronologically
    df = df.sort_index()

    # =====================================================================
    # FINAL OUTPUT
    # =====================================================================

    print()
    print("Final columns:")
    print(df.columns.tolist())

    print()
    print("Final shape:")
    print(df.shape)

    print()
    print("Date range:")
    print(
        df.index.min(),
        "to",
        df.index.max()
    )

    print()
    print("First rows:")
    print(df.head())

    print()
    print("Last rows:")
    print(df.tail())

    # =====================================================================
    # SANITY CHECK
    # =====================================================================

    if len(df) <= WINDOW_SIZE:

        raise ValueError(
            f"Not enough NIFTY data for a "
            f"{WINDOW_SIZE}-day rolling window.\n"
            f"Rows available: {len(df)}"
        )

    if not (
        df["high"] >= df["low"]
    ).all():

        raise ValueError(
            "Invalid OHLC data: High < Low detected."
        )

    if not (
        df["volume"] >= 0
    ).all():

        raise ValueError(
            "Invalid NIFTY data: negative volume detected."
        )

    print()
    print("NIFTY OHLCV validation: PASSED")

    return df


# =============================================================================
# AMOUNT COLUMN
# =============================================================================

def add_amount_column(df):

    result = df.copy()

    result["amount"] = (
        result["volume"]
        * result[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ].mean(axis=1)
    )

    return result


# =============================================================================
# LoRA LINEAR LAYER
# =============================================================================

class LoRALinear(nn.Module):

    def __init__(
        self,
        base_layer,
        rank,
        alpha,
        dropout,
    ):

        super().__init__()

        if not isinstance(
            base_layer,
            nn.Linear
        ):

            raise TypeError(
                "base_layer must be nn.Linear"
            )

        self.base = base_layer

        self.rank = int(rank)

        self.alpha = float(alpha)

        self.scaling = (
            self.alpha / self.rank
        )

        self.dropout = nn.Dropout(
            float(dropout)
        )

        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base_layer.in_features,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )

        self.lora_B = nn.Parameter(
            torch.zeros(
                base_layer.out_features,
                self.rank,
                device=base_layer.weight.device,
                dtype=base_layer.weight.dtype,
            )
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=np.sqrt(5)
        )

        # Base weights are always frozen.
        self.base.weight.requires_grad = False

        if self.base.bias is not None:

            self.base.bias.requires_grad = False

    def forward(self, x):

        base_output = self.base(x)

        lora_output = F.linear(
            self.dropout(x),
            self.lora_A
        )

        lora_output = F.linear(
            lora_output,
            self.lora_B
        )

        return (
            base_output
            + self.scaling * lora_output
        )


# =============================================================================
# ADD LoRA TO MODEL
# =============================================================================

def add_lora_to_model(
    module,
    rank,
    alpha,
    dropout,
):

    replaced = 0

    for name, child in list(
        module.named_children()
    ):

        if isinstance(
            child,
            LoRALinear
        ):

            continue

        if isinstance(
            child,
            nn.Linear
        ):

            wrapped = LoRALinear(
                base_layer=child,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

            setattr(
                module,
                name,
                wrapped
            )

            replaced += 1

        else:

            replaced += add_lora_to_model(
                child,
                rank,
                alpha,
                dropout,
            )

    return replaced


# =============================================================================
# FREEZE EVERYTHING EXCEPT LoRA
# =============================================================================

def freeze_base_enable_lora(
    model
):

    for parameter in model.parameters():

        parameter.requires_grad = False

    trainable = 0

    total = 0

    for module in model.modules():

        if isinstance(
            module,
            LoRALinear
        ):

            module.lora_A.requires_grad = True

            module.lora_B.requires_grad = True

            trainable += (
                module.lora_A.numel()
                + module.lora_B.numel()
            )

    for parameter in model.parameters():

        total += parameter.numel()

    return trainable, total


# =============================================================================
# RESET LoRA
# =============================================================================

def reset_lora_weights(model):

    count = 0

    for module in model.modules():

        if isinstance(
            module,
            LoRALinear
        ):

            nn.init.kaiming_uniform_(
                module.lora_A,
                a=np.sqrt(5)
            )

            nn.init.zeros_(
                module.lora_B
            )

            count += 1

    return count


# =============================================================================
# COUNT LoRA PARAMETERS
# =============================================================================

def count_lora_parameters(model):

    trainable = 0

    total = 0

    for parameter in model.parameters():

        total += parameter.numel()

        if parameter.requires_grad:

            trainable += parameter.numel()

    return trainable, total


# =============================================================================
# TIME STAMPS
# =============================================================================

def make_time_stamps(index):

    timestamps = pd.Series(
        pd.to_datetime(index)
    ).reset_index(
        drop=True
    )

    result = pd.DataFrame()

    result["minute"] = (
        timestamps.dt.minute
    )

    result["hour"] = (
        timestamps.dt.hour
    )

    result["weekday"] = (
        timestamps.dt.weekday
    )

    result["day"] = (
        timestamps.dt.day
    )

    result["month"] = (
        timestamps.dt.month
    )

    return result


# =============================================================================
# PREPARE KRONOS INPUT
# =============================================================================

def prepare_kronos_array(df):

    working = add_amount_column(
        df
    )

    values = working[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]
    ].values.astype(
        np.float32
    )

    mean = np.mean(
        values,
        axis=0
    )

    std = np.std(
        values,
        axis=0
    )

    normalized = (
        values
        - mean
    ) / (
        std + 1e-5
    )

    normalized = np.clip(
        normalized,
        -5,
        5
    )

    return (
        normalized.astype(
            np.float32
        ),
        mean.astype(
            np.float32
        ),
        std.astype(
            np.float32
        ),
    )


# =============================================================================
# TOKENIZE TRAINING WINDOW
# =============================================================================

def tokenize_window(
    tokenizer,
    df
):

    normalized, _, _ = (
        prepare_kronos_array(
            df
        )
    )

    tensor = torch.from_numpy(
        normalized
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    tokenizer.eval()

    with torch.no_grad():

        tokens = tokenizer.encode(
            tensor,
            half=True
        )

    return tokens


# =============================================================================
# ONE EPOCH LoRA TRAINING
# =============================================================================

def train_one_epoch(
    model,
    tokenizer,
    df,
    optimizer,
):

    model.train()

    tokens = tokenize_window(
        tokenizer,
        df
    )

    s1_ids = tokens[0]

    s2_ids = tokens[1]

    if s1_ids.shape[1] < 3:

        raise ValueError(
            "Training window is too short."
        )

    # -------------------------------------------------------------------------
    # Next-token objective
    #
    # Input:
    #   tokens[0:-1]
    #
    # Target:
    #   tokens[1:]
    #
    # Kronos transformer uses causal attention.
    # -------------------------------------------------------------------------

    input_s1 = (
        s1_ids[:, :-1]
    )

    input_s2 = (
        s2_ids[:, :-1]
    )

    target_s1 = (
        s1_ids[:, 1:]
    )

    target_s2 = (
        s2_ids[:, 1:]
    )

    stamps = make_time_stamps(
        df.index
    ).values.astype(
        np.float32
    )

    stamps = torch.from_numpy(
        stamps
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    input_stamps = (
        stamps[:, :-1]
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    s1_logits, s2_logits = model(
        input_s1,
        input_s2,
        stamp=input_stamps,
        use_teacher_forcing=True,
        s1_targets=target_s1,
    )

    loss_s1 = F.cross_entropy(
        s1_logits.reshape(
            -1,
            s1_logits.shape[-1]
        ),
        target_s1.reshape(
            -1
        )
    )

    loss_s2 = F.cross_entropy(
        s2_logits.reshape(
            -1,
            s2_logits.shape[-1]
        ),
        target_s2.reshape(
            -1
        )
    )

    loss = (
        loss_s1
        + loss_s2
    ) / 2.0

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        max_norm=1.0
    )

    optimizer.step()

    return float(
        loss.detach().cpu()
    )


# =============================================================================
# LoRA TRAIN
# =============================================================================

def train_lora(
    model,
    tokenizer,
    df,
    lr,
    epochs,
    verbose=False,
):

    optimizer = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=lr,
        weight_decay=0.01,
    )

    history = []

    for epoch in range(
        int(epochs)
    ):

        loss = train_one_epoch(
            model=model,
            tokenizer=tokenizer,
            df=df,
            optimizer=optimizer,
        )

        history.append(
            loss
        )

        if verbose:

            print(
                f"      Epoch "
                f"{epoch + 1}/{epochs} "
                f"loss={loss:.6f}"
            )

    return history


# =============================================================================
# TOKEN VALIDATION LOSS
# =============================================================================

def token_validation_loss(
    model,
    tokenizer,
    df,
):

    model.eval()

    tokens = tokenize_window(
        tokenizer,
        df
    )

    s1_ids = tokens[0]

    s2_ids = tokens[1]

    if s1_ids.shape[1] < 3:

        return float("inf")

    input_s1 = s1_ids[:, :-1]

    input_s2 = s2_ids[:, :-1]

    target_s1 = s1_ids[:, 1:]

    target_s2 = s2_ids[:, 1:]

    stamps = make_time_stamps(
        df.index
    ).values.astype(
        np.float32
    )

    stamps = torch.from_numpy(
        stamps
    ).unsqueeze(
        0
    ).to(
        DEVICE
    )

    with torch.no_grad():

        s1_logits, s2_logits = model(
            input_s1,
            input_s2,
            stamp=stamps[:, :-1],
            use_teacher_forcing=True,
            s1_targets=target_s1,
        )

        loss_s1 = F.cross_entropy(
            s1_logits.reshape(
                -1,
                s1_logits.shape[-1]
            ),
            target_s1.reshape(
                -1
            )
        )

        loss_s2 = F.cross_entropy(
            s2_logits.reshape(
                -1,
                s2_logits.shape[-1]
            ),
            target_s2.reshape(
                -1
            )
        )

        loss = (
            loss_s1
            + loss_s2
        ) / 2.0

    return float(
        loss.cpu()
    )


# =============================================================================
# CLOSE RETURN METRIC
# =============================================================================

def close_return_rmse(
    actual_close,
    predicted_close,
    previous_close,
):

    actual_return = (
        actual_close
        / previous_close
        - 1.0
    )

    predicted_return = (
        predicted_close
        / previous_close
        - 1.0
    )

    return float(
        np.sqrt(
            mean_squared_error(
                [actual_return],
                [predicted_return]
            )
        )
    )


# =============================================================================
# KRONOS PREDICTION
# =============================================================================

def kronos_predict(
    model,
    tokenizer,
    KronosPredictor,
    history_df,
    future_index,
    temperature,
    top_k,
    top_p,
    sample_count,
):

    model.eval()

    tokenizer.eval()

    predictor = KronosPredictor(
        model,
        tokenizer,
        device=DEVICE,
        max_context=MAX_CONTEXT,
    )

    x_df = history_df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
    ].copy()

    x_timestamp = pd.Series(
        pd.to_datetime(
            history_df.index
        )
    )

    y_timestamp = pd.Series(
        pd.to_datetime(
            future_index
        )
    )

    with torch.no_grad():

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=len(
                y_timestamp
            ),
            T=float(
                temperature
            ),
            top_k=int(
                top_k
            ),
            top_p=float(
                top_p
            ),
            sample_count=int(
                sample_count
            ),
            verbose=False,
        )

    return pred_df


# =============================================================================
# LOAD FRESH KRONOS
# =============================================================================

def load_fresh_kronos(
    Kronos,
    KronosTokenizer,
):

    tokenizer = (
        KronosTokenizer.from_pretrained(
            str(TOKENIZER_PATH)
        )
    )

    model = (
        Kronos.from_pretrained(
            str(MODEL_PATH)
        )
    )

    tokenizer = tokenizer.to(
        DEVICE
    )

    model = model.to(
        DEVICE
    )

    tokenizer.eval()

    model.eval()

    return (
        model,
        tokenizer,
    )


# =============================================================================
# OPTUNA OBJECTIVE
# =============================================================================

def make_optuna_objective(
    Kronos,
    KronosTokenizer,
    KronosPredictor,
    calibration_df,
):

    # -------------------------------------------------------------------------
    # Chronological calibration split
    # -------------------------------------------------------------------------

    train_df = calibration_df.iloc[
        :CALIBRATION_TRAIN_DAYS
    ].copy()

    validation_df = calibration_df.iloc[
        CALIBRATION_TRAIN_DAYS:
        CALIBRATION_TRAIN_DAYS
        + CALIBRATION_VALIDATION_DAYS
    ].copy()

    def objective(trial):

        rank = trial.suggest_categorical(
            "rank",
            [4, 8, 16]
        )

        alpha = trial.suggest_categorical(
            "alpha",
            [8, 16, 32]
        )

        dropout = trial.suggest_float(
            "dropout",
            0.0,
            0.10,
            step=0.025
        )

        lr = trial.suggest_float(
            "learning_rate",
            1e-5,
            5e-4,
            log=True
        )

        epochs = trial.suggest_int(
            "epochs",
            2,
            6
        )

        temperature = trial.suggest_float(
            "temperature",
            0.7,
            1.3
        )

        top_k = trial.suggest_categorical(
            "top_k",
            [0, 5, 10, 20]
        )

        top_p = trial.suggest_float(
            "top_p",
            0.75,
            1.0
        )

        sample_count = trial.suggest_categorical(
            "sample_count",
            [1, 3]
        )

        print()
        print(
            f"[Optuna Trial "
            f"{trial.number + 1}/{OPTUNA_TRIALS}]"
        )

        print(
            f"rank={rank}, "
            f"alpha={alpha}, "
            f"dropout={dropout:.3f}, "
            f"lr={lr:.2e}, "
            f"epochs={epochs}, "
            f"T={temperature:.3f}, "
            f"top_k={top_k}, "
            f"top_p={top_p:.3f}, "
            f"samples={sample_count}"
        )

        # ---------------------------------------------------------------------
        # Fresh Kronos Base for every Optuna trial
        # ---------------------------------------------------------------------

        model, tokenizer = (
            load_fresh_kronos(
                Kronos,
                KronosTokenizer,
            )
        )

        # ---------------------------------------------------------------------
        # Add LoRA
        # ---------------------------------------------------------------------

        replaced = add_lora_to_model(
            model,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

        trainable, total = (
            freeze_base_enable_lora(
                model
            )
        )

        print(
            f"  LoRA Linear layers: {replaced}"
        )

        print(
            f"  Trainable parameters: "
            f"{trainable:,}"
        )

        # ---------------------------------------------------------------------
        # Train only on chronological calibration train section
        # ---------------------------------------------------------------------

        train_lora(
            model=model,
            tokenizer=tokenizer,
            df=train_df,
            lr=lr,
            epochs=epochs,
            verbose=False,
        )

        # ---------------------------------------------------------------------
        # Validation forecast
        # ---------------------------------------------------------------------

        try:

            pred_df = kronos_predict(
                model=model,
                tokenizer=tokenizer,
                KronosPredictor=KronosPredictor,
                history_df=train_df,
                future_index=validation_df.index,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                sample_count=sample_count,
            )

            previous_close = float(
                train_df["close"].iloc[-1]
            )

            actual_close = (
                validation_df[
                    "close"
                ].values
            )

            predicted_close = (
                pred_df[
                    "close"
                ].values
            )

            # -------------------------------------------------------------
            # Validation return series
            # -------------------------------------------------------------

            actual_returns = (
                actual_close
                / np.r_[
                    previous_close,
                    actual_close[:-1]
                ]
                - 1.0
            )

            predicted_returns = (
                predicted_close
                / np.r_[
                    previous_close,
                    actual_close[:-1]
                ]
                - 1.0
            )

            score = float(
                np.sqrt(
                    mean_squared_error(
                        actual_returns,
                        predicted_returns,
                    )
                )
            )

            if not np.isfinite(
                score
            ):

                score = 1e9

        except Exception as e:

            print(
                f"  Trial failed: {e}"
            )

            score = 1e9

        # ---------------------------------------------------------------------
        # Cleanup
        # ---------------------------------------------------------------------

        del model

        del tokenizer

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        print(
            f"  Validation return RMSE: "
            f"{score:.8f}"
        )

        return score

    return objective


# =============================================================================
# OPTUNA TUNING
# =============================================================================

def run_optuna(
    Kronos,
    KronosTokenizer,
    KronosPredictor,
    df,
):

    print()
    print("=" * 100)
    print("STARTING OPTUNA TUNING")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # First eligible rolling window
    # -------------------------------------------------------------------------

    eligible_positions = np.where(
        df.index >= REQUESTED_START_DATE
    )[0]

    if len(
        eligible_positions
    ) == 0:

        raise ValueError(
            "No dates after requested start."
        )

    first_target_position = (
        eligible_positions[0]
    )

    if (
        first_target_position
        < WINDOW_SIZE
    ):

        first_target_position = (
            WINDOW_SIZE
        )

    calibration_start = (
        first_target_position
        - WINDOW_SIZE
    )

    calibration_end = (
        first_target_position
    )

    calibration_df = df.iloc[
        calibration_start:
        calibration_end
    ].copy()

    if len(
        calibration_df
    ) < WINDOW_SIZE:

        raise ValueError(
            "Calibration window too short."
        )

    if (
        CALIBRATION_TRAIN_DAYS
        + CALIBRATION_VALIDATION_DAYS
        > len(calibration_df)
    ):

        raise ValueError(
            "Optuna calibration split is too large."
        )

    print()
    print("Calibration window:")
    print(
        calibration_df.index.min(),
        "to",
        calibration_df.index.max()
    )

    print()
    print(
        "Calibration training days:",
        CALIBRATION_TRAIN_DAYS
    )

    print(
        "Calibration validation days:",
        CALIBRATION_VALIDATION_DAYS
    )

    study = optuna.create_study(
        study_name=OPTUNA_STUDY_NAME,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=SEED
        ),
    )

    objective = make_optuna_objective(
        Kronos=Kronos,
        KronosTokenizer=KronosTokenizer,
        KronosPredictor=KronosPredictor,
        calibration_df=calibration_df,
    )

    study.optimize(
        objective,
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=False,
    )

    print()
    print("=" * 100)
    print("OPTUNA FINISHED")
    print("=" * 100)

    print()
    print("Best value:")
    print(
        study.best_value
    )

    print()
    print("Best parameters:")

    for key, value in (
        study.best_params.items()
    ):

        print(
            f"  {key}: {value}"
        )

    # -------------------------------------------------------------------------
    # Save Optuna history
    # -------------------------------------------------------------------------

    trials_df = (
        study.trials_dataframe()
    )

    trials_df.to_csv(
        RESULTS_DIR
        / "optuna_history.csv",
        index=False
    )

    best_params = dict(
        study.best_params
    )

    with open(
        RESULTS_DIR
        / "optuna_best_params.json",
        "w"
    ) as f:

        json.dump(
            best_params,
            f,
            indent=4
        )

    return best_params


# =============================================================================
# ROLLING PREDICTION
# =============================================================================

def run_rolling_lora(
    df,
    Kronos,
    KronosTokenizer,
    KronosPredictor,
    best_params,
):

    print()
    print("=" * 100)
    print("STARTING 118-DAY ROLLING LoRA")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Fresh Kronos Base loaded ONCE.
    #
    # Base parameters are frozen forever.
    #
    # Every window gets freshly initialized LoRA parameters.
    #
    # This is equivalent to:
    #
    #   Fresh Base
    #       +
    #   Fresh LoRA
    #
    # because Base weights never change.
    # -------------------------------------------------------------------------

    model, tokenizer = (
        load_fresh_kronos(
            Kronos,
            KronosTokenizer,
        )
    )

    replaced = add_lora_to_model(
        model,
        rank=int(
            best_params["rank"]
        ),
        alpha=float(
            best_params["alpha"]
        ),
        dropout=float(
            best_params["dropout"]
        ),
    )

    trainable, total = (
        freeze_base_enable_lora(
            model
        )
    )

    print()
    print(
        f"LoRA Linear layers: {replaced}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

    print(
        f"Total model parameters: "
        f"{total:,}"
    )

    print()
    print("Best Optuna configuration:")

    for key, value in (
        best_params.items()
    ):

        print(
            f"  {key}: {value}"
        )

    # -------------------------------------------------------------------------
    # Determine rolling positions
    # -------------------------------------------------------------------------

    positions = []

    for target_position in range(
        WINDOW_SIZE,
        len(df)
    ):

        target_date = (
            df.index[
                target_position
            ]
        )

        if (
            target_date
            >= REQUESTED_START_DATE
        ):

            positions.append(
                target_position
            )

    print()
    print(
        "Total rolling predictions:",
        len(positions)
    )

    if not positions:

        raise ValueError(
            "No rolling prediction positions."
        )

    results = []

    # -------------------------------------------------------------------------
    # Rolling loop
    # -------------------------------------------------------------------------

    for counter, target_position in enumerate(
        positions,
        start=1
    ):

        target_date = (
            df.index[
                target_position
            ]
        )

        window_start = (
            target_position
            - WINDOW_SIZE
        )

        window_end = (
            target_position
        )

        history_df = df.iloc[
            window_start:
            window_end
        ].copy()

        actual_row = (
            df.iloc[
                target_position
            ]
        )

        # ---------------------------------------------------------------------
        # Reset LoRA.
        #
        # Base weights remain exactly frozen.
        # LoRA B starts at zero and A is randomly initialized.
        # ---------------------------------------------------------------------

        reset_lora_weights(
            model
        )

        # ---------------------------------------------------------------------
        # Train LoRA
        # ---------------------------------------------------------------------

        history = train_lora(
            model=model,
            tokenizer=tokenizer,
            df=history_df,
            lr=float(
                best_params[
                    "learning_rate"
                ]
            ),
            epochs=int(
                best_params[
                    "epochs"
                ]
            ),
            verbose=False,
        )

        # ---------------------------------------------------------------------
        # Predict next ONE day
        # ---------------------------------------------------------------------

        future_index = pd.DatetimeIndex(
            [
                target_date
            ]
        )

        pred_df = kronos_predict(
            model=model,
            tokenizer=tokenizer,
            KronosPredictor=KronosPredictor,
            history_df=history_df,
            future_index=future_index,
            temperature=float(
                best_params[
                    "temperature"
                ]
            ),
            top_k=int(
                best_params[
                    "top_k"
                ]
            ),
            top_p=float(
                best_params[
                    "top_p"
                ]
            ),
            sample_count=int(
                best_params[
                    "sample_count"
                ]
            ),
        )

        predicted_row = (
            pred_df.iloc[0]
        )

        # ---------------------------------------------------------------------
        # Save result
        # ---------------------------------------------------------------------

        results.append(
            {
                "Date": target_date,

                "actual_open":
                    float(
                        actual_row["open"]
                    ),

                "actual_high":
                    float(
                        actual_row["high"]
                    ),

                "actual_low":
                    float(
                        actual_row["low"]
                    ),

                "actual_close":
                    float(
                        actual_row["close"]
                    ),

                "actual_volume":
                    float(
                        actual_row["volume"]
                    ),

                "predicted_open":
                    float(
                        predicted_row["open"]
                    ),

                "predicted_high":
                    float(
                        predicted_row["high"]
                    ),

                "predicted_low":
                    float(
                        predicted_row["low"]
                    ),

                "predicted_close":
                    float(
                        predicted_row["close"]
                    ),

                "predicted_volume":
                    float(
                        predicted_row["volume"]
                    ),

                "predicted_amount":
                    float(
                        predicted_row["amount"]
                    ),

                "training_loss":
                    float(
                        history[-1]
                    ),

                "training_epochs":
                    int(
                        best_params[
                            "epochs"
                        ]
                    ),

                "lora_rank":
                    int(
                        best_params[
                            "rank"
                        ]
                    ),

                "lora_alpha":
                    float(
                        best_params[
                            "alpha"
                        ]
                    ),
            }
        )

        # ---------------------------------------------------------------------
        # Progress
        # ---------------------------------------------------------------------

        if (
            counter <= 5
            or counter % 25 == 0
            or counter == len(positions)
        ):

            print(
                f"[{counter:4d}/"
                f"{len(positions):4d}] "
                f"{target_date.date()} | "
                f"Actual Close="
                f"{actual_row['close']:.2f} | "
                f"Predicted Close="
                f"{predicted_row['close']:.2f}"
            )

    # -------------------------------------------------------------------------
    # DataFrame
    # -------------------------------------------------------------------------

    result_df = pd.DataFrame(
        results
    )

    result_df["Date"] = pd.to_datetime(
        result_df["Date"]
    )

    result_df = result_df.set_index(
        "Date"
    )

    result_df = result_df.sort_index()

    return result_df


# =============================================================================
# RETURN-BASED METRICS
# =============================================================================

def calculate_return_metrics(
    predictions_df
):

    df = predictions_df.copy()

    # -------------------------------------------------------------------------
    # Previous actual close
    # -------------------------------------------------------------------------

    df["previous_actual_close"] = (
        df["actual_close"].shift(1)
    )

    # -------------------------------------------------------------------------
    # Daily close returns
    # -------------------------------------------------------------------------

    df["actual_return"] = (
        df["actual_close"]
        / df["previous_actual_close"]
        - 1.0
    )

    df["predicted_return"] = (
        df["predicted_close"]
        / df["previous_actual_close"]
        - 1.0
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    evaluation = df[
        [
            "actual_return",
            "predicted_return",
        ]
    ].dropna()

    if len(
        evaluation
    ) < 2:

        raise ValueError(
            "Not enough observations for return metrics."
        )

    y_true = (
        evaluation[
            "actual_return"
        ].values
    )

    y_pred = (
        evaluation[
            "predicted_return"
        ].values
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred
        )
    )

    mape = float(
        mean_absolute_percentage_error(
            y_true,
            y_pred
        )
        * 100.0
    )

    r2 = float(
        r2_score(
            y_true,
            y_pred
        )
    )

    direction_accuracy = float(
        np.mean(
            np.sign(y_true)
            == np.sign(y_pred)
        )
        * 100.0
    )

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):

        pearson = float(
            "nan"
        )

    else:

        pearson = float(
            np.corrcoef(
                y_true,
                y_pred
            )[0, 1]
        )

    # -------------------------------------------------------------------------
    # Total compounded return
    # -------------------------------------------------------------------------

    actual_total_return = float(
        (
            np.prod(
                1.0 + y_true
            )
            - 1.0
        )
        * 100.0
    )

    predicted_total_return = float(
        (
            np.prod(
                1.0 + y_pred
            )
            - 1.0
        )
        * 100.0
    )

    # -------------------------------------------------------------------------
    # Add return columns
    # -------------------------------------------------------------------------

    df["actual_return_pct"] = (
        df["actual_return"]
        * 100.0
    )

    df["predicted_return_pct"] = (
        df["predicted_return"]
        * 100.0
    )

    df["return_error_pct"] = (
        df["predicted_return_pct"]
        - df["actual_return_pct"]
    )

    metrics = {

        "evaluation_basis":
            "daily_close_returns_only",

        "rmse":
            rmse,

        "mae":
            mae,

        "mape_percent":
            mape,

        "r2":
            r2,

        "direction_accuracy_percent":
            direction_accuracy,

        "actual_total_return_percent":
            actual_total_return,

        "predicted_total_return_percent":
            predicted_total_return,

        "pearson_correlation":
            pearson,

        "number_of_evaluated_days":
            int(
                len(evaluation)
            ),

        "rolling_window_days":
            WINDOW_SIZE,

        "prediction_length_days":
            PRED_LEN,

        "fresh_lora_every_window":
            True,

        "fresh_base_weights":
            True,

        "optuna_tuned":
            True,
    }

    return (
        df,
        metrics
    )


# =============================================================================
# FINAL DASHBOARD
# =============================================================================

def create_dashboard(
    predictions_df,
    metrics,
):

    print()
    print("=" * 100)
    print("CREATING FINAL DASHBOARD")
    print("=" * 100)

    df = predictions_df.copy()

    # -------------------------------------------------------------------------
    # Close difference
    # -------------------------------------------------------------------------

    df["close_difference"] = (
        df["actual_close"]
        - df["predicted_close"]
    )

    # -------------------------------------------------------------------------
    # Create large baseline-style layout
    #
    # Left:
    #   Actual candlestick
    #   Predicted candlestick
    #   Close difference
    #
    # Right:
    #   Metrics table
    # -------------------------------------------------------------------------

    fig = make_subplots(

        rows=3,

        cols=2,

        column_widths=[
            0.82,
            0.18
        ],

        row_heights=[
            0.36,
            0.36,
            0.28
        ],

        horizontal_spacing=0.035,

        vertical_spacing=0.07,

        specs=[
            [
                {
                    "type":
                    "candlestick"
                },
                {
                    "type":
                    "table",
                    "rowspan": 2
                },
            ],

            [
                {
                    "type":
                    "candlestick"
                },
                None,
            ],

            [
                {
                    "type":
                    "xy"
                },
                None,
            ],
        ],

        subplot_titles=[
            "1. Actual NIFTY 50 Daily OHLCV",
            "2. Kronos Base + 118-Day Rolling LoRA Predicted Daily OHLCV",
            "3. Actual Close − Predicted Close",
        ],
    )

    # =========================================================================
    # ACTUAL CANDLESTICK
    # =========================================================================

    fig.add_trace(

        go.Candlestick(

            x=df.index,

            open=df[
                "actual_open"
            ],

            high=df[
                "actual_high"
            ],

            low=df[
                "actual_low"
            ],

            close=df[
                "actual_close"
            ],

            name="Actual NIFTY 50",

            increasing=dict(
                line=dict(
                    color="#00897B"
                ),

                fillcolor="#00897B",
            ),

            decreasing=dict(
                line=dict(
                    color="#E53935"
                ),

                fillcolor="#E53935",
            ),
        ),

        row=1,

        col=1,
    )

    # =========================================================================
    # PREDICTED CANDLESTICK
    # =========================================================================

    fig.add_trace(

        go.Candlestick(

            x=df.index,

            open=df[
                "predicted_open"
            ],

            high=df[
                "predicted_high"
            ],

            low=df[
                "predicted_low"
            ],

            close=df[
                "predicted_close"
            ],

            name="Predicted NIFTY 50",

            increasing=dict(
                line=dict(
                    color="#00897B"
                ),

                fillcolor="#00897B",
            ),

            decreasing=dict(
                line=dict(
                    color="#E53935"
                ),

                fillcolor="#E53935",
            ),
        ),

        row=2,

        col=1,
    )

    # =========================================================================
    # CLOSE DIFFERENCE
    # =========================================================================

    fig.add_trace(

        go.Scatter(

            x=df.index,

            y=df[
                "close_difference"
            ],

            mode="lines",

            name=(
                "Actual Close − "
                "Predicted Close"
            ),

            line=dict(
                color="#7B1FA2",
                width=1.5,
            ),

            hovertemplate=(
                "%{x|%Y-%m-%d}"
                "<br>"
                "Actual Close − "
                "Predicted Close: "
                "%{y:.2f}"
                "<extra></extra>"
            ),
        ),

        row=3,

        col=1,
    )

    # =========================================================================
    # ZERO LINE
    # =========================================================================

    fig.add_hline(

        y=0,

        line=dict(
            color="#444444",
            dash="dash",
            width=1,
        ),

        row=3,

        col=1,
    )

    # =========================================================================
    # METRICS TABLE
    # =========================================================================

    metric_names = [

        "RMSE",

        "MAE",

        "MAPE",

        "R²",

        "Direction Accuracy",

        "Return (% change)",

        "Pearson Correlation",
    ]

    metric_values = [

        f"{metrics['rmse']:.8f}",

        f"{metrics['mae']:.8f}",

        f"{metrics['mape_percent']:.4f}%",

        f"{metrics['r2']:.8f}",

        (
            f"{metrics['direction_accuracy_percent']:.4f}%"
        ),

        (
            f"Actual: "
            f"{metrics['actual_total_return_percent']:.4f}%"
            "<br>"
            f"Predicted: "
            f"{metrics['predicted_total_return_percent']:.4f}%"
        ),

        (
            f"{metrics['pearson_correlation']:.8f}"
            if np.isfinite(
                metrics[
                    "pearson_correlation"
                ]
            )
            else "NaN"
        ),
    ]

    fig.add_trace(

        go.Table(

            header=dict(

                values=[
                    "<b>METRIC</b>",
                    "<b>VALUE</b>",
                ],

                fill_color="#243B5A",

                font=dict(
                    color="white",
                    size=15,
                ),

                align=[
                    "left",
                    "right",
                ],

                height=45,
            ),

            cells=dict(

                values=[
                    metric_names,
                    metric_values,
                ],

                fill_color=[
                    [
                        "#F5F7F9",
                        "#FFFFFF",
                        "#F5F7F9",
                        "#FFFFFF",
                        "#F5F7F9",
                        "#FFFFFF",
                        "#F5F7F9",
                    ],
                    [
                        "#F5F7F9",
                        "#FFFFFF",
                        "#F5F7F9",
                        "#FFFFFF",
                        "#F5F7F9",
                        "#FFFFFF",
                        "#F5F7F9",
                    ],
                ],

                font=dict(
                    color="#263238",
                    size=14,
                ),

                align=[
                    "left",
                    "right",
                ],

                height=62,
            ),
        ),

        row=1,

        col=2,
    )

    # =========================================================================
    # AXES
    # =========================================================================

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
            "Actual Close − "
            "Predicted Close"
        ),

        row=3,

        col=1,
    )

    fig.update_xaxes(

        title_text="Trading Day",

        row=3,

        col=1,
    )

    # =========================================================================
    # CANDLE RANGE SLIDERS OFF
    # =========================================================================

    fig.update_xaxes(

        rangeslider_visible=False,

        row=1,

        col=1,
    )

    fig.update_xaxes(

        rangeslider_visible=False,

        row=2,

        col=1,
    )

    # =========================================================================
    # TITLE
    # =========================================================================

    fig.update_layout(

        title=dict(

            text=(
                "<b>NIFTY 50 — Kronos Base + "
                "Adaptive Rolling LoRA Dashboard</b>"
                "<br>"
                "<sup>"
                "118-Day Sliding Window | "
                "Fresh LoRA Every Window | "
                "1-Day Ahead Prediction | "
                "Optuna Tuned | "
                "Metrics Calculated ONLY on Close Returns"
                "</sup>"
            ),

            x=0.015,

            xanchor="left",

            y=0.985,

            font=dict(
                size=25,
                color="#243B5A",
            ),
        ),

        width=1900,

        height=1450,

        template="plotly_white",

        margin=dict(
            l=80,
            r=45,
            t=125,
            b=70,
        ),

        hovermode="x unified",

        showlegend=False,
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    output_path = (
        RESULTS_DIR
        / "nifty50_lora_dashboard.html"
    )

    fig.write_html(

        str(output_path),

        include_plotlyjs=True,

        full_html=True,
    )

    print()
    print("Dashboard:")
    print(output_path)

    return output_path


# =============================================================================
# SAVE ALL RESULTS
# =============================================================================

def save_results(
    predictions_df,
    metrics,
    best_params,
):

    print()
    print("=" * 100)
    print("SAVING RESULTS")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    parquet_path = (
        RESULTS_DIR
        / "nifty50_lora_rolling_predictions.parquet"
    )

    csv_path = (
        RESULTS_DIR
        / "nifty50_lora_rolling_predictions.csv"
    )

    predictions_df.to_parquet(
        parquet_path
    )

    predictions_df.to_csv(
        csv_path
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------

    metrics_path = (
        RESULTS_DIR
        / "metrics.json"
    )

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
            allow_nan=True,
        )

    # -------------------------------------------------------------------------
    # Experiment config
    # -------------------------------------------------------------------------

    config = {

        "data_path":
            str(DATA_PATH),

        "kronos_dir":
            str(KRONOS_DIR),

        "model_path":
            str(MODEL_PATH),

        "tokenizer_path":
            str(TOKENIZER_PATH),

        "device":
            DEVICE,

        "rolling_window_days":
            WINDOW_SIZE,

        "prediction_length_days":
            PRED_LEN,

        "requested_start_date":
            str(REQUESTED_START_DATE.date()),

        "max_context":
            MAX_CONTEXT,

        "optuna_trials":
            OPTUNA_TRIALS,

        "fresh_base_weights":
            True,

        "fresh_lora_every_window":
            True,

        "metrics_on":
            "daily_close_returns_only",

        "best_optuna_parameters":
            best_params,
    }

    config_path = (
        RESULTS_DIR
        / "experiment_config.json"
    )

    with open(
        config_path,
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=4,
        )

    print()
    print("Predictions Parquet:")
    print(parquet_path)

    print()
    print("Predictions CSV:")
    print(csv_path)

    print()
    print("Metrics:")
    print(metrics_path)

    print()
    print("Experiment config:")
    print(config_path)


# =============================================================================
# PRINT FINAL METRICS
# =============================================================================

def print_final_metrics(
    metrics
):

    print()
    print()
    print("=" * 100)
    print("FINAL NIFTY 50 LoRA RETURN METRICS")
    print("=" * 100)

    print()
    print(
        f"RMSE               : "
        f"{metrics['rmse']:.8f}"
    )

    print(
        f"MAE                : "
        f"{metrics['mae']:.8f}"
    )

    print(
        f"MAPE               : "
        f"{metrics['mape_percent']:.4f}%"
    )

    print(
        f"R²                 : "
        f"{metrics['r2']:.8f}"
    )

    print(
        f"Direction Accuracy : "
        f"{metrics['direction_accuracy_percent']:.4f}%"
    )

    print(
        f"Actual Return      : "
        f"{metrics['actual_total_return_percent']:.4f}%"
    )

    print(
        f"Predicted Return   : "
        f"{metrics['predicted_total_return_percent']:.4f}%"
    )

    print(
        f"Pearson Correlation: "
        f"{metrics['pearson_correlation']:.8f}"
    )

    print()
    print(
        f"Evaluated days     : "
        f"{metrics['number_of_evaluated_days']}"
    )

    print()
    print(
        "Evaluation basis   : "
        "DAILY CLOSE RETURNS ONLY"
    )

    print("=" * 100)


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # Seed
    # -------------------------------------------------------------------------

    set_seed(
        SEED
    )

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    print_configuration()

    # -------------------------------------------------------------------------
    # Clean results directory
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("CREATING CLEAN RESULTS DIRECTORY")
    print("=" * 100)

    if RESULTS_DIR.exists():

        shutil.rmtree(
            RESULTS_DIR
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        f"Fresh results directory created:\n"
        f"{RESULTS_DIR}"
    )

    # -------------------------------------------------------------------------
    # Import Kronos
    # -------------------------------------------------------------------------

    (
        Kronos,
        KronosTokenizer,
        KronosPredictor,
    ) = import_kronos()

    # -------------------------------------------------------------------------
    # Load NIFTY
    # -------------------------------------------------------------------------

    df = load_nifty_data()

    # -------------------------------------------------------------------------
    # Optuna
    # -------------------------------------------------------------------------

    best_params = run_optuna(
        Kronos=Kronos,
        KronosTokenizer=KronosTokenizer,
        KronosPredictor=KronosPredictor,
        df=df,
    )

    # -------------------------------------------------------------------------
    # Rolling LoRA
    # -------------------------------------------------------------------------

    predictions_df = run_rolling_lora(
        df=df,
        Kronos=Kronos,
        KronosTokenizer=KronosTokenizer,
        KronosPredictor=KronosPredictor,
        best_params=best_params,
    )

    # -------------------------------------------------------------------------
    # Return metrics
    # -------------------------------------------------------------------------

    (
        predictions_with_returns,
        metrics,
    ) = calculate_return_metrics(
        predictions_df
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_results(
        predictions_df=predictions_with_returns,
        metrics=metrics,
        best_params=best_params,
    )

    # -------------------------------------------------------------------------
    # Dashboard
    # -------------------------------------------------------------------------

    dashboard_path = create_dashboard(
        predictions_df=predictions_with_returns,
        metrics=metrics,
    )

    # -------------------------------------------------------------------------
    # Final metrics
    # -------------------------------------------------------------------------

    print_final_metrics(
        metrics
    )

    # -------------------------------------------------------------------------
    # Final paths
    # -------------------------------------------------------------------------

    print()
    print("=" * 100)
    print("EXPERIMENT COMPLETE")
    print("=" * 100)

    print()
    print("Results directory:")
    print(
        RESULTS_DIR
    )

    print()
    print("Dashboard:")
    print(
        dashboard_path
    )

    print()
    print("Open dashboard with:")
    print(
        f"python -m http.server 8000 "
        f"--directory '{RESULTS_DIR}'"
    )

    print()
    print("=" * 100)


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()