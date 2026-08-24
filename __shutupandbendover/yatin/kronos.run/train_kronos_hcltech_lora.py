# ================================================================
# HCLTECH + KRONOS + LoRA FINE-TUNING
# ================================================================
#
# Purpose:
#   1. Load HCLTECH daily OHLCV data
#   2. Use 400 trading days BEFORE 2022-01-01
#   3. Fine-tune Kronos-small using LoRA adapters
#   4. Freeze original Kronos weights
#   5. Train only LoRA parameters
#   6. Use validation loss for model selection
#   7. Automatically stop if validation loss stops improving
#   8. Save LoRA weights separately
#   9. Save training history
#
# IMPORTANT:
#   This implementation does NOT require PEFT.
#
# ================================================================

import os
import sys
import math
import copy
import json
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")


# ================================================================
# 1. PATHS
# ================================================================

KRONOS_ROOT = "/home/soq/Kronos"

DATA_FILE = (
    "/home/soq/yatin/yatin2/data/parquet_data/"
    "HCLTECH-EQ_1min.parquet"
)

OUTPUT_DIR = "/home/soq/yatin/kronos.run/hcltech_lora"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================
# 2. EXPERIMENT SETTINGS
# ================================================================

SYMBOL = "HCLTECH.NS"

TRAIN_END = pd.Timestamp("2022-01-01")
TEST_START = pd.Timestamp("2022-01-01")
TEST_END = pd.Timestamp("2026-08-14")

TRAIN_TRADING_DAYS = 400

LOOKBACK = 400

# Maximum prediction target used during training.
# We use a shorter horizon during LoRA training because
# the model's context is limited.
PRED_LEN = 20

# Training parameters
BATCH_SIZE = 1

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30

PATIENCE = 5

MIN_DELTA = 1e-5

# LoRA
LORA_RANK = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.05

# Gradient accumulation
GRAD_ACCUMULATION = 8

# Random seed
SEED = 42


# ================================================================
# 3. DEVICE
# ================================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 80)
print("HCLTECH KRONOS LoRA TRAINING")
print("=" * 80)

print("Device :", DEVICE)

if torch.cuda.is_available():
    print("GPU    :", torch.cuda.get_device_name(0))

print("=" * 80)


# ================================================================
# 4. RANDOM SEED
# ================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ================================================================
# 5. IMPORT KRONOS
# ================================================================

sys.path.insert(0, KRONOS_ROOT)

try:

    from model import Kronos
    from model import KronosTokenizer
    from model import KronosPredictor

except Exception as e:

    print()
    print("ERROR: Could not import Kronos.")
    print()
    print("Kronos root:")
    print(KRONOS_ROOT)
    print()
    print("Original error:")
    print(e)

    sys.exit(1)


print()
print("Kronos import successful.")


# ================================================================
# 6. LOAD DATA
# ================================================================

print()
print("=" * 80)
print("STEP 1: LOADING HCLTECH DATA")
print("=" * 80)

if not os.path.exists(DATA_FILE):

    raise FileNotFoundError(
        f"\nData file does not exist:\n{DATA_FILE}"
    )


df = pd.read_parquet(DATA_FILE)

print()
print("Original shape:", df.shape)

print("Original columns:")
print(df.columns.tolist())


# ================================================================
# 7. FIND TIMESTAMP COLUMN
# ================================================================

timestamp_column = None

possible_timestamp_columns = [
    "timestamp",
    "timestamps",
    "datetime",
    "date",
    "Date"
]

for col in possible_timestamp_columns:

    if col in df.columns:

        timestamp_column = col
        break


if timestamp_column is None:

    if isinstance(df.index, pd.DatetimeIndex):

        df = df.reset_index()

        timestamp_column = df.columns[0]

    else:

        raise ValueError(
            "Could not find timestamp/date column."
        )


print()
print("Timestamp column:", timestamp_column)


# ================================================================
# 8. NORMALIZE TIMESTAMP
# ================================================================

df[timestamp_column] = pd.to_datetime(
    df[timestamp_column],
    errors="coerce"
)


df = df.dropna(
    subset=[timestamp_column]
)


# Handle timezone safely
#
# This is important because earlier you received:
#
# TypeError:
# Invalid comparison between dtype=datetime64[ns, UTC+05:30]
# and Timestamp
#
# We convert everything to timezone-naive timestamps.
# ================================================================

try:

    if df[timestamp_column].dt.tz is not None:

        df[timestamp_column] = (
            df[timestamp_column]
            .dt.tz_localize(None)
        )

except Exception:

    pass


df = df.sort_values(
    timestamp_column
).reset_index(drop=True)


print()
print("Data start:")
print(df[timestamp_column].min())

print()
print("Data end:")
print(df[timestamp_column].max())


# ================================================================
# 9. FIND OHLCV COLUMNS
# ================================================================

def find_column(df, possible_names):

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in possible_names:

        if name.lower() in lower_map:

            return lower_map[name.lower()]

    return None


open_col = find_column(
    df,
    ["open", "Open", "OPEN"]
)

high_col = find_column(
    df,
    ["high", "High", "HIGH"]
)

low_col = find_column(
    df,
    ["low", "Low", "LOW"]
)

close_col = find_column(
    df,
    ["close", "Close", "CLOSE", "adj close", "Adj Close"]
)

volume_col = find_column(
    df,
    ["volume", "Volume", "VOLUME"]
)


required = {
    "open": open_col,
    "high": high_col,
    "low": low_col,
    "close": close_col,
    "volume": volume_col
}


print()
print("Detected columns:")

for key, value in required.items():

    print(
        f"{key:10s}: {value}"
    )


if any(
    value is None
    for value in [
        open_col,
        high_col,
        low_col,
        close_col
    ]
):

    raise ValueError(
        "\nCould not identify all OHLC columns."
    )


# ================================================================
# 10. KEEP REQUIRED COLUMNS
# ================================================================

rename_map = {

    timestamp_column: "timestamp",

    open_col: "open",

    high_col: "high",

    low_col: "low",

    close_col: "close"
}


if volume_col is not None:

    rename_map[volume_col] = "volume"


df = df.rename(
    columns=rename_map
)


if "volume" not in df.columns:

    df["volume"] = 0.0


df = df[
    [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]
].copy()


# ================================================================
# 11. REMOVE BAD VALUES
# ================================================================

for col in [
    "open",
    "high",
    "low",
    "close",
    "volume"
]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df = df.dropna(
    subset=[
        "open",
        "high",
        "low",
        "close"
    ]
)


df = df[
    df["close"] > 0
].copy()


# ================================================================
# 12. CONVERT INTRADAY DATA TO DAILY DATA
# ================================================================
#
# Your local HCLTECH file is 1-minute data.
#
# We decided to use DAILY data for this experiment.
#
# Therefore:
#
# 1-minute data
#       ↓
# group by trading date
#       ↓
# daily OHLCV
#
# ================================================================

df["date"] = df["timestamp"].dt.normalize()


daily = (
    df
    .groupby("date")
    .agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum")
    )
    .reset_index()
)


daily = daily.sort_values(
    "date"
).reset_index(drop=True)


print()
print("=" * 80)
print("DAILY DATA")
print("=" * 80)

print("Daily rows:", len(daily))

print("Start:", daily["date"].min())

print("End:", daily["date"].max())


# ================================================================
# 13. TRAINING DATA
# ================================================================
#
# We need:
#
#       400 TRADING DAYS
#
# immediately BEFORE:
#
#       2022-01-01
#
# ================================================================

before_2022 = daily[
    daily["date"] < TRAIN_END
].copy()


if len(before_2022) < TRAIN_TRADING_DAYS:

    raise ValueError(
        f"""
Only {len(before_2022)} trading days are available
before {TRAIN_END.date()}.

Need {TRAIN_TRADING_DAYS}.

Use Yahoo Finance / another historical source to obtain
older data.
"""
    )


train_data = before_2022.tail(
    TRAIN_TRADING_DAYS
).copy()


print()
print("=" * 80)
print("400-DAY TRAINING DATA")
print("=" * 80)

print(
    "Training start:",
    train_data["date"].min()
)

print(
    "Training end:",
    train_data["date"].max()
)

print(
    "Training rows:",
    len(train_data)
)


# ================================================================
# 14. TEST DATA
# ================================================================

test_data = daily[
    (daily["date"] >= TEST_START)
    &
    (daily["date"] <= TEST_END)
].copy()


print()
print("=" * 80)
print("TEST / PREDICTION DATA")
print("=" * 80)

print(
    "Test start:",
    test_data["date"].min()
)

print(
    "Test end:",
    test_data["date"].max()
)

print(
    "Test rows:",
    len(test_data)
)


if len(test_data) == 0:

    raise ValueError(
        "No test data found."
    )


# ================================================================
# 15. SAVE DATA SPLIT
# ================================================================

train_data.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_400_days.csv"
    ),
    index=False
)


test_data.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_2022_2026.csv"
    ),
    index=False
)


# ================================================================
# 16. PREPARE KRONOS INPUT
# ================================================================

MODEL_NAME = "NeoQuasar/Kronos-small"

TOKENIZER_NAME = (
    "NeoQuasar/Kronos-Tokenizer-base"
)


print()
print("=" * 80)
print("STEP 2: LOADING PRETRAINED KRONOS")
print("=" * 80)

print("Tokenizer:", TOKENIZER_NAME)

print("Model:", MODEL_NAME)


tokenizer = KronosTokenizer.from_pretrained(
    TOKENIZER_NAME
)


model = Kronos.from_pretrained(
    MODEL_NAME
)


model = model.to(DEVICE)


print()
print("Kronos model loaded.")


# ================================================================
# 17. LORA IMPLEMENTATION
# ================================================================

class LoRALinear(nn.Module):

    """
    Linear layer with a LoRA adapter.

    Original transformation:

        y = Wx + b

    LoRA transformation:

        y = Wx + scaling * B(Ax)

    W remains frozen.

    A and B are trainable.
    """

    def __init__(
        self,
        original_layer,
        rank=8,
        alpha=16,
        dropout=0.05
    ):

        super().__init__()

        if not isinstance(
            original_layer,
            nn.Linear
        ):

            raise TypeError(
                "LoRALinear requires nn.Linear"
            )


        self.in_features = (
            original_layer.in_features
        )

        self.out_features = (
            original_layer.out_features
        )


        self.rank = rank

        self.alpha = alpha

        self.scaling = (
            alpha / rank
        )


        self.dropout = nn.Dropout(
            dropout
        )


        # Frozen original layer
        self.original = original_layer

        for param in self.original.parameters():

            param.requires_grad = False


        # LoRA A
        self.lora_A = nn.Parameter(
            torch.zeros(
                rank,
                self.in_features,
                device=original_layer.weight.device,
                dtype=original_layer.weight.dtype
            )
        )


        # LoRA B
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
                device=original_layer.weight.device,
                dtype=original_layer.weight.dtype
            )
        )


        # Initialization
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5)
        )

        nn.init.zeros_(
            self.lora_B
        )


    def forward(self, x):

        original_output = self.original(x)

        lora_output = F.linear(
            self.dropout(x),
            self.lora_A
        )

        lora_output = F.linear(
            lora_output,
            self.lora_B
        )

        return (
            original_output
            +
            self.scaling * lora_output
        )


# ================================================================
# 18. WHICH LINEAR LAYERS TO APPLY LORA TO
# ================================================================

def replace_linear_layers(
    module,
    rank,
    alpha,
    dropout,
    prefix=""
):

    replaced = []

    for name, child in list(
        module.named_children()
    ):

        full_name = (
            f"{prefix}.{name}"
            if prefix
            else name
        )


        if isinstance(
            child,
            LoRALinear
        ):

            continue


        if isinstance(
            child,
            nn.Linear
        ):

            # Avoid modifying final prediction head.
            #
            # We primarily adapt transformer
            # internal projections.
            #
            if (
                "head" not in full_name.lower()
            ):

                new_layer = LoRALinear(
                    child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout
                )


                setattr(
                    module,
                    name,
                    new_layer
                )


                replaced.append(
                    full_name
                )

        else:

            replaced.extend(
                replace_linear_layers(
                    child,
                    rank,
                    alpha,
                    dropout,
                    full_name
                )
            )


    return replaced


# ================================================================
# 19. FREEZE ENTIRE KRONOS MODEL
# ================================================================

for param in model.parameters():

    param.requires_grad = False


# ================================================================
# 20. INSERT LORA
# ================================================================

print()
print("=" * 80)
print("STEP 3: INSERTING LoRA")
print("=" * 80)


lora_layers = replace_linear_layers(
    model,
    rank=LORA_RANK,
    alpha=LORA_ALPHA,
    dropout=LORA_DROPOUT
)


print()
print(
    "LoRA layers inserted:",
    len(lora_layers)
)


for name in lora_layers:

    print(
        "  ",
        name
    )


# ================================================================
# 21. TRAINABLE PARAMETER COUNT
# ================================================================

total_parameters = 0

trainable_parameters = 0


for name, param in model.named_parameters():

    total_parameters += param.numel()

    if param.requires_grad:

        trainable_parameters += (
            param.numel()
        )


print()
print("=" * 80)
print("PARAMETERS")
print("=" * 80)

print(
    "Total parameters     :",
    f"{total_parameters:,}"
)

print(
    "Trainable parameters :",
    f"{trainable_parameters:,}"
)

print(
    "Trainable percentage :",
    f"{100 * trainable_parameters / total_parameters:.4f}%"
)


# ================================================================
# 22. SAVE LORA CONFIG
# ================================================================

lora_config = {

    "base_model": MODEL_NAME,

    "tokenizer": TOKENIZER_NAME,

    "symbol": SYMBOL,

    "rank": LORA_RANK,

    "alpha": LORA_ALPHA,

    "dropout": LORA_DROPOUT,

    "lookback": LOOKBACK,

    "pred_len": PRED_LEN,

    "train_days": TRAIN_TRADING_DAYS,

    "train_end": str(
        TRAIN_END.date()
    ),

    "test_start": str(
        TEST_START.date()
    ),

    "test_end": str(
        TEST_END.date()
    )
}


with open(
    os.path.join(
        OUTPUT_DIR,
        "lora_config.json"
    ),
    "w"
) as f:

    json.dump(
        lora_config,
        f,
        indent=4
    )


# ================================================================
# 23. TRAINING DATA WINDOWS
# ================================================================
#
# We cannot train on the entire 400-day sequence as a single
# supervised example.
#
# Instead:
#
# 400 days
#    ↓
# rolling windows
#
# Example:
#
# days 1-100 -> predict next 20
# days 2-101 -> predict next 20
# days 3-102 -> predict next 20
#
# etc.
#
# ================================================================

# We use a validation split from the END of the training period.
#
# IMPORTANT:
# This validation data is STILL before 2022.
#
# The 2022-2026 period remains completely untouched.

VALIDATION_DAYS = 60


if len(train_data) <= (
    LOOKBACK + VALIDATION_DAYS
):

    raise ValueError(
        "Not enough training data for "
        "lookback + validation."
    )


training_core = train_data.iloc[
    :-VALIDATION_DAYS
].copy()


validation_data = train_data.iloc[
    -VALIDATION_DAYS:
].copy()


# ================================================================
# 24. PREPARE WINDOW LIST
# ================================================================

def create_windows(
    data,
    lookback,
    pred_len
):

    windows = []

    required = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]


    # Since we need historical context + target,
    # create windows from the available data.
    #
    # We don't train on impossible windows.

    for start in range(
        0,
        len(data) - lookback - pred_len + 1
    ):

        context = data.iloc[
            start:start + lookback
        ].copy()


        target = data.iloc[
            start + lookback:
            start + lookback + pred_len
        ].copy()


        windows.append(
            (
                context.reset_index(drop=True),
                target.reset_index(drop=True)
            )
        )


    return windows


train_windows = create_windows(
    training_core,
    LOOKBACK,
    PRED_LEN
)


val_windows = create_windows(
    validation_data,
    min(
        LOOKBACK,
        max(
            1,
            len(validation_data) - PRED_LEN
        )
    ),
    PRED_LEN
)


print()
print("=" * 80)
print("TRAINING WINDOWS")
print("=" * 80)

print(
    "Training windows:",
    len(train_windows)
)

print(
    "Validation windows:",
    len(val_windows)
)


# ================================================================
# IMPORTANT
# ================================================================
#
# With only 400 training days, using LOOKBACK=400 means there
# are too few independent windows.
#
# Therefore for the actual fine-tuning dataset we use a shorter
# context window while retaining the complete 400-day period
# as the training source.
#
# ================================================================

LOOKBACK_TRAIN = 128


train_windows = create_windows(
    training_core,
    LOOKBACK_TRAIN,
    PRED_LEN
)


# Validation:
#
# combine the end of training_core with validation data so
# validation has enough historical context.

validation_full = pd.concat(
    [
        training_core.tail(
            LOOKBACK_TRAIN
        ),
        validation_data
    ],
    ignore_index=True
)


val_windows = create_windows(
    validation_full,
    LOOKBACK_TRAIN,
    PRED_LEN
)


print()
print(
    "Final training windows:",
    len(train_windows)
)

print(
    "Final validation windows:",
    len(val_windows)
)


if len(train_windows) == 0:

    raise ValueError(
        "No training windows available."
    )


if len(val_windows) == 0:

    raise ValueError(
        "No validation windows available."
    )


# ================================================================
# 25. TOKENIZATION FUNCTION
# ================================================================

def prepare_window(
    context,
    target
):

    cols = [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]


    x_df = context[
        cols
    ].copy()


    x_timestamp = pd.Series(
        pd.to_datetime(
            context["date"]
        )
    )


    y_timestamp = pd.Series(
        pd.to_datetime(
            target["date"]
        )
    )


    return (
        x_df,
        x_timestamp,
        y_timestamp
    )


# ================================================================
# 26. KRONOS PREDICTOR
# ================================================================

predictor = KronosPredictor(
    model,
    tokenizer,
    device=DEVICE,
    max_context=512
)


# ================================================================
# 27. LOSS FUNCTION
# ================================================================
#
# Kronos's public predictor is designed primarily for inference.
#
# Therefore, to avoid making assumptions about undocumented
# internal training APIs, we use prediction-based validation.
#
# The model produces future OHLCV values.
#
# We compare predicted CLOSE with actual CLOSE.
#
# This gives us:
#
#       MSE(predicted_close, actual_close)
#
# ================================================================


def run_prediction_loss(
    context,
    target
):

    (
        x_df,
        x_timestamp,
        y_timestamp
    ) = prepare_window(
        context,
        target
    )


    pred_len = len(target)


    try:

        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.7,
            top_p=0.9,
            sample_count=1,
            verbose=False
        )

    except Exception as e:

        print()
        print(
            "Prediction error:",
            e
        )

        return None


    # Kronos prediction may return a dataframe.
    if isinstance(
        pred_df,
        tuple
    ):

        pred_df = pred_df[0]


    if not isinstance(
        pred_df,
        pd.DataFrame
    ):

        return None


    if "close" not in pred_df.columns:

        return None


    pred_close = pd.to_numeric(
        pred_df["close"],
        errors="coerce"
    ).values


    actual_close = pd.to_numeric(
        target["close"],
        errors="coerce"
    ).values


    n = min(
        len(pred_close),
        len(actual_close)
    )


    pred_close = pred_close[:n]

    actual_close = actual_close[:n]


    mask = (
        np.isfinite(pred_close)
        &
        np.isfinite(actual_close)
    )


    if mask.sum() == 0:

        return None


    # Normalize error by actual price.
    #
    # This makes the loss more stable across different
    # price levels.

    percentage_error = (
        (
            pred_close[mask]
            -
            actual_close[mask]
        )
        /
        actual_close[mask]
    )


    mse = np.mean(
        percentage_error ** 2
    )


    return float(mse)


# ================================================================
# 28. NOTE ABOUT GRADIENTS
# ================================================================
#
# IMPORTANT:
#
# predictor.predict() is an autoregressive sampling API and
# internally uses inference operations.
#
# It is NOT a training API.
#
# Therefore it cannot directly backpropagate a loss into LoRA.
#
# For true gradient-based LoRA training we need to use Kronos's
# internal tokenizer -> model -> logits training path.
#
# We therefore inspect the model interface before starting.
#
# ================================================================


print()
print("=" * 80)
print("CHECKING KRONOS TRAINING INTERFACE")
print("=" * 80)


print(
    "Model forward signature:"
)

try:

    import inspect

    print(
        inspect.signature(
            model.forward
        )
    )

except Exception as e:

    print(
        "Could not inspect:",
        e
    )


# ================================================================
# 29. MODEL MODULE SUMMARY
# ================================================================

print()
print("Kronos model modules:")

for name, module in model.named_modules():

    if isinstance(
        module,
        LoRALinear
    ):

        print(
            name
        )


# ================================================================
# 30. SAVE INITIAL LORA WEIGHTS
# ================================================================

def get_lora_state_dict(
    model
):

    state = {}

    for name, param in model.named_parameters():

        if (
            "lora_A" in name
            or
            "lora_B" in name
        ):

            state[name] = (
                param.detach()
                .cpu()
                .clone()
            )


    return state


initial_lora = get_lora_state_dict(
    model
)


torch.save(
    initial_lora,
    os.path.join(
        OUTPUT_DIR,
        "lora_initial.pt"
    )
)


print()
print(
    "Initial LoRA weights saved."
)


# ================================================================
# 31. IMPORTANT SAFETY STOP
# ================================================================
#
# We do NOT silently perform fake training.
#
# The public Kronos Predictor API is inference-oriented.
#
# The official repository has its own fine-tuning implementation.
# We should connect our LoRA adapters to that internal training
# path rather than claiming that prediction-loss calculations
# are actually updating LoRA.
#
# ================================================================


print()
print("=" * 80)
print("NEXT STEP REQUIRED")
print("=" * 80)

print(
    """
The Kronos model and LoRA adapters loaded successfully.

However, the public KronosPredictor.predict() API is an
inference API. It does not expose a differentiable training
loss.

I have intentionally NOT performed fake "training" here.

The correct next step is to connect the LoRA layers to
Kronos's official finetune/train_predictor.py training path.

This prevents us from saving LoRA weights that were never
actually trained.
"""
)


print()
print(
    "LoRA output directory:"
)

print(
    OUTPUT_DIR
)

print()
print("=" * 80)