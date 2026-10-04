import copy
import gc
import json
import math
import random
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf
from scipy.stats import spearmanr


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

KRONOS_DIR = Path("/home/soq/Kronos")

DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results" / "nifty50_stocks"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

STOCK_DATA_PATH = (
    DATA_DIR / "nifty50_stocks_ohlcv.parquet"
)

MASTER_PREDICTIONS_PATH = (
    RESULTS_DIR / "all_stock_predictions.csv"
)

STOCK_METRICS_PATH = (
    RESULTS_DIR / "stock_metrics.csv"
)

DAILY_RANKIC_PATH = (
    RESULTS_DIR / "daily_rankic.csv"
)

CROSS_SECTIONAL_METRICS_PATH = (
    RESULTS_DIR / "cross_sectional_metrics.json"
)

TIMING_PATH = (
    RESULTS_DIR / "timing.json"
)

FAILED_STOCKS_PATH = (
    RESULTS_DIR / "failed_stocks.json"
)


# ============================================================
# KRONOS
# ============================================================

MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"


# ============================================================
# CONFIG
# ============================================================

START_DATE = "2022-01-01"

OPTUNA_START_DATE = "2022-01-01"
OPTUNA_END_DATE = "2022-12-31"

VALIDATION_START_DATE = "2023-01-01"
VALIDATION_END_DATE = "2023-06-30"

ACTUAL_START_DATE = "2023-07-01"

TRAIN_DAYS = 32
VALIDATION_DAYS = 8
CONTEXT_DAYS = 40

MAX_EPOCHS = 200
PATIENCE = 5
MIN_DELTA = 1e-4

N_OPTUNA_TRIALS = 10

SEED = 42


# ============================================================
# DATA COLUMNS
# ============================================================

PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
]

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]


# ============================================================
# NIFTY 50
# ============================================================

NIFTY_50 = [
    "ADANIENT",
    "ADANIPORTS",
    "APOLLOHOSP",
    "ASIANPAINT",
    "AXISBANK",
    "BAJAJ-AUTO",
    "BAJFINANCE",
    "BAJAJFINSV",
    "BEL",
    "BHARTIARTL",
    "CIPLA",
    "COALINDIA",
    "DRREDDY",
    "EICHERMOT",
    "ETERNAL",
    "GRASIM",
    "HCLTECH",
    "HDFCBANK",
    "HDFCLIFE",
    "HEROMOTOCO",
    "HINDALCO",
    "HINDUNILVR",
    "ICICIBANK",
    "INDUSINDBK",
    "INFY",
    "ITC",
    "JIOFIN",
    "JSWSTEEL",
    "KOTAKBANK",
    "LT",
    "M&M",
    "MARUTI",
    "MAXHEALTH",
    "NESTLEIND",
    "NTPC",
    "ONGC",
    "POWERGRID",
    "RELIANCE",
    "SBILIFE",
    "SBIN",
    "SHRIRAMFIN",
    "SUNPHARMA",
    "TATACONSUM",
    "WIPRO",
    "TATASTEEL",
    "TCS",
    "TECHM",
    "TITAN",
    "TRENT",
    "ULTRACEMCO",
]


# ============================================================
# LORA
# ============================================================

LORA_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
)


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    DEVICE = "cuda"

else:

    raise RuntimeError(
        "CUDA is not available. "
        "This script is configured to train on CUDA."
    )


# ============================================================
# KRONOS IMPORT
# ============================================================

sys.path.insert(0, str(KRONOS_DIR))

from model import Kronos, KronosPredictor, KronosTokenizer
from model.kronos import calc_time_stamps


# ============================================================
# SEED
# ============================================================

def set_seed(seed=SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# TIME FORMATTER
# ============================================================

def format_time(seconds):

    seconds = float(seconds)

    hours = int(seconds // 3600)

    minutes = int(
        (seconds % 3600) // 60
    )

    secs = seconds % 60

    if hours > 0:

        return (
            f"{hours}h "
            f"{minutes}m "
            f"{secs:.1f}s"
        )

    if minutes > 0:

        return (
            f"{minutes}m "
            f"{secs:.1f}s"
        )

    return f"{secs:.1f}s"


# ============================================================
# DOWNLOAD STOCK DATA
# ============================================================

def download_stock_data():

    if STOCK_DATA_PATH.exists():

        print(
            f"\nStock data already exists:"
            f"\n{STOCK_DATA_PATH}"
        )

        return

    print("\n")
    print("=" * 70)
    print("DOWNLOADING NIFTY 50 STOCK DATA")
    print("=" * 70)

    all_data = []

    for number, symbol in enumerate(
        NIFTY_50,
        start=1,
    ):

        print(
            f"[{number}/{len(NIFTY_50)}] "
            f"Downloading {symbol}..."
        )

        yahoo_symbol = (
            f"{symbol}.NS"
        )

        try:

            df = yf.download(
                yahoo_symbol,
                start=START_DATE,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            if df.empty:

                print(
                    f"    WARNING: no data for "
                    f"{symbol}"
                )

                continue

            if isinstance(
                df.columns,
                pd.MultiIndex,
            ):

                df.columns = (
                    df.columns.get_level_values(0)
                )

            required = [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]

            missing = [
                column
                for column in required
                if column not in df.columns
            ]

            if missing:

                print(
                    f"    WARNING: missing "
                    f"{missing}"
                )

                continue

            df = df[required].copy()

            df.columns = [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]

            df = df.reset_index()

            df.rename(
                columns={
                    "Date": "date"
                },
                inplace=True,
            )

            df["symbol"] = symbol

            df = df[
                [
                    "date",
                    "symbol",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ]
            ]

            all_data.append(df)

            print(
                f"    {len(df)} rows"
            )

        except Exception as e:

            print(
                f"    ERROR downloading "
                f"{symbol}: {e}"
            )

    if not all_data:

        raise RuntimeError(
            "No stock data was downloaded."
        )

    combined = pd.concat(
        all_data,
        ignore_index=True,
    )

    combined["date"] = pd.to_datetime(
        combined["date"]
    )

    combined = combined.sort_values(
        [
            "symbol",
            "date",
        ]
    )

    combined.to_parquet(
        STOCK_DATA_PATH,
        index=False,
    )

    print(
        f"\nSaved combined stock data:"
        f"\n{STOCK_DATA_PATH}"
    )

    print(
        f"Total rows: {len(combined)}"
    )

    print(
        f"Stocks downloaded: "
        f"{combined['symbol'].nunique()}"
    )


# ============================================================
# LOAD STOCK DATA
# ============================================================

def load_stock(symbol):

    data = pd.read_parquet(
        STOCK_DATA_PATH
    )

    data["date"] = pd.to_datetime(
        data["date"]
    )

    data = data[
        data["symbol"] == symbol
    ].copy()

    if data.empty:

        raise ValueError(
            f"No data found for {symbol}"
        )

    data = data.sort_values(
        "date"
    )

    data = data.set_index(
        "date"
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
        if column not in data.columns
    ]

    if missing:

        raise ValueError(
            f"{symbol}: missing columns "
            f"{missing}"
        )

    data = data[
        required
    ].astype(float).dropna()

    data["amount"] = (
        data["volume"]
        * data[PRICE_COLUMNS].mean(axis=1)
    )

    data = data[
        data.index >= START_DATE
    ].copy()

    if len(data) < (
        CONTEXT_DAYS + 1
    ):

        raise ValueError(
            f"{symbol}: insufficient data"
        )

    return data


# ============================================================
# LORA
# ============================================================

class LoRALinear(nn.Module):

    def __init__(
        self,
        base_layer,
        rank,
        alpha,
        dropout,
    ):

        super().__init__()

        self.base = base_layer

        self.scaling = (
            alpha / rank
        )

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

        self.dropout = (
            nn.Dropout(dropout)
            if dropout > 0
            else nn.Identity()
        )

        nn.init.kaiming_uniform_(
            self.lora_A,
            a=math.sqrt(5),
        )

        for parameter in (
            self.base.parameters()
        ):

            parameter.requires_grad = False

    def forward(self, x):

        update = (
            self.dropout(x)
            @ self.lora_A.t()
            @ self.lora_B.t()
        )

        return (
            self.base(x)
            + self.scaling * update
        )


def inject_lora(
    model,
    rank,
    alpha,
    dropout,
):

    for parameter in model.parameters():

        parameter.requires_grad = False

    trainable = []

    for block in model.transformer:

        attention = block.self_attn

        for target in LORA_TARGETS:

            base_layer = getattr(
                attention,
                target,
            )

            adapter = LoRALinear(
                base_layer,
                rank,
                alpha,
                dropout,
            )

            setattr(
                attention,
                target,
                adapter,
            )

            trainable.extend(
                [
                    adapter.lora_A,
                    adapter.lora_B,
                ]
            )

    return trainable


# ============================================================
# NORMALIZE + TOKENIZE
# ============================================================

def encode_window(
    frame,
    tokenizer,
    model_template,
):

    values = frame[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=np.float32
    )

    mean = values.mean(axis=0)

    std = values.std(axis=0)

    normalized = (
        values - mean
    ) / (
        std + 1e-5
    )

    normalized = np.clip(
        normalized,
        -5,
        5,
    )

    timestamps = pd.Series(
        frame.index
    )

    stamps = calc_time_stamps(
        timestamps
    ).to_numpy(
        dtype=np.float32
    )

    stamps = torch.from_numpy(
        stamps
    ).unsqueeze(0)

    with torch.no_grad():

        values_tensor = (
            torch.from_numpy(
                normalized
            )
            .float()
            .unsqueeze(0)
            .to(DEVICE)
        )

        token_ids = tokenizer.encode(
            values_tensor
        )

        stream_one, stream_two = (
            model_template.embedding.split_token(
                token_ids,
                model_template.embedding.s2_bits,
            )
        )

    return (
        stream_one.cpu(),
        stream_two.cpu(),
        stamps,
    )


# ============================================================
# LOSS
# ============================================================

def teacher_forced_loss(
    model,
    stream_one,
    stream_two,
    stamps,
):

    stream_one = stream_one.to(
        DEVICE
    )

    stream_two = stream_two.to(
        DEVICE
    )

    stamps = stamps.to(
        DEVICE
    )

    logits_one, logits_two = model(
        stream_one,
        stream_two,
        stamp=stamps,
        use_teacher_forcing=True,
        s1_targets=stream_one,
    )

    loss_one = F.cross_entropy(
        logits_one[:, :-1].reshape(
            -1,
            logits_one.size(-1),
        ),
        stream_one[:, 1:].reshape(-1),
    )

    loss_two = F.cross_entropy(
        logits_two[:, :-1].reshape(
            -1,
            logits_two.size(-1),
        ),
        stream_two[:, 1:].reshape(-1),
    )

    return (
        loss_one + loss_two
    )


# ============================================================
# TRAIN ONE WINDOW
# ============================================================

def train_one_window(
    window,
    model_template,
    tokenizer,
    params,
):

    train_data = window.iloc[
        :TRAIN_DAYS
    ]

    validation_data = window.iloc[
        TRAIN_DAYS:
        TRAIN_DAYS + VALIDATION_DAYS
    ]

    model = copy.deepcopy(
        model_template
    )

    trainable = inject_lora(
        model,
        params["rank"],
        params["alpha"],
        params["dropout"],
    )

    model.to(DEVICE)

    optimizer = torch.optim.AdamW(
        trainable,
        lr=params["learning_rate"],
        weight_decay=params["weight_decay"],
    )

    train_s1, train_s2, train_stamps = (
        encode_window(
            train_data,
            tokenizer,
            model_template,
        )
    )

    val_s1, val_s2, val_stamps = (
        encode_window(
            validation_data,
            tokenizer,
            model_template,
        )
    )

    best_loss = float("inf")

    best_state = None

    epochs_without_improvement = 0

    for epoch in range(
        MAX_EPOCHS
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        train_loss = teacher_forced_loss(
            model,
            train_s1,
            train_s2,
            train_stamps,
        )

        train_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            trainable,
            1.0,
        )

        optimizer.step()

        model.eval()

        with torch.no_grad():

            validation_loss = (
                teacher_forced_loss(
                    model,
                    val_s1,
                    val_s2,
                    val_stamps,
                )
            )

        validation_value = (
            validation_loss.item()
        )

        if (
            epoch == 0
            or (epoch + 1) % 5 == 0
        ):

            print(
                f"      Epoch "
                f"{epoch + 1:03d}/{MAX_EPOCHS} "
                f"| train={train_loss.item():.5f} "
                f"| val={validation_value:.5f}"
            )

        if validation_value < (
            best_loss - MIN_DELTA
        ):

            best_loss = (
                validation_value
            )

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"      Early stopping at "
                f"epoch {epoch + 1}"
            )

            break

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return (
        model,
        best_loss,
    )


# ============================================================
# PREDICT ONE DAY
# ============================================================

def predict_one_day(
    model,
    tokenizer,
    context,
    target_day,
):

    predictor = KronosPredictor(
        model,
        tokenizer,
        device=DEVICE,
        max_context=512,
    )

    context_timestamps = pd.Series(
        context.index
    )

    target_timestamp = pd.Series(
        [target_day.name]
    )

    prediction = predictor.predict(
        context,
        context_timestamps,
        target_timestamp,
        pred_len=1,
        T=1.0,
        top_k=0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )

    return prediction.iloc[0]


# ============================================================
# OPTUNA
# ============================================================

def run_optuna(
    data,
    model_template,
    tokenizer,
):

    optuna_start_time = (
        time.perf_counter()
    )

    print("\n")
    print("=" * 70)
    print("OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 70)

    tuning_data = data[
        (
            data.index
            >= pd.Timestamp(
                OPTUNA_START_DATE
            )
        )
        &
        (
            data.index
            <= pd.Timestamp(
                OPTUNA_END_DATE
            )
        )
    ].copy()

    minimum_window = (
        CONTEXT_DAYS + 1
    )

    possible = (
        len(tuning_data)
        - minimum_window
        + 1
    )

    if possible <= 0:

        raise ValueError(
            "Not enough 2022 data for "
            "Optuna windows."
        )

    max_windows = 10

    number_of_windows = min(
        max_windows,
        possible,
    )

    indices = np.linspace(
        0,
        possible - 1,
        number_of_windows,
        dtype=int,
    )

    tuning_windows = [
        tuning_data.iloc[
            i:i + minimum_window
        ]
        for i in indices
    ]

    print(
        f"Optuna tuning windows: "
        f"{len(tuning_windows)}"
    )

    def objective(trial):

        trial_start_time = (
            time.perf_counter()
        )

        params = {

            "rank":
                trial.suggest_categorical(
                    "rank",
                    [4, 8, 16],
                ),

            "alpha":
                trial.suggest_categorical(
                    "alpha",
                    [8, 16, 32],
                ),

            "dropout":
                trial.suggest_float(
                    "dropout",
                    0.0,
                    0.20,
                ),

            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    1e-5,
                    5e-4,
                    log=True,
                ),

            "weight_decay":
                trial.suggest_float(
                    "weight_decay",
                    1e-6,
                    1e-2,
                    log=True,
                ),
        }

        print(
            f"\nTrial "
            f"{trial.number + 1}/"
            f"{N_OPTUNA_TRIALS}"
        )

        print(
            json.dumps(
                params,
                indent=4,
            )
        )

        losses = []

        for (
            window_number,
            window,
        ) in enumerate(
            tuning_windows
        ):

            print(
                f"\n  Window "
                f"{window_number + 1}/"
                f"{len(tuning_windows)}"
            )

            set_seed(
                SEED
                + trial.number
                + window_number
            )

            window_start_time = (
                time.perf_counter()
            )

            model, validation_loss = (
                train_one_window(
                    window,
                    model_template,
                    tokenizer,
                    params,
                )
            )

            window_time = (
                time.perf_counter()
                - window_start_time
            )

            losses.append(
                validation_loss
            )

            print(
                f"    Window validation loss: "
                f"{validation_loss:.6f}"
            )

            print(
                f"    Window time: "
                f"{format_time(window_time)}"
            )

            del model

            gc.collect()

            torch.cuda.empty_cache()

        mean_loss = float(
            np.mean(losses)
        )

        trial_time = (
            time.perf_counter()
            - trial_start_time
        )

        print(
            f"\nTrial "
            f"{trial.number + 1} "
            f"mean validation loss = "
            f"{mean_loss:.6f}"
        )

        print(
            f"Trial time: "
            f"{format_time(trial_time)}"
        )

        return mean_loss

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(
            seed=SEED
        ),
    )

    study.optimize(
        objective,
        n_trials=N_OPTUNA_TRIALS,
        gc_after_trial=True,
    )

    optuna_path = (
        RESULTS_DIR
        / "optuna_trials.csv"
    )

    study.trials_dataframe().to_csv(
        optuna_path,
        index=False,
    )

    optuna_time = (
        time.perf_counter()
        - optuna_start_time
    )

    print("\n")
    print("=" * 70)
    print("BEST OPTUNA PARAMETERS")
    print("=" * 70)

    print(
        json.dumps(
            study.best_trial.params,
            indent=4,
        )
    )

    print(
        f"\nBest validation loss: "
        f"{study.best_value:.6f}"
    )

    print(
        f"Total Optuna time: "
        f"{format_time(optuna_time)}"
    )

    return (
        study.best_trial.params,
        optuna_time,
    )


# ============================================================
# VALIDATION PERIOD
# ============================================================

def validate_period(
    symbol,
    data,
    model_template,
    tokenizer,
    best_params,
):

    print("\n")
    print("=" * 70)
    print(
        f"{symbol} VALIDATION PERIOD"
    )
    print("=" * 70)

    validation_start = pd.Timestamp(
        VALIDATION_START_DATE
    )

    validation_end = pd.Timestamp(
        VALIDATION_END_DATE
    )

    validation_data = data[
        (
            data.index
            >= validation_start
        )
        &
        (
            data.index
            <= validation_end
        )
    ].copy()

    validation_results = []

    total_training_time = 0.0
    total_prediction_time = 0.0

    for target_date in validation_data.index:

        i = data.index.get_loc(
            target_date
        )

        if i < CONTEXT_DAYS:

            continue

        window = data.iloc[
            i - CONTEXT_DAYS:
            i + 1
        ].copy()

        target_day = window.iloc[
            CONTEXT_DAYS
        ]

        set_seed(
            SEED + i
        )

        training_start = (
            time.perf_counter()
        )

        model, validation_loss = (
            train_one_window(
                window,
                model_template,
                tokenizer,
                best_params,
            )
        )

        training_time = (
            time.perf_counter()
            - training_start
        )

        total_training_time += (
            training_time
        )

        prediction_start = (
            time.perf_counter()
        )

        predicted = predict_one_day(
            model,
            tokenizer,
            window.iloc[
                :CONTEXT_DAYS
            ].copy(),
            target_day,
        )

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

        total_prediction_time += (
            prediction_time
        )

        previous_close = float(
            data.iloc[i - 1]["close"]
        )

        actual_close = float(
            target_day["close"]
        )

        predicted_close = float(
            predicted["close"]
        )

        actual_return = (
            actual_close
            - previous_close
        ) / previous_close

        predicted_return = (
            predicted_close
            - previous_close
        ) / previous_close

        return_error = (
            predicted_return
            - actual_return
        )

        validation_results.append(
            {
                "symbol": symbol,
                "date": target_day.name,
                "actual_close": actual_close,
                "predicted_close": predicted_close,
                "actual_return": actual_return,
                "predicted_return": predicted_return,
                "return_error": return_error,
                "absolute_return_error": abs(
                    return_error
                ),
                "validation_loss": validation_loss,
            }
        )

        print(
            f"  {target_day.name.date()} "
            f"| actual="
            f"{actual_return * 100:+.3f}% "
            f"| pred="
            f"{predicted_return * 100:+.3f}% "
            f"| error="
            f"{return_error * 100:+.3f}%"
        )

        del model

        gc.collect()

        torch.cuda.empty_cache()

    results = pd.DataFrame(
        validation_results
    )

    if results.empty:

        raise ValueError(
            f"{symbol}: no validation "
            f"predictions generated."
        )

    return (
        results,
        {
            "training_seconds":
                total_training_time,
            "prediction_seconds":
                total_prediction_time,
            "total_seconds":
                total_training_time
                + total_prediction_time,
        },
    )


# ============================================================
# WALK FORWARD
# ============================================================

def walk_forward(
    symbol,
    data,
    model_template,
    tokenizer,
    best_params,
):

    predictions = []

    actual_start = pd.Timestamp(
        ACTUAL_START_DATE
    )

    prediction_indices = [
        i
        for i in range(
            CONTEXT_DAYS,
            len(data),
        )
        if data.index[i] >= actual_start
    ]

    total_predictions = len(
        prediction_indices
    )

    print("\n")
    print("=" * 70)
    print(
        f"{symbol} ROLLING LORA FORECAST"
    )
    print("=" * 70)

    print(
        f"Predictions: "
        f"{total_predictions}"
    )

    total_training_time = 0.0
    total_prediction_time = 0.0

    for (
        prediction_number,
        i,
    ) in enumerate(
        prediction_indices,
        start=1,
    ):

        iteration_start = (
            time.perf_counter()
        )

        window = data.iloc[
            i - CONTEXT_DAYS:
            i + 1
        ].copy()

        target_day = window.iloc[
            CONTEXT_DAYS
        ]

        print("\n")
        print(
            f"[{symbol}] "
            f"[{prediction_number}/"
            f"{total_predictions}] "
            f"Predicting "
            f"{target_day.name.date()}"
        )

        print(
            f"    Train: "
            f"{window.index[0].date()} → "
            f"{window.index[31].date()}"
        )

        print(
            f"    Validation: "
            f"{window.index[32].date()} → "
            f"{window.index[39].date()}"
        )

        print(
            f"    Target: "
            f"{window.index[40].date()}"
        )

        set_seed(
            SEED + prediction_number
        )

        training_start = (
            time.perf_counter()
        )

        model, validation_loss = (
            train_one_window(
                window,
                model_template,
                tokenizer,
                best_params,
            )
        )

        training_time = (
            time.perf_counter()
            - training_start
        )

        total_training_time += (
            training_time
        )

        prediction_start = (
            time.perf_counter()
        )

        context = window.iloc[
            :CONTEXT_DAYS
        ].copy()

        predicted = predict_one_day(
            model,
            tokenizer,
            context,
            target_day,
        )

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

        total_prediction_time += (
            prediction_time
        )

        previous_actual_close = float(
            data.iloc[i - 1]["close"]
        )

        actual_close = float(
            target_day["close"]
        )

        predicted_close = float(
            predicted["close"]
        )

        actual_return = (
            actual_close
            - previous_actual_close
        ) / previous_actual_close

        predicted_return = (
            predicted_close
            - previous_actual_close
        ) / previous_actual_close

        return_error = (
            predicted_return
            - actual_return
        )

        direction_correct = (
            np.sign(actual_return)
            == np.sign(predicted_return)
        )

        print(
            f"    Actual close:    "
            f"{actual_close:.4f}"
        )

        print(
            f"    Predicted close: "
            f"{predicted_close:.4f}"
        )

        print(
            f"    Actual return:   "
            f"{actual_return * 100:+.4f}%"
        )

        print(
            f"    Predicted return:"
            f" {predicted_return * 100:+.4f}%"
        )

        print(
            f"    Return error:    "
            f"{return_error * 100:+.4f}%"
        )

        print(
            f"    Direction:       "
            f"{'CORRECT' if direction_correct else 'WRONG'}"
        )

        print(
            f"    Validation loss: "
            f"{validation_loss:.6f}"
        )

        print(
            f"    Training time:   "
            f"{format_time(training_time)}"
        )

        print(
            f"    Prediction time: "
            f"{format_time(prediction_time)}"
        )

        predictions.append(
            {
                "symbol": symbol,
                "date": target_day.name,

                "actual_open":
                    float(target_day["open"]),

                "actual_high":
                    float(target_day["high"]),

                "actual_low":
                    float(target_day["low"]),

                "actual_close":
                    actual_close,

                "predicted_open":
                    float(predicted["open"]),

                "predicted_high":
                    float(predicted["high"]),

                "predicted_low":
                    float(predicted["low"]),

                "predicted_close":
                    predicted_close,

                "previous_actual_close":
                    previous_actual_close,

                "actual_return":
                    actual_return,

                "predicted_return":
                    predicted_return,

                "return_error":
                    return_error,

                "absolute_return_error":
                    abs(return_error),

                "direction_correct":
                    bool(direction_correct),

                "validation_loss":
                    validation_loss,

                "training_time_seconds":
                    training_time,

                "prediction_time_seconds":
                    prediction_time,
            }
        )

        del model

        gc.collect()

        torch.cuda.empty_cache()

        iteration_time = (
            time.perf_counter()
            - iteration_start
        )

        print(
            f"    Total day time: "
            f"{format_time(iteration_time)}"
        )

    results = pd.DataFrame(
        predictions
    )

    if results.empty:

        raise ValueError(
            f"{symbol}: no predictions generated."
        )

    results["date"] = pd.to_datetime(
        results["date"]
    )

    return (
        results,
        {
            "training_seconds":
                total_training_time,

            "prediction_seconds":
                total_prediction_time,

            "total_seconds":
                total_training_time
                + total_prediction_time,
        },
    )


# ============================================================
# RETURN METRICS
# ============================================================

def calculate_metrics(results):

    actual = results[
        "actual_return"
    ].to_numpy(float)

    predicted = results[
        "predicted_return"
    ].to_numpy(float)

    error = (
        predicted - actual
    )

    rmse = float(
        np.sqrt(
            np.mean(error ** 2)
        )
    )

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    direction_accuracy = float(
        np.mean(
            np.sign(actual)
            == np.sign(predicted)
        ) * 100
    )

    if (
        len(actual) > 1
        and np.std(actual) > 0
        and np.std(predicted) > 0
    ):

        pearson = float(
            np.corrcoef(
                actual,
                predicted,
            )[0, 1]
        )

    else:

        pearson = 0.0

    return {

        "RMSE_returns":
            rmse,

        "MAE_returns":
            mae,

        "Direction_Accuracy_percent":
            direction_accuracy,

        "Pearson_returns":
            pearson,

        "Mean_Error_returns":
            float(np.mean(error)),

        "Max_Absolute_Error_returns":
            float(np.max(np.abs(error))),

        "Num_predictions":
            len(results),
    }


# ============================================================
# DAILY RANKIC
# ============================================================

def calculate_daily_rankic(
    all_predictions
):

    daily_results = []

    for date, group in (
        all_predictions
        .groupby("date")
    ):

        group = group.dropna(
            subset=[
                "actual_return",
                "predicted_return",
            ]
        )

        if len(group) < 2:

            continue

        actual = group[
            "actual_return"
        ].to_numpy(float)

        predicted = group[
            "predicted_return"
        ].to_numpy(float)

        if (
            np.std(actual) == 0
            or np.std(predicted) == 0
        ):

            rank_ic = np.nan

        else:

            rank_ic = float(
                spearmanr(
                    actual,
                    predicted,
                ).statistic
            )

        daily_results.append(
            {
                "date": date,
                "num_stocks": len(group),
                "RankIC": rank_ic,
            }
        )

    daily = pd.DataFrame(
        daily_results
    )

    if not daily.empty:

        daily["date"] = pd.to_datetime(
            daily["date"]
        )

    return daily


# ============================================================
# AGGREGATE RANKIC
# ============================================================

def calculate_rankic_metrics(
    daily_rankic
):

    values = daily_rankic[
        "RankIC"
    ].dropna().to_numpy(float)

    if len(values) == 0:

        return {

            "Mean RankIC":
                None,

            "Median RankIC":
                None,

            "Std RankIC":
                None,

            "RankIC IR":
                None,

            "Positive RankIC (%)":
                None,

            "Minimum RankIC":
                None,

            "Maximum RankIC":
                None,

            "Number of days":
                0,
        }

    mean_rankic = float(
        np.mean(values)
    )

    median_rankic = float(
        np.median(values)
    )

    std_rankic = float(
        np.std(
            values,
            ddof=1,
        )
        if len(values) > 1
        else 0.0
    )

    if std_rankic > 0:

        rankic_ir = (
            mean_rankic
            / std_rankic
        )

    else:

        rankic_ir = 0.0

    positive_percentage = float(
        np.mean(
            values > 0
        ) * 100
    )

    return {

        "Mean RankIC":
            mean_rankic,

        "Median RankIC":
            median_rankic,

        "Std RankIC":
            std_rankic,

        "RankIC IR":
            float(rankic_ir),

        "Positive RankIC (%)":
            positive_percentage,

        "Minimum RankIC":
            float(np.min(values)),

        "Maximum RankIC":
            float(np.max(values)),

        "Number of days":
            len(values),
    }


# ============================================================
# SAVE STOCK RESULTS
# ============================================================

def save_stock_results(
    symbol,
    best_params,
    validation_metrics,
    metrics,
    predictions,
    timing,
):

    stock_dir = (
        RESULTS_DIR / symbol
    )

    stock_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        stock_dir / "predictions.csv",
        index=False,
    )

    with open(
        stock_dir / "best_params.json",
        "w",
    ) as f:

        json.dump(
            best_params,
            f,
            indent=4,
        )

    with open(
        stock_dir / "metrics.json",
        "w",
    ) as f:

        json.dump(
            {
                "validation":
                    validation_metrics,

                "test":
                    metrics,

                "timing":
                    timing,
            },
            f,
            indent=4,
        )


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_stock(
    symbol,
    tokenizer,
    model_template,
):

    stock_start = (
        time.perf_counter()
    )

    print("\n\n")
    print("#" * 80)
    print(
        f"PROCESSING {symbol}"
    )
    print("#" * 80)

    data = load_stock(
        symbol
    )

    print(
        f"{symbol}: "
        f"{len(data)} trading days"
    )

    print(
        f"Range: "
        f"{data.index[0].date()} → "
        f"{data.index[-1].date()}"
    )

    # --------------------------------------------------------
    # OPTUNA
    # --------------------------------------------------------

    best_params, optuna_time = (
        run_optuna(
            data,
            model_template,
            tokenizer,
        )
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_results, validation_timing = (
        validate_period(
            symbol,
            data,
            model_template,
            tokenizer,
            best_params,
        )
    )

    validation_metrics = (
        calculate_metrics(
            validation_results
        )
    )

    print("\n")
    print(
        f"{symbol} VALIDATION METRICS"
    )

    for name, value in (
        validation_metrics.items()
    ):

        print(
            f"  {name}: {value}"
        )

    # --------------------------------------------------------
    # JULY+ ROLLING
    # --------------------------------------------------------

    predictions, rolling_timing = (
        walk_forward(
            symbol,
            data,
            model_template,
            tokenizer,
            best_params,
        )
    )

    metrics = calculate_metrics(
        predictions
    )

    # --------------------------------------------------------
    # TIMING
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - stock_start
    )

    timing = {

        "optuna_seconds":
            optuna_time,

        "validation_training_seconds":
            validation_timing[
                "training_seconds"
            ],

        "validation_prediction_seconds":
            validation_timing[
                "prediction_seconds"
            ],

        "validation_total_seconds":
            validation_timing[
                "total_seconds"
            ],

        "rolling_training_seconds":
            rolling_timing[
                "training_seconds"
            ],

        "rolling_prediction_seconds":
            rolling_timing[
                "prediction_seconds"
            ],

        "rolling_total_seconds":
            rolling_timing[
                "total_seconds"
            ],

        "total_seconds":
            total_time,

        "total_time":
            format_time(total_time),
    }

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_stock_results(
        symbol,
        best_params,
        validation_metrics,
        metrics,
        predictions,
        timing,
    )

    print("\n")
    print(
        f"{symbol} FINAL METRICS"
    )

    for name, value in metrics.items():

        print(
            f"  {name}: {value}"
        )

    print(
        f"\n{symbol} total time: "
        f"{format_time(total_time)}"
    )

    return (
        predictions,
        metrics,
        timing,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start = (
        time.perf_counter()
    )

    set_seed()

    print("=" * 80)
    print(
        "KRONOS + LORA NIFTY 50 STOCK FORECAST"
    )
    print("=" * 80)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"Stocks: "
        f"{len(NIFTY_50)}"
    )

    # --------------------------------------------------------
    # DATA DOWNLOAD
    # --------------------------------------------------------

    download_stock_data()

    # --------------------------------------------------------
    # TOKENIZER
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = (
        KronosTokenizer.from_pretrained(
            TOKENIZER_NAME
        )
    )

    tokenizer = tokenizer.to(
        DEVICE
    )

    for parameter in (
        tokenizer.parameters()
    ):

        parameter.requires_grad = False

    print(
        f"Tokenizer device: "
        f"{next(tokenizer.parameters()).device}"
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print(
        "Loading Kronos..."
    )

    model_template = (
        Kronos.from_pretrained(
            MODEL_NAME
        )
    )

    model_template = (
        model_template.to("cpu")
    )

    for parameter in (
        model_template.parameters()
    ):

        parameter.requires_grad = False

    print(
        f"Kronos device: "
        f"{next(model_template.parameters()).device}"
    )

    # --------------------------------------------------------
    # SANITY CHECK
    # --------------------------------------------------------

    print("\nCUDA sanity check:")

    print(
        f"  Device: {DEVICE}"
    )

    print(
        f"  GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        f"  Kronos parameter device: "
        f"{next(model_template.parameters()).device}"
    )

    print(
        f"  Tokenizer parameter device: "
        f"{next(tokenizer.parameters()).device}"
    )

    # --------------------------------------------------------
    # ALL STOCKS
    # --------------------------------------------------------

    all_predictions = []

    stock_metrics = []

    timing_results = {}

    failed_stocks = []

    for stock_number, symbol in enumerate(
        NIFTY_50,
        start=1,
    ):

        stock_result_dir = (
            RESULTS_DIR / symbol
        )

        metrics_file = (
            stock_result_dir / "metrics.json"
        )

        predictions_file = (
            stock_result_dir / "predictions.csv"
        )

        # ----------------------------------------------------
        # RESUME COMPLETED STOCK
        # ----------------------------------------------------

        if (
            metrics_file.exists()
            and predictions_file.exists()
        ):

            print(
                f"[{stock_number}/50] "
                f"{symbol} already completed. "
                f"Loading existing results."
            )

            try:

                predictions = pd.read_csv(
                    predictions_file
                )

                predictions["date"] = pd.to_datetime(
                    predictions["date"]
                )

                all_predictions.append(
                    predictions
                )

                with open(
                    metrics_file,
                    "r",
                ) as f:

                    saved_metrics = json.load(f)

                stock_metrics.append(
                    {
                        "symbol": symbol,
                        **saved_metrics["test"],
                    }
                )

                timing_results[
                    symbol
                ] = saved_metrics["timing"]

                print(
                    f"    Loaded "
                    f"{len(predictions)} predictions."
                )

            except Exception as e:

                print(
                    f"    WARNING: Could not load "
                    f"existing results for {symbol}: "
                    f"{e}"
                )

                failed_stocks.append(
                    {
                        "symbol": symbol,
                        "error":
                            f"Could not load existing "
                            f"results: {e}",
                    }
                )

            continue

        # ----------------------------------------------------
        # PROCESS INCOMPLETE STOCK
        # ----------------------------------------------------

        print("\n\n")
        print("=" * 80)
        print(
            f"STOCK {stock_number}/"
            f"{len(NIFTY_50)}: "
            f"{symbol}"
        )
        print("=" * 80)

        try:

            predictions, metrics, timing = (
                process_stock(
                    symbol,
                    tokenizer,
                    model_template,
                )
            )

            all_predictions.append(
                predictions
            )

            stock_metrics.append(
                {
                    "symbol":
                        symbol,
                    **metrics,
                }
            )

            timing_results[
                symbol
            ] = timing

            print(
                f"\n{symbol} completed successfully."
            )

        except Exception as e:

            print("\n")
            print(
                f"ERROR processing "
                f"{symbol}: {e}"
            )

            traceback.print_exc()

            failed_stocks.append(
                {
                    "symbol": symbol,
                    "error": str(e),
                    "traceback":
                        traceback.format_exc(),
                }
            )

        finally:

            gc.collect()

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

    # --------------------------------------------------------
    # COMBINE PREDICTIONS
    # --------------------------------------------------------

    if all_predictions:

        combined_predictions = pd.concat(
            all_predictions,
            ignore_index=True,
        )

        combined_predictions[
            "date"
        ] = pd.to_datetime(
            combined_predictions["date"]
        )

        combined_predictions = (
            combined_predictions.sort_values(
                [
                    "date",
                    "symbol",
                ]
            )
        )

        combined_predictions.to_csv(
            MASTER_PREDICTIONS_PATH,
            index=False,
        )

        print(
            f"\nSaved all predictions:"
            f"\n{MASTER_PREDICTIONS_PATH}"
        )

    else:

        raise RuntimeError(
            "No stocks completed successfully."
        )

    # --------------------------------------------------------
    # STOCK METRICS
    # --------------------------------------------------------

    metrics_df = pd.DataFrame(
        stock_metrics
    )

    metrics_df.to_csv(
        STOCK_METRICS_PATH,
        index=False,
    )

    print(
        f"Saved stock metrics:"
        f"\n{STOCK_METRICS_PATH}"
    )

    # --------------------------------------------------------
    # RANKIC
    # --------------------------------------------------------

    daily_rankic = (
        calculate_daily_rankic(
            combined_predictions
        )
    )

    daily_rankic.to_csv(
        DAILY_RANKIC_PATH,
        index=False,
    )

    rankic_metrics = (
        calculate_rankic_metrics(
            daily_rankic
        )
    )

    with open(
        CROSS_SECTIONAL_METRICS_PATH,
        "w",
    ) as f:

        json.dump(
            rankic_metrics,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # TIMING
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    timing_output = {

        "total_seconds":
            total_time,

        "total_time":
            format_time(total_time),

        "stocks_completed":
            len(stock_metrics),

        "stocks_failed":
            len(failed_stocks),

        "per_stock":
            timing_results,
    }

    with open(
        TIMING_PATH,
        "w",
    ) as f:

        json.dump(
            timing_output,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # FAILED STOCKS
    # --------------------------------------------------------

    with open(
        FAILED_STOCKS_PATH,
        "w",
    ) as f:

        json.dump(
            failed_stocks,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n\n")
    print("=" * 80)
    print("NIFTY 50 EXPERIMENT COMPLETE")
    print("=" * 80)

    print(
        f"\nStocks completed: "
        f"{len(stock_metrics)}/"
        f"{len(NIFTY_50)}"
    )

    print(
        f"Stocks failed: "
        f"{len(failed_stocks)}"
    )

    print(
        f"\nTotal runtime: "
        f"{format_time(total_time)}"
    )

    print("\n")
    print("=" * 80)
    print("CROSS-SECTIONAL RANKIC")
    print("=" * 80)

    for name, value in (
        rankic_metrics.items()
    ):

        print(
            f"{name:<30}: "
            f"{value}"
        )

    print("\n")
    print("=" * 80)
    print("OUTPUT FILES")
    print("=" * 80)

    print(
        f"\nAll predictions:"
        f"\n{MASTER_PREDICTIONS_PATH}"
    )

    print(
        f"\nStock metrics:"
        f"\n{STOCK_METRICS_PATH}"
    )

    print(
        f"\nDaily RankIC:"
        f"\n{DAILY_RANKIC_PATH}"
    )

    print(
        f"\nRankIC metrics:"
        f"\n{CROSS_SECTIONAL_METRICS_PATH}"
    )

    print(
        f"\nTiming:"
        f"\n{TIMING_PATH}"
    )

    print(
        f"\nFailed stocks:"
        f"\n{FAILED_STOCKS_PATH}"
    )


if __name__ == "__main__":

    main()