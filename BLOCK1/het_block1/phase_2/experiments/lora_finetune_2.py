import copy
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from plotly.subplots import make_subplots


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]

KRONOS_DIR = Path("/home/soq/Kronos")

DATA_PATH = PROJECT_DIR / "data" / "nifty_ohlcv.parquet"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_PATH = RESULTS_DIR / "lora_predictions.csv"
METRICS_PATH = RESULTS_DIR / "lora_metrics.json"
PLOT_PATH = RESULTS_DIR / "lora_forecast.html"
OPTUNA_PATH = RESULTS_DIR / "lora_optuna_trials.csv"


# ============================================================
# CONFIG
# ============================================================

START_DATE = "2022-01-01"

# ------------------------------------------------------------
# DATA SPLITS
# ------------------------------------------------------------

OPTUNA_START_DATE = "2022-01-01"
OPTUNA_END_DATE = "2022-12-31"

VALIDATION_START_DATE = "2023-01-01"
VALIDATION_END_DATE = "2023-06-30"

ACTUAL_START_DATE = "2023-07-01"

# ------------------------------------------------------------
# ROLLING TRAINING
# ------------------------------------------------------------

TRAIN_DAYS = 32
VALIDATION_DAYS = 8
CONTEXT_DAYS = 40

MAX_EPOCHS = 200
PATIENCE = 5
MIN_DELTA = 1e-4

N_OPTUNA_TRIALS = 10

SEED = 42

MODEL_NAME = "NeoQuasar/Kronos-base"
TOKENIZER_NAME = "NeoQuasar/Kronos-Tokenizer-base"

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
    minutes = int((seconds % 3600) // 60)
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
# LOAD DATA
# ============================================================

def load_data():

    print("Loading data...")

    data = pd.read_parquet(DATA_PATH)

    data.index = pd.to_datetime(data.index)
    data = data.sort_index()

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
            f"Missing columns: {missing}"
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

    print(
        f"Loaded {len(data)} trading days"
    )

    print(
        f"Range: "
        f"{data.index[0].date()} → "
        f"{data.index[-1].date()}"
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

        self.scaling = alpha / rank

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

    # Freeze everything.
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
            torch.from_numpy(normalized)
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

    stream_one = stream_one.to(DEVICE)
    stream_two = stream_two.to(DEVICE)
    stamps = stamps.to(DEVICE)

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

    return loss_one + loss_two


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

    # --------------------------------------------------------
    # FRESH MODEL
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TOKENIZE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

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

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # CLEAN EPOCH PRINTING
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if validation_value < (
            best_loss - MIN_DELTA
        ):

            best_loss = validation_value

            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print(
                f"      Early stopping at "
                f"epoch {epoch + 1}"
            )

            break

    # --------------------------------------------------------
    # RESTORE BEST WEIGHTS
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

    return model, best_loss


# ============================================================
# OPTUNA
# ============================================================

def run_optuna(
    data,
    model_template,
    tokenizer,
):

    optuna_start_time = time.perf_counter()

    print("\n")
    print("=" * 70)
    print("OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 70)

    print(
        f"Optuna data: "
        f"{OPTUNA_START_DATE} → "
        f"{OPTUNA_END_DATE}"
    )

    # --------------------------------------------------------
    # ONLY 2022 DATA FOR OPTUNA
    # --------------------------------------------------------

    optuna_start = pd.Timestamp(
        OPTUNA_START_DATE
    )

    optuna_end = pd.Timestamp(
        OPTUNA_END_DATE
    )

    tuning_data = data[
        (data.index >= optuna_start)
        & (data.index <= optuna_end)
    ].copy()

    if tuning_data.empty:

        raise ValueError(
            "No data found for Optuna period."
        )

    # Need 41 days for each window.
    minimum_window = (
        CONTEXT_DAYS + 1
    )

    if len(tuning_data) < minimum_window:

        raise ValueError(
            "Not enough data for Optuna."
        )

    # --------------------------------------------------------
    # SELECT HISTORICAL WINDOWS
    # --------------------------------------------------------

    max_windows = 10

    possible = (
        len(tuning_data)
        - minimum_window
        + 1
    )

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

    # --------------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------------

    def objective(trial):

        trial_start_time = time.perf_counter()

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
            f"\nTrial {trial.number + 1}/"
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
            window
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

            if torch.cuda.is_available():

                torch.cuda.empty_cache()

        mean_loss = float(
            np.mean(losses)
        )

        trial_time = (
            time.perf_counter()
            - trial_start_time
        )

        print(
            f"\nTrial {trial.number + 1} "
            f"mean validation loss = "
            f"{mean_loss:.6f}"
        )

        print(
            f"Trial time: "
            f"{format_time(trial_time)}"
        )

        return mean_loss

    # --------------------------------------------------------
    # CREATE STUDY
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SAVE OPTUNA RESULTS
    # --------------------------------------------------------

    study.trials_dataframe().to_csv(
        OPTUNA_PATH,
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
# ROLLING LORA FORECAST
# ============================================================

def walk_forward(
    data,
    model_template,
    tokenizer,
    best_params,
):

    predictions = []

    actual_start = pd.Timestamp(
        ACTUAL_START_DATE
    )

    # --------------------------------------------------------
    # FIND FIRST PREDICTION INDEX
    # --------------------------------------------------------

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
    print("ROLLING LORA FORECAST")
    print("=" * 70)

    print(
        f"Actual training / prediction starts: "
        f"{ACTUAL_START_DATE}"
    )

    print(
        f"Predictions: "
        f"{total_predictions}"
    )

    total_training_time = 0.0
    total_prediction_time = 0.0

    # --------------------------------------------------------
    # ROLLING LOOP
    # --------------------------------------------------------

    for (
        prediction_number,
        i
    ) in enumerate(
        prediction_indices,
        start=1,
    ):

        iteration_start_time = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # EXACTLY 41 ACTUAL DAYS
        # ----------------------------------------------------

        window = data.iloc[
            i - CONTEXT_DAYS:
            i + 1
        ].copy()

        target_day = window.iloc[
            CONTEXT_DAYS
        ]

        print(
            f"\n[{prediction_number}/"
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

        # ----------------------------------------------------
        # FRESH LORA
        # ----------------------------------------------------

        set_seed(
            SEED + prediction_number
        )

        training_start_time = (
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
            - training_start_time
        )

        total_training_time += (
            training_time
        )

        # ----------------------------------------------------
        # PREDICT DAY
        # ----------------------------------------------------

        prediction_start_time = (
            time.perf_counter()
        )

        predictor = KronosPredictor(
            model,
            tokenizer,
            device=DEVICE,
            max_context=512,
        )

        context = window.iloc[
            :CONTEXT_DAYS
        ].copy()

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

        predicted = prediction.iloc[0]

        prediction_time = (
            time.perf_counter()
            - prediction_start_time
        )

        total_prediction_time += (
            prediction_time
        )

        # ----------------------------------------------------
        # RETURN CALCULATION
        # ----------------------------------------------------

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

        absolute_return_error = abs(
            return_error
        )

        # ----------------------------------------------------
        # DAILY ERROR PRINT
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        predictions.append(
            {
                "date":
                    target_day.name,

                "actual_open":
                    target_day["open"],

                "actual_high":
                    target_day["high"],

                "actual_low":
                    target_day["low"],

                "actual_close":
                    actual_close,

                "predicted_open":
                    predicted["open"],

                "predicted_high":
                    predicted["high"],

                "predicted_low":
                    predicted["low"],

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
                    absolute_return_error,

                "validation_loss":
                    validation_loss,

                "training_time_seconds":
                    training_time,

                "prediction_time_seconds":
                    prediction_time,
            }
        )

        del predictor
        del model

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

        iteration_time = (
            time.perf_counter()
            - iteration_start_time
        )

        print(
            f"    Total day time: "
            f"{format_time(iteration_time)}"
        )

    results = pd.DataFrame(
        predictions
    )

    results["date"] = pd.to_datetime(
        results["date"]
    )

    results.set_index(
        "date",
        inplace=True,
    )

    timing = {
        "training_seconds":
            total_training_time,

        "prediction_seconds":
            total_prediction_time,

        "total_rolling_seconds":
            total_training_time
            + total_prediction_time,
    }

    return results, timing


# ============================================================
# VALIDATION DATASET
# ============================================================

def validate_period(
    data,
    model_template,
    tokenizer,
    best_params,
):

    print("\n")
    print("=" * 70)
    print("VALIDATION PERIOD")
    print("=" * 70)

    print(
        f"Validation: "
        f"{VALIDATION_START_DATE} → "
        f"{VALIDATION_END_DATE}"
    )

    validation_start = pd.Timestamp(
        VALIDATION_START_DATE
    )

    validation_end = pd.Timestamp(
        VALIDATION_END_DATE
    )

    validation_data = data[
        (data.index >= validation_start)
        & (data.index <= validation_end)
    ].copy()

    if validation_data.empty:

        raise ValueError(
            "No validation data found."
        )

    print(
        f"Validation days: "
        f"{len(validation_data)}"
    )

    # --------------------------------------------------------
    # IMPORTANT
    # --------------------------------------------------------
    #
    # This period is kept separate from the final
    # July+ prediction results.
    #
    # The best Optuna parameters are evaluated here
    # before actual July+ rolling training begins.
    #
    # We use rolling 40-day windows so the model sees
    # realistic historical context.
    # --------------------------------------------------------

    validation_results = []

    validation_training_time = 0.0
    validation_prediction_time = 0.0

    # Need historical data before validation starts
    # for the 40-day context.

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

        validation_training_time += (
            training_time
        )

        prediction_start = (
            time.perf_counter()
        )

        predictor = KronosPredictor(
            model,
            tokenizer,
            device=DEVICE,
            max_context=512,
        )

        context = window.iloc[
            :CONTEXT_DAYS
        ].copy()

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

        predicted = prediction.iloc[0]

        prediction_time = (
            time.perf_counter()
            - prediction_start
        )

        validation_prediction_time += (
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
                "date":
                    target_day.name,

                "actual_close":
                    actual_close,

                "predicted_close":
                    predicted_close,

                "actual_return":
                    actual_return,

                "predicted_return":
                    predicted_return,

                "return_error":
                    return_error,

                "absolute_return_error":
                    abs(return_error),

                "validation_loss":
                    validation_loss,
            }
        )

        print(
            f"  {target_day.name.date()} "
            f"| actual_ret="
            f"{actual_return * 100:+.3f}% "
            f"| pred_ret="
            f"{predicted_return * 100:+.3f}% "
            f"| error="
            f"{return_error * 100:+.3f}%"
        )

        del predictor
        del model

        gc.collect()

        if torch.cuda.is_available():

            torch.cuda.empty_cache()

    validation_results = pd.DataFrame(
        validation_results
    )

    if validation_results.empty:

        raise ValueError(
            "No validation predictions generated."
        )

    validation_results["date"] = (
        pd.to_datetime(
            validation_results["date"]
        )
    )

    validation_results.set_index(
        "date",
        inplace=True,
    )

    timing = {
        "training_seconds":
            validation_training_time,

        "prediction_seconds":
            validation_prediction_time,

        "total_seconds":
            validation_training_time
            + validation_prediction_time,
    }

    return validation_results, timing


# ============================================================
# RETURN METRICS
# ============================================================

def calculate_metrics(results):

    actual_ret = results[
        "actual_return"
    ].to_numpy(float)

    predicted_ret = results[
        "predicted_return"
    ].to_numpy(float)

    error = (
        predicted_ret
        - actual_ret
    )

    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = float(
        np.sqrt(
            np.mean(
                error ** 2
            )
        )
    )

    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = float(
        np.mean(
            np.abs(error)
        )
    )

    # --------------------------------------------------------
    # DIRECTION ACCURACY
    # --------------------------------------------------------

    direction_match = (
        np.sign(actual_ret)
        == np.sign(predicted_ret)
    )

    direction_accuracy = float(
        np.mean(
            direction_match
        ) * 100
    )

    # --------------------------------------------------------
    # PEARSON CORRELATION
    # --------------------------------------------------------

    pearson = float(
        np.corrcoef(
            actual_ret,
            predicted_ret,
        )[0, 1]
        if len(actual_ret) > 1
        and np.std(actual_ret) > 0
        and np.std(predicted_ret) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # MEAN ERROR
    # --------------------------------------------------------

    mean_error = float(
        np.mean(error)
    )

    # --------------------------------------------------------
    # MAX ABS ERROR
    # --------------------------------------------------------

    max_absolute_error = float(
        np.max(
            np.abs(error)
        )
    )

    return {

        "RMSE (returns)":
            rmse,

        "MAE (returns)":
            mae,

        "Direction Accuracy (%)":
            direction_accuracy,

        "Pearson (returns)":
            pearson,

        "Mean Error (returns)":
            mean_error,

        "Max Absolute Error (returns)":
            max_absolute_error,

    }


# ============================================================
# PLOTLY
# ============================================================

def create_plot(
    results,
    metrics,
):

    print(
        "\nCreating Plotly graph..."
    )

    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[
            0.40,
            0.40,
            0.20,
        ],
        subplot_titles=(
            "Actual Daily OHLC",
            "Kronos + LoRA Predicted Daily OHLC",
            "Return Error (Predicted - Actual)",
        ),
    )

    # --------------------------------------------------------
    # ACTUAL
    # --------------------------------------------------------

    figure.add_trace(
        go.Candlestick(
            x=results.index,
            open=results["actual_open"],
            high=results["actual_high"],
            low=results["actual_low"],
            close=results["actual_close"],
            name="Actual",
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # PREDICTED
    # --------------------------------------------------------

    figure.add_trace(
        go.Candlestick(
            x=results.index,
            open=results["predicted_open"],
            high=results["predicted_high"],
            low=results["predicted_low"],
            close=results["predicted_close"],
            name="Predicted",
        ),
        row=2,
        col=1,
    )

    # --------------------------------------------------------
    # RETURN ERROR
    # --------------------------------------------------------

    figure.add_trace(
        go.Scatter(
            x=results.index,
            y=results["return_error"] * 100,
            mode="lines",
            name="Return Error",
        ),
        row=3,
        col=1,
    )

    figure.add_hline(
        y=0,
        row=3,
        col=1,
        line_dash="dash",
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    metrics_text = (
        f"<b>LoRA Return Metrics</b>&nbsp;&nbsp;&nbsp;"
        f"RMSE: {metrics['RMSE (returns)']:.6f}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"MAE: {metrics['MAE (returns)']:.6f}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Dir Acc: "
        f"{metrics['Direction Accuracy (%)']:.2f}%"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Pearson: "
        f"{metrics['Pearson (returns)']:.4f}"
    )

    figure.add_annotation(
        text=metrics_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.045,
        xanchor="center",
        yanchor="bottom",
        showarrow=False,
        align="center",
        bgcolor="white",
        bordercolor="gray",
        borderwidth=1,
        font={"size": 12},
    )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    figure.update_layout(
        title=(
            "Kronos + LoRA: "
            "Rolling Daily OHLC Forecast "
            "from July 2023"
        ),
        height=1100,
        template="plotly_white",
        hovermode="x unified",

        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,

        margin=dict(
            t=150,
            b=70,
            l=80,
            r=40,
        ),
    )

    figure.update_yaxes(
        title_text="Actual Price",
        row=1,
        col=1,
    )

    figure.update_yaxes(
        title_text="Predicted Price",
        row=2,
        col=1,
    )

    figure.update_yaxes(
        title_text="Return Error (%)",
        row=3,
        col=1,
    )

    figure.update_xaxes(
        title_text="Trading Day",
        row=3,
        col=1,
    )

    figure.write_html(
        str(PLOT_PATH),
        include_plotlyjs="cdn",
    )

    print(
        f"Plotly graph: {PLOT_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    total_start_time = time.perf_counter()

    set_seed()

    print("=" * 70)
    print("KRONOS + LORA ROLLING FORECAST")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    data = load_data()

    # --------------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        KronosTokenizer.from_pretrained(
            TOKENIZER_NAME
        )
    )

    tokenizer = tokenizer.to(DEVICE)

    for parameter in (
        tokenizer.parameters()
    ):

        parameter.requires_grad = False

    print(
        f"Tokenizer device: "
        f"{next(tokenizer.parameters()).device}"
    )

    # --------------------------------------------------------
    # LOAD KRONOS
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
    # CUDA SANITY CHECK
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
    # OPTUNA
    # --------------------------------------------------------

    best_params, optuna_time = run_optuna(
        data,
        model_template,
        tokenizer,
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    validation_results, validation_timing = (
        validate_period(
            data,
            model_template,
            tokenizer,
            best_params,
        )
    )

    validation_metrics = calculate_metrics(
        validation_results
    )

    print("\n")
    print("=" * 70)
    print("VALIDATION METRICS")
    print("=" * 70)

    for name, value in validation_metrics.items():

        if "%" in name:

            print(
                f"{name:<30}: "
                f"{value:.2f}%"
            )

        else:

            print(
                f"{name:<30}: "
                f"{value:.6f}"
            )

    # --------------------------------------------------------
    # ACTUAL JULY+ ROLLING TRAINING
    # --------------------------------------------------------

    results, rolling_timing = walk_forward(
        data,
        model_template,
        tokenizer,
        best_params,
    )

    # --------------------------------------------------------
    # FINAL METRICS
    # --------------------------------------------------------

    metrics = calculate_metrics(
        results
    )

    print("\n")
    print("=" * 70)
    print(
        f"FINAL METRICS "
        f"(from {ACTUAL_START_DATE})"
    )
    print("=" * 70)

    for name, value in metrics.items():

        if "%" in name:

            print(
                f"{name:<30}: "
                f"{value:.2f}%"
            )

        else:

            print(
                f"{name:<30}: "
                f"{value:.6f}"
            )

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    results.to_csv(
        PREDICTIONS_PATH
    )

    # --------------------------------------------------------
    # TOTAL TIMING
    # --------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start_time
    )

    timing = {

        "optuna_seconds":
            optuna_time,

        "optuna_time":
            format_time(optuna_time),

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

        "validation_total_time":
            format_time(
                validation_timing[
                    "total_seconds"
                ]
            ),

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
                "total_rolling_seconds"
            ],

        "rolling_total_time":
            format_time(
                rolling_timing[
                    "total_rolling_seconds"
                ]
            ),

        "total_seconds":
            total_time,

        "total_time":
            format_time(total_time),
    }

    # --------------------------------------------------------
    # SAVE METRICS + TIMING
    # --------------------------------------------------------

    with open(
        METRICS_PATH,
        "w",
    ) as f:

        json.dump(
            {
                "test_period":
                    metrics,

                "validation_period":
                    validation_metrics,

                "timing":
                    timing,

                "splits":
                    {
                        "optuna":
                            f"{OPTUNA_START_DATE} → "
                            f"{OPTUNA_END_DATE}",

                        "validation":
                            f"{VALIDATION_START_DATE} → "
                            f"{VALIDATION_END_DATE}",

                        "actual_training":
                            f"{ACTUAL_START_DATE} → "
                            "end of dataset",
                    },

                "best_params":
                    best_params,

                "num_predictions":
                    len(results),

            },
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # PLOT
    # --------------------------------------------------------

    create_plot(
        results,
        metrics,
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)

    print(
        f"Optuna:              "
        f"{format_time(optuna_time)}"
    )

    print(
        f"Validation:          "
        f"{format_time(validation_timing['total_seconds'])}"
    )

    print(
        f"Actual training:     "
        f"{format_time(rolling_timing['training_seconds'])}"
    )

    print(
        f"Actual prediction:   "
        f"{format_time(rolling_timing['prediction_seconds'])}"
    )

    print(
        f"Actual rolling:      "
        f"{format_time(rolling_timing['total_rolling_seconds'])}"
    )

    print(
        f"TOTAL RUNTIME:       "
        f"{format_time(total_time)}"
    )

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(
        f"\nPredictions: {PREDICTIONS_PATH}"
    )

    print(
        f"Metrics:     {METRICS_PATH}"
    )

    print(
        f"Optuna:      {OPTUNA_PATH}"
    )

    print(
        f"Plotly:      {PLOT_PATH}"
    )


if __name__ == "__main__":

    main()