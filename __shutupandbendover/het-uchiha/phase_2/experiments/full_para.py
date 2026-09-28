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

PREDICTIONS_PATH = RESULTS_DIR / "full_finetune_predictions.csv"
METRICS_PATH = RESULTS_DIR / "full_finetune_metrics.json"
PLOT_PATH = RESULTS_DIR / "full_finetune_forecast.html"
OPTUNA_PATH = RESULTS_DIR / "full_finetune_optuna_trials.csv"
TIMING_PATH = RESULTS_DIR / "full_finetune_timing.json"


# ============================================================
# CONFIG
# ============================================================

START_DATE = "2022-01-01"

# Optuna only sees calendar year 2022.
OPTUNA_START_DATE = "2022-01-01"
OPTUNA_END_DATE = "2022-12-31"

# Separate validation period before the clean test period.
VALIDATION_START_DATE = "2023-01-01"
VALIDATION_END_DATE = "2023-06-30"

# Actual walk-forward test starts here.
TEST_START_DATE = "2023-07-01"

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


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available. "
        "This script is configured to train on CUDA."
    )

DEVICE = "cuda"


# ============================================================
# KRONOS IMPORT
# ============================================================

sys.path.insert(0, str(KRONOS_DIR))

from model import Kronos, KronosPredictor, KronosTokenizer
from model.kronos import calc_time_stamps


# ============================================================
# TIMING
# ============================================================

TIMINGS = {
    "script_start": None,
    "data_loading_seconds": None,
    "model_loading_seconds": None,
    "optuna_seconds": None,
    "walk_forward_seconds": None,
    "total_seconds": None,
    "optuna_trials": [],
    "walk_forward_windows": [],
}


def save_timings():
    with open(TIMING_PATH, "w") as f:
        json.dump(TIMINGS, f, indent=4)


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

    print(f"Loaded {len(data)} trading days")

    print(
        f"Range: "
        f"{data.index[0].date()} → "
        f"{data.index[-1].date()}"
    )

    return data


# ============================================================
# FULL-PARAMETER TRAINING
# ============================================================

def make_trainable_model(model_template):

    # Fresh copy of the original pretrained Kronos.
    model = copy.deepcopy(model_template)

    # IMPORTANT:
    # Every Kronos parameter is trainable.
    for parameter in model.parameters():
        parameter.requires_grad = True

    model.to(DEVICE)

    return model


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

    timestamps = pd.Series(frame.index)

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

    window_start_time = time.perf_counter()

    train_data = window.iloc[:TRAIN_DAYS]

    validation_data = window.iloc[
        TRAIN_DAYS:
        TRAIN_DAYS + VALIDATION_DAYS
    ]

    # --------------------------------------------------------
    # FRESH FULL KRONOS MODEL
    # --------------------------------------------------------

    model = make_trainable_model(
        model_template
    )

    # ALL parameters are optimized.
    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    for epoch in range(MAX_EPOCHS):

        epoch_start = time.perf_counter()

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
            model.parameters(),
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

        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )

        # ----------------------------------------------------
        # CLEAN PRINTING
        # ----------------------------------------------------

        if (
            epoch == 0
            or (epoch + 1) % 5 == 0
        ):

            print(
                f"      Epoch "
                f"{epoch + 1:03d}/{MAX_EPOCHS}"
                f" | train={train_loss.item():.5f}"
                f" | val={validation_value:.5f}"
                f" | {epoch_seconds:.2f}s"
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

    window_seconds = (
        time.perf_counter()
        - window_start_time
    )

    return (
        model,
        best_loss,
        window_seconds,
    )


# ============================================================
# OPTUNA
# ============================================================

def run_optuna(
    data,
    model_template,
    tokenizer,
):

    optuna_start = time.perf_counter()

    print("\n")
    print("=" * 70)
    print("OPTUNA HYPERPARAMETER SEARCH")
    print("=" * 70)

    optuna_start_date = pd.Timestamp(
        OPTUNA_START_DATE
    )

    optuna_end_date = (
        pd.Timestamp(OPTUNA_END_DATE)
        + pd.Timedelta(days=1)
    )

    tuning_data = data[
        (data.index >= optuna_start_date)
        & (data.index < optuna_end_date)
    ].copy()

    if tuning_data.empty:
        raise ValueError(
            "No data found for Optuna."
        )

    print(
        f"Optuna data: "
        f"{tuning_data.index[0].date()} → "
        f"{tuning_data.index[-1].date()}"
    )

    minimum_window = CONTEXT_DAYS + 1

    if len(tuning_data) < minimum_window:
        raise ValueError(
            "Not enough data for Optuna."
        )

    # Select historical windows from 2022.
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

    def objective(trial):

        trial_start = time.perf_counter()

        params = {
            "learning_rate":
                trial.suggest_float(
                    "learning_rate",
                    1e-6,
                    1e-4,
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
            f"  learning_rate="
            f"{params['learning_rate']:.3e} | "
            f"weight_decay="
            f"{params['weight_decay']:.3e}"
        )

        losses = []

        for (
            window_number,
            window
        ) in enumerate(tuning_windows):

            print(
                f"    Window "
                f"{window_number + 1}/"
                f"{len(tuning_windows)}"
            )

            set_seed(
                SEED
                + trial.number
                + window_number
            )

            (
                model,
                validation_loss,
                window_seconds,
            ) = train_one_window(
                window,
                model_template,
                tokenizer,
                params,
            )

            losses.append(
                validation_loss
            )

            del model

            gc.collect()

            torch.cuda.empty_cache()

        mean_loss = float(
            np.mean(losses)
        )

        trial_seconds = (
            time.perf_counter()
            - trial_start
        )

        TIMINGS["optuna_trials"].append(
            {
                "trial": trial.number,
                "seconds": trial_seconds,
                "mean_validation_loss": mean_loss,
                "params": params,
            }
        )

        print(
            f"  Trial {trial.number + 1} finished "
            f"in {trial_seconds / 60:.2f} min | "
            f"mean val loss={mean_loss:.6f}"
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

    study.trials_dataframe().to_csv(
        OPTUNA_PATH,
        index=False,
    )

    optuna_seconds = (
        time.perf_counter()
        - optuna_start
    )

    TIMINGS["optuna_seconds"] = optuna_seconds

    save_timings()

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
        f"Optuna time: "
        f"{optuna_seconds / 3600:.2f} hours"
    )

    return study.best_trial.params


# ============================================================
# PRE-TEST VALIDATION
# ============================================================

def run_validation_period(
    data,
    model_template,
    tokenizer,
    best_params,
):

    print("\n")
    print("=" * 70)
    print("2023 H1 VALIDATION PERIOD")
    print("=" * 70)

    validation_data = data[
        (data.index >= pd.Timestamp(VALIDATION_START_DATE))
        & (data.index <= pd.Timestamp(VALIDATION_END_DATE))
    ].copy()

    if validation_data.empty:
        raise ValueError(
            "No data found for the 2023 validation period."
        )

    print(
        f"Validation data: "
        f"{validation_data.index[0].date()} → "
        f"{validation_data.index[-1].date()}"
    )

    # This period is kept separate from the final test.
    # We use rolling 40-day windows here, exactly like the
    # walk-forward setup, but do NOT include these predictions
    # in the final test metrics.
    #
    # Only windows whose 41st target day lies inside the
    # requested validation period are evaluated.

    available = data[
        data.index <= pd.Timestamp(VALIDATION_END_DATE)
    ].copy()

    validation_predictions = []

    target_indices = [
        i
        for i in range(CONTEXT_DAYS, len(available))
        if (
            available.index[i]
            >= pd.Timestamp(VALIDATION_START_DATE)
            and available.index[i]
            <= pd.Timestamp(VALIDATION_END_DATE)
        )
    ]

    print(
        f"Validation predictions: "
        f"{len(target_indices)}"
    )

    for number, i in enumerate(
        target_indices,
        start=1,
    ):

        window = available.iloc[
            i - CONTEXT_DAYS:
            i + 1
        ].copy()

        target_day = window.iloc[
            CONTEXT_DAYS
        ]

        print(
            f"\nValidation [{number}/"
            f"{len(target_indices)}] "
            f"{target_day.name.date()}"
        )

        set_seed(
            SEED + number
        )

        (
            model,
            validation_loss,
            window_seconds,
        ) = train_one_window(
            window,
            model_template,
            tokenizer,
            best_params,
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

        prediction = predictor.predict(
            context,
            pd.Series(context.index),
            pd.Series([target_day.name]),
            pred_len=1,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )

        predicted = prediction.iloc[0]

        actual_close = float(
            target_day["close"]
        )

        predicted_close = float(
            predicted["close"]
        )

        previous_close = float(
            window.iloc[CONTEXT_DAYS - 1]["close"]
        )

        actual_return = (
            actual_close - previous_close
        ) / max(abs(previous_close), 1e-8)

        predicted_return = (
            predicted_close - previous_close
        ) / max(abs(previous_close), 1e-8)

        return_error = (
            actual_return - predicted_return
        )

        direction_correct = (
            np.sign(actual_return)
            == np.sign(predicted_return)
        )

        print(
            f"  Actual={actual_close:.2f} | "
            f"Predicted={predicted_close:.2f} | "
            f"Return error={return_error * 100:.4f}% | "
            f"Direction={'✓' if direction_correct else '✗'}"
        )

        validation_predictions.append(
            {
                "date": target_day.name,
                "actual_close": actual_close,
                "predicted_close": predicted_close,
                "actual_return": actual_return,
                "predicted_return": predicted_return,
                "return_error": return_error,
                "direction_correct": bool(direction_correct),
                "training_validation_loss": validation_loss,
                "training_seconds": window_seconds,
            }
        )

        del predictor
        del model

        gc.collect()
        torch.cuda.empty_cache()

    return pd.DataFrame(
        validation_predictions
    )


# ============================================================
# WALK FORWARD TEST
# ============================================================

def walk_forward(
    data,
    model_template,
    tokenizer,
    best_params,
):

    walk_start = time.perf_counter()

    predictions = []

    test_start = pd.Timestamp(
        TEST_START_DATE
    )

    # Need 40 actual context days before the first target.
    candidate_indices = [
        i
        for i in range(CONTEXT_DAYS, len(data))
        if data.index[i] >= test_start
    ]

    total_predictions = len(candidate_indices)

    print("\n")
    print("=" * 70)
    print("ROLLING FULL-PARAMETER FORECAST")
    print("=" * 70)

    print(
        f"Test starts: {TEST_START_DATE}"
    )

    print(
        f"Predictions: {total_predictions}"
    )

    for prediction_number, i in enumerate(
        candidate_indices,
        start=1,
    ):

        window = data.iloc[
            i - CONTEXT_DAYS:
            i + 1
        ].copy()

        target_day = window.iloc[
            CONTEXT_DAYS
        ]

        print("\n")
        print("-" * 70)
        print(
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

        window_start = time.perf_counter()

        # ----------------------------------------------------
        # FRESH PRETRAINED KRONOS
        # ----------------------------------------------------

        set_seed(
            SEED + prediction_number
        )

        (
            model,
            validation_loss,
            training_seconds,
        ) = train_one_window(
            window,
            model_template,
            tokenizer,
            best_params,
        )

        # ----------------------------------------------------
        # PREDICT DAY 41
        # ----------------------------------------------------

        prediction_start = time.perf_counter()

        predictor = KronosPredictor(
            model,
            tokenizer,
            device=DEVICE,
            max_context=512,
        )

        context = window.iloc[
            :CONTEXT_DAYS
        ].copy()

        target_timestamp = pd.Series(
            [target_day.name]
        )

        prediction = predictor.predict(
            context,
            pd.Series(context.index),
            target_timestamp,
            pred_len=1,
            T=1.0,
            top_k=0,
            top_p=0.9,
            sample_count=1,
            verbose=False,
        )

        predicted = prediction.iloc[0]

        prediction_seconds = (
            time.perf_counter()
            - prediction_start
        )

        # ----------------------------------------------------
        # DAILY RETURN ERROR
        # ----------------------------------------------------

        actual_close = float(
            target_day["close"]
        )

        predicted_close = float(
            predicted["close"]
        )

        previous_close = float(
            window.iloc[
                CONTEXT_DAYS - 1
            ]["close"]
        )

        actual_return = (
            actual_close - previous_close
        ) / max(
            abs(previous_close),
            1e-8,
        )

        predicted_return = (
            predicted_close - previous_close
        ) / max(
            abs(previous_close),
            1e-8,
        )

        return_error = (
            actual_return
            - predicted_return
        )

        direction_correct = (
            np.sign(actual_return)
            == np.sign(predicted_return)
        )

        absolute_return_error = abs(
            return_error
        )

        total_window_seconds = (
            time.perf_counter()
            - window_start
        )

        # ----------------------------------------------------
        # PRINT DAILY RESULT
        # ----------------------------------------------------

        print(
            f"    Actual close:    "
            f"{actual_close:,.2f}"
        )

        print(
            f"    Predicted close: "
            f"{predicted_close:,.2f}"
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
            f"    Abs return error:"
            f" {absolute_return_error * 100:.4f}%"
        )

        print(
            f"    Direction:       "
            f"{'CORRECT ✓' if direction_correct else 'WRONG ✗'}"
        )

        print(
            f"    Training time:   "
            f"{training_seconds / 60:.2f} min"
        )

        print(
            f"    Prediction time: "
            f"{prediction_seconds:.2f} sec"
        )

        print(
            f"    Total window:    "
            f"{total_window_seconds / 60:.2f} min"
        )

        predictions.append(
            {
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
                    previous_close,

                "actual_return":
                    actual_return,

                "predicted_return":
                    predicted_return,

                "return_error":
                    return_error,

                "absolute_return_error":
                    absolute_return_error,

                "direction_correct":
                    bool(direction_correct),

                "training_validation_loss":
                    validation_loss,

                "training_seconds":
                    training_seconds,

                "prediction_seconds":
                    prediction_seconds,

                "total_window_seconds":
                    total_window_seconds,
            }
        )

        TIMINGS["walk_forward_windows"].append(
            {
                "prediction_number":
                    prediction_number,
                "date":
                    str(target_day.name),
                "training_seconds":
                    training_seconds,
                "prediction_seconds":
                    prediction_seconds,
                "total_window_seconds":
                    total_window_seconds,
            }
        )

        save_timings()

        del predictor
        del model

        gc.collect()
        torch.cuda.empty_cache()

    walk_seconds = (
        time.perf_counter()
        - walk_start
    )

    TIMINGS["walk_forward_seconds"] = walk_seconds

    save_timings()

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

    return results


# ============================================================
# RETURN METRICS
# ============================================================

def calculate_metrics(results):

    actual_returns = results[
        "actual_return"
    ].to_numpy(float)

    predicted_returns = results[
        "predicted_return"
    ].to_numpy(float)

    return_errors = (
        actual_returns
        - predicted_returns
    )

    rmse_returns = float(
        np.sqrt(
            np.mean(
                return_errors ** 2
            )
        )
    )

    mae_returns = float(
        np.mean(
            np.abs(return_errors)
        )
    )

    direction_accuracy = float(
        np.mean(
            results[
                "direction_correct"
            ].to_numpy(bool)
        )
        * 100
    )

    if (
        len(actual_returns) > 1
        and np.std(actual_returns) > 0
        and np.std(predicted_returns) > 0
    ):

        pearson_returns = float(
            np.corrcoef(
                actual_returns,
                predicted_returns,
            )[0, 1]
        )

    else:

        pearson_returns = 0.0

    # Simple directional strategy:
    # long when predicted return > 0,
    # short when predicted return < 0.
    strategy_returns = (
        np.sign(predicted_returns)
        * actual_returns
    )

    strategy_growth = np.prod(
        1 + strategy_returns
    )

    strategy_return = float(
        (strategy_growth - 1) * 100
    )

    return {
        "RMSE (returns)": rmse_returns,
        "MAE (returns)": mae_returns,
        "Direction Accuracy (%)":
            direction_accuracy,
        "Pearson (returns)":
            pearson_returns,
        "Directional Strategy Return (%)":
            strategy_return,
        "Mean Actual Return (%)":
            float(
                np.mean(actual_returns)
                * 100
            ),
        "Mean Predicted Return (%)":
            float(
                np.mean(predicted_returns)
                * 100
            ),
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

    return_difference = (
        results["actual_return"]
        - results["predicted_return"]
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
            "Predicted Daily OHLC",
            "Daily Return Error (Actual - Predicted)",
        ),
    )

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

    figure.add_trace(
        go.Scatter(
            x=results.index,
            y=return_difference * 100,
            mode="lines",
            name="Return Error (%)",
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

    metrics_text = (
        f"<b>Return Metrics</b>&nbsp;&nbsp;&nbsp;"
        f"RMSE: {metrics['RMSE (returns)']:.6f}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"MAE: {metrics['MAE (returns)']:.6f}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Direction: "
        f"{metrics['Direction Accuracy (%)']:.2f}%"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Pearson: "
        f"{metrics['Pearson (returns)']:.4f}"
        f"&nbsp;&nbsp;|&nbsp;&nbsp;"
        f"Strategy: "
        f"{metrics['Directional Strategy Return (%)']:.2f}%"
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

    figure.update_layout(
        title=(
            "Kronos Full-Parameter Fine-Tuning: "
            "Daily Walk-Forward Forecast"
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

    TIMINGS["script_start"] = time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    total_start = time.perf_counter()

    set_seed()

    print("=" * 70)
    print("KRONOS FULL-PARAMETER FINE-TUNING")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    data_start = time.perf_counter()

    data = load_data()

    TIMINGS["data_loading_seconds"] = (
        time.perf_counter()
        - data_start
    )

    # --------------------------------------------------------
    # Load Kronos
    # --------------------------------------------------------

    model_start = time.perf_counter()

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        KronosTokenizer.from_pretrained(
            TOKENIZER_NAME
        )
    )

    tokenizer = tokenizer.to(DEVICE)

    for parameter in tokenizer.parameters():
        parameter.requires_grad = False

    print(
        f"Tokenizer device: "
        f"{next(tokenizer.parameters()).device}"
    )

    print(
        "Loading Kronos..."
    )

    model_template = (
        Kronos.from_pretrained(
            MODEL_NAME
        )
    )

    model_template = model_template.cpu()

    print(
        f"Kronos template device: "
        f"{next(model_template.parameters()).device}"
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model_template.parameters()
    )

    print(
        f"Total Kronos parameters: "
        f"{total_parameters:,}"
    )

    TIMINGS["model_loading_seconds"] = (
        time.perf_counter()
        - model_start
    )

    # --------------------------------------------------------
    # CUDA sanity check
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
        f"  Kronos template device: "
        f"{next(model_template.parameters()).device}"
    )

    print(
        f"  Tokenizer device: "
        f"{next(tokenizer.parameters()).device}"
    )

    # --------------------------------------------------------
    # Optuna on 2022 ONLY
    # --------------------------------------------------------

    best_params = run_optuna(
        data,
        model_template,
        tokenizer,
    )

    # --------------------------------------------------------
    # 2023 H1 validation
    # --------------------------------------------------------

    validation_results = run_validation_period(
        data,
        model_template,
        tokenizer,
        best_params,
    )

    if not validation_results.empty:

        validation_metrics = calculate_metrics(
            validation_results
        )

        print("\n")
        print("=" * 70)
        print("2023 H1 VALIDATION METRICS")
        print("=" * 70)

        for name, value in validation_metrics.items():

            if "%" in name:

                print(
                    f"{name:<35}: "
                    f"{value:.2f}%"
                )

            elif "Pearson" in name:

                print(
                    f"{name:<35}: "
                    f"{value:.4f}"
                )

            else:

                print(
                    f"{name:<35}: "
                    f"{value:.6f}"
                )

    else:

        validation_metrics = {}

    # --------------------------------------------------------
    # Actual test: July 2023 onwards
    # --------------------------------------------------------

    results = walk_forward(
        data,
        model_template,
        tokenizer,
        best_params,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    results.to_csv(
        PREDICTIONS_PATH
    )

    # --------------------------------------------------------
    # Final test metrics
    # --------------------------------------------------------

    metrics = calculate_metrics(
        results
    )

    TIMINGS["total_seconds"] = (
        time.perf_counter()
        - total_start
    )

    save_timings()

    with open(
        METRICS_PATH,
        "w",
    ) as f:

        json.dump(
            {
                "test_period": metrics,
                "validation_period": validation_metrics,
                "test_start": TEST_START_DATE,
                "validation_start": VALIDATION_START_DATE,
                "validation_end": VALIDATION_END_DATE,
                "optuna_start": OPTUNA_START_DATE,
                "optuna_end": OPTUNA_END_DATE,
                "best_optuna_parameters": best_params,
            },
            f,
            indent=4,
        )

    print("\n")
    print("=" * 70)
    print(
        f"FINAL TEST METRICS "
        f"(from {TEST_START_DATE})"
    )
    print("=" * 70)

    for name, value in metrics.items():

        if "%" in name:

            print(
                f"{name:<35}: "
                f"{value:.2f}%"
            )

        elif "Pearson" in name:

            print(
                f"{name:<35}: "
                f"{value:.4f}"
            )

        else:

            print(
                f"{name:<35}: "
                f"{value:.6f}"
            )

    # --------------------------------------------------------
    # Timing summary
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)

    print(
        f"Data loading:     "
        f"{TIMINGS['data_loading_seconds']:.2f} sec"
    )

    print(
        f"Model loading:    "
        f"{TIMINGS['model_loading_seconds']:.2f} sec"
    )

    print(
        f"Optuna:            "
        f"{TIMINGS['optuna_seconds'] / 3600:.2f} hours"
    )

    print(
        f"Walk-forward:      "
        f"{TIMINGS['walk_forward_seconds'] / 3600:.2f} hours"
    )

    print(
        f"Total:             "
        f"{TIMINGS['total_seconds'] / 3600:.2f} hours"
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    create_plot(
        results,
        metrics,
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
        f"Timing:      {TIMING_PATH}"
    )

    print(
        f"Plotly:      {PLOT_PATH}"
    )


if __name__ == "__main__":
    maind()
