import os
import sys
import json
import warnings
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


warnings.filterwarnings("ignore")


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

BASE_DIR = Path("/home/soq/__shutupandbendover/yatin/baseline")
KRONOS_DIR = Path("/home/soq/Kronos")

DATA_PATH = BASE_DIR / "NIFTY50_5Y_OHLCV.parquet"
RESULTS_DIR = BASE_DIR / "baseline_results"

TOKENIZER_PATH = Path(
    "/home/soq/__shutupandbendover/het-uchiha/weights/Kronos-Tokenizer-base"
)

MODEL_PATH = Path(
    "/home/soq/__shutupandbendover/het-uchiha/weights/Kronos-base"
)


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

REQUESTED_START_DATE = pd.Timestamp("2022-01-01")

WINDOW_SIZE = 118
PRED_LEN = 1

TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 0.90
SAMPLE_COUNT = 1

MAX_CONTEXT = 512

SHOW_VERBOSE_MODEL_OUTPUT = False

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# IMPORT KRONOS
# =============================================================================

def import_kronos():
    print("=" * 80)
    print("IMPORTING KRONOS BASE")
    print("=" * 80)

    if str(KRONOS_DIR) not in sys.path:
        sys.path.insert(0, str(KRONOS_DIR))

    try:
        from model.kronos import Kronos
        from model.kronos import KronosTokenizer
        from model.kronos import KronosPredictor

        print("Kronos import successful.")
        print("Imported classes:")
        print("  - Kronos")
        print("  - KronosTokenizer")
        print("  - KronosPredictor")

        return Kronos, KronosTokenizer, KronosPredictor

    except Exception as e:
        print("\nERROR: Could not import Kronos.")
        print("Details:", str(e))
        raise


# =============================================================================
# COLUMN NORMALIZATION
# =============================================================================

def normalize_column_name(column):
    """
    Handles normal columns such as:
        close

    Also handles string representations such as:
        ('close', '^NSEI')
        ('Close', '^NSEI')
    """

    original = str(column).strip()

    lower = original.lower()

    standard_names = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "date": "date",
        "datetime": "date",
        "timestamp": "date",
        "timestamps": "date",
    }

    if lower in standard_names:
        return standard_names[lower]

    # Try parsing tuple-like column strings
    try:
        parsed = ast.literal_eval(original)

        if isinstance(parsed, tuple) and len(parsed) > 0:
            first = str(parsed[0]).strip().lower()

            if first in standard_names:
                return standard_names[first]

    except Exception:
        pass

    # Regex fallback
    for key in ["open", "high", "low", "close", "volume"]:
        if re.search(rf"\b{key}\b", lower):
            return key

    return lower


def load_ohlcv_data(data_path):
    print("\n" + "=" * 80)
    print("LOADING NIFTY50 OHLCV DATA")
    print("=" * 80)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_parquet(data_path)

    print("\nOriginal columns:")
    print(list(df.columns))

    # ================================================================
    # FIX MULTIINDEX / TUPLE COLUMNS
    # Handles:
    # 1. Pandas MultiIndex columns
    # 2. Tuple columns such as ('Close', '^NSEI')
    # 3. Normal string columns such as Close, close, OPEN, etc.
    # ================================================================

    new_columns = []

    for col in df.columns:

        # Case 1: Tuple column
        # Example: ('Close', '^NSEI') -> 'close'
        if isinstance(col, tuple):
            base_name = str(col[0]).strip().lower()

        # Case 2: Normal string column
        else:
            base_name = str(col).strip().lower()

            # Extra protection if a tuple was converted to a string
            # Example: "('Close', '^NSEI')" -> "close"
            if base_name.startswith("("):
                if "close" in base_name:
                    base_name = "close"
                elif "open" in base_name:
                    base_name = "open"
                elif "high" in base_name:
                    base_name = "high"
                elif "low" in base_name:
                    base_name = "low"
                elif "volume" in base_name:
                    base_name = "volume"

        new_columns.append(base_name)

    df.columns = new_columns

    print("\nColumns after normalization:")
    print(list(df.columns))

    # ================================================================
    # REMOVE DUPLICATE COLUMN NAMES IF ANY
    # ================================================================
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # ================================================================
    # REQUIRED OHLCV COLUMNS
    # ================================================================
    required_columns = ["open", "high", "low", "close", "volume"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {missing_columns}\n"
            f"Available columns after normalization: {list(df.columns)}"
        )

    # ================================================================
    # KEEP ONLY REQUIRED COLUMNS
    # ================================================================
    df = df[required_columns].copy()

    # ================================================================
    # FIX DATE INDEX
    # ================================================================
    if not isinstance(df.index, pd.DatetimeIndex):

        # Check whether Date is a normal column
        possible_date_columns = [
            "date",
            "datetime",
            "timestamp",
            "time"
        ]

        date_column = None

        for col in possible_date_columns:
            if col in df.columns:
                date_column = col
                break

        if date_column is not None:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)

        else:
            try:
                df.index = pd.to_datetime(df.index)
            except Exception as e:
                raise ValueError(
                    "Could not convert the DataFrame index to datetime."
                ) from e

    # ================================================================
    # SORT DATA
    # ================================================================
    df = df.sort_index()

    # Remove duplicate dates
    df = df[~df.index.duplicated(keep="last")]

    # ================================================================
    # CONVERT ALL OHLCV VALUES TO NUMERIC
    # ================================================================
    for col in required_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ================================================================
    # REMOVE INVALID VALUES
    # ================================================================
    rows_before = len(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=required_columns)

    rows_after = len(df)

    if rows_after < rows_before:
        print(
            f"\nRemoved {rows_before - rows_after} rows "
            f"containing invalid/missing OHLCV values."
        )

    # ================================================================
    # ENSURE VALID PRICE VALUES
    # ================================================================
    df = df[
        (df["open"] > 0) &
        (df["high"] > 0) &
        (df["low"] > 0) &
        (df["close"] > 0) &
        (df["volume"] >= 0)
    ].copy()

    # ================================================================
    # FINAL VALIDATION
    # ================================================================
    if len(df) == 0:
        raise ValueError("No valid OHLCV rows remain after cleaning.")

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    print("\nData successfully loaded.")

    print(f"\nData range: {df.index.min().date()} to {df.index.max().date()}")
    print(f"Total trading days: {len(df)}")

    print("\nColumns passed to Kronos:")
    print(list(df.columns))

    print("\nFirst rows:")
    print(df.head())

    return df

# =============================================================================
# FIND FIRST VALID PREDICTION POSITION
# =============================================================================

def get_prediction_start_position(df, requested_start, window_size):

    requested_start = pd.Timestamp(requested_start)

    requested_position = int(
        df.index.searchsorted(requested_start)
    )

    if requested_position >= len(df):
        raise ValueError(
            "Requested prediction date is after the dataset."
        )

    first_valid_position = max(
        requested_position,
        window_size
    )

    available_history = requested_position

    print("\n" + "=" * 80)
    print("WALK-FORWARD PREDICTION RANGE")
    print("=" * 80)

    print(
        f"\nRequested prediction start: "
        f"{requested_start.date()}"
    )

    print(
        f"Sliding window size: "
        f"{window_size} trading days"
    )

    print(
        f"Trading days available before requested start: "
        f"{available_history}"
    )

    if available_history < window_size:

        missing_days = window_size - available_history

        print(
            f"\nNOTE: {requested_start.date()} does not have enough "
            f"history for a {window_size}-trading-day window."
        )

        print(
            f"Additional trading days required: "
            f"{missing_days}"
        )

        print(
            "\nPrediction start automatically moved to the first "
            f"date with a complete {window_size}-day historical window."
        )

    actual_first_date = df.index[first_valid_position]

    total_predictions = len(df) - first_valid_position

    print(
        f"\nActual first prediction date: "
        f"{actual_first_date.date()}"
    )

    print(
        f"Last prediction date:         "
        f"{df.index[-1].date()}"
    )

    print(
        f"Total one-step-ahead predictions: "
        f"{total_predictions}"
    )

    return first_valid_position


# =============================================================================
# LOAD PRETRAINED KRONOS BASE
# =============================================================================

def load_kronos_predictor(
    Kronos,
    KronosTokenizer,
    KronosPredictor
):

    print("\n" + "=" * 80)
    print("LOADING PRETRAINED KRONOS BASE MODEL")
    print("=" * 80)

    print("\nTokenizer path:")
    print(TOKENIZER_PATH)

    print("\nModel path:")
    print(MODEL_PATH)

    if not TOKENIZER_PATH.exists():
        raise FileNotFoundError(
            f"Tokenizer directory not found:\n{TOKENIZER_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model directory not found:\n{MODEL_PATH}"
        )

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    print(f"\nUsing device: {device}")

    print("\nLoading tokenizer...")

    tokenizer = KronosTokenizer.from_pretrained(
        str(TOKENIZER_PATH)
    )

    print("Loading Kronos Base model...")

    model = Kronos.from_pretrained(
        str(MODEL_PATH)
    )

    model.eval()

    print("\nCreating Kronos predictor...")

    predictor = KronosPredictor(
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_context=MAX_CONTEXT
    )

    print("Kronos Base ready.")

    print("\nIMPORTANT:")
    print("  No LoRA is loaded.")
    print("  No fine-tuning is performed.")
    print("  Original pretrained Kronos Base weights are unchanged.")

    return predictor


# =============================================================================
# FIX PREDICTED OHLC VALUES
# =============================================================================

def fix_predicted_ohlc(prediction_df):
    """
    Kronos can theoretically output OHLC values where:
        high < open/close
    or:
        low > open/close

    For a valid candlestick, force:
        high = maximum of O,H,L,C
        low  = minimum of O,H,L,C
    """

    pred = prediction_df.copy()

    required = ["open", "high", "low", "close"]

    for col in required:
        pred[col] = pd.to_numeric(
            pred[col],
            errors="coerce"
        )

    if pred[required].isnull().values.any():
        raise ValueError(
            "Kronos prediction contains NaN values."
        )

    values = pred[required]

    pred["high"] = values.max(axis=1)
    pred["low"] = values.min(axis=1)

    if "volume" not in pred.columns:
        pred["volume"] = 0.0

    pred["volume"] = (
        pd.to_numeric(
            pred["volume"],
            errors="coerce"
        )
        .fillna(0.0)
        .clip(lower=0.0)
    )

    return pred


# =============================================================================
# ONE-DAY PREDICTION
# =============================================================================

def predict_next_day(
    predictor,
    context_df,
    context_dates,
    target_date
):

    # IMPORTANT:
    # Kronos calc_time_stamps uses x_timestamp.dt
    # Therefore we pass pandas Series, NOT DatetimeIndex.

    x_timestamp = pd.Series(
        pd.to_datetime(context_dates).to_numpy()
    )

    y_timestamp = pd.Series(
        [pd.Timestamp(target_date)]
    )

    prediction_df = predictor.predict(
        df=context_df[
            ["open", "high", "low", "close", "volume"]
        ].copy(),

        x_timestamp=x_timestamp,

        y_timestamp=y_timestamp,

        pred_len=1,

        T=TEMPERATURE,

        top_k=TOP_K,

        top_p=TOP_P,

        sample_count=SAMPLE_COUNT,

        verbose=SHOW_VERBOSE_MODEL_OUTPUT
    )

    prediction_df = fix_predicted_ohlc(
        prediction_df
    )

    return prediction_df


# =============================================================================
# TEST FIRST PREDICTION
# =============================================================================

def test_first_prediction(
    predictor,
    df,
    start_position
):

    print("\n" + "=" * 80)
    print("TESTING FIRST ZERO-SHOT KRONOS PREDICTION")
    print("=" * 80)

    context_start_position = (
        start_position - WINDOW_SIZE
    )

    context_end_position = (
        start_position - 1
    )

    context_df = df.iloc[
        context_start_position:
        start_position
    ].copy()

    context_dates = context_df.index

    target_date = df.index[start_position]

    actual_row = df.iloc[start_position]

    print(
        f"\nContext starts: "
        f"{context_dates[0].date()}"
    )

    print(
        f"Context ends:   "
        f"{context_dates[-1].date()}"
    )

    print(
        f"Context days:   "
        f"{len(context_df)}"
    )

    print(
        f"\nNext-day target: "
        f"{target_date.date()}"
    )

    print("\nActual next-day OHLCV:")
    print(actual_row)

    print(
        "\nRunning one Kronos Base zero-shot test prediction..."
    )

    prediction_df = predict_next_day(
        predictor=predictor,
        context_df=context_df,
        context_dates=context_dates,
        target_date=target_date
    )

    print("\nPredicted next-day OHLCV:")
    print(prediction_df.iloc[0])

    return prediction_df


# =============================================================================
# WALK-FORWARD PREDICTION
# =============================================================================

def run_walk_forward_predictions(
    predictor,
    df,
    start_position
):

    print("\n" + "=" * 80)
    print("RUNNING WALK-FORWARD ZERO-SHOT PREDICTIONS")
    print("=" * 80)

    prediction_records = []

    total_predictions = len(df) - start_position

    for count, target_position in enumerate(
        range(start_position, len(df)),
        start=1
    ):

        context_start = (
            target_position - WINDOW_SIZE
        )

        context_end = target_position

        context_df = df.iloc[
            context_start:
            context_end
        ].copy()

        context_dates = context_df.index

        target_date = df.index[target_position]

        try:

            prediction_df = predict_next_day(
                predictor=predictor,
                context_df=context_df,
                context_dates=context_dates,
                target_date=target_date
            )

            predicted_row = prediction_df.iloc[0]

            actual_row = df.iloc[target_position]

            record = {
                "Date": target_date,

                "actual_open": float(actual_row["open"]),
                "actual_high": float(actual_row["high"]),
                "actual_low": float(actual_row["low"]),
                "actual_close": float(actual_row["close"]),
                "actual_volume": float(actual_row["volume"]),

                "predicted_open": float(predicted_row["open"]),
                "predicted_high": float(predicted_row["high"]),
                "predicted_low": float(predicted_row["low"]),
                "predicted_close": float(predicted_row["close"]),
                "predicted_volume": float(
                    predicted_row.get(
                        "volume",
                        0.0
                    )
                ),
            }

            prediction_records.append(record)

        except Exception as e:

            print(
                f"\nERROR on {target_date.date()}: {e}"
            )

            raise

        if (
            count == 1
            or count % 25 == 0
            or count == total_predictions
        ):

            print(
                f"Completed {count}/{total_predictions} "
                f"predictions | "
                f"Target: {target_date.date()}"
            )

    results_df = pd.DataFrame(
        prediction_records
    )

    results_df["Date"] = pd.to_datetime(
        results_df["Date"]
    )

    results_df = results_df.sort_values(
        "Date"
    ).reset_index(drop=True)

    return results_df


# =============================================================================
# CALCULATE RETURN-BASED METRICS
# =============================================================================

def calculate_return_metrics(
    results_df,
    full_df,
    start_position
):
    """
    ALL predictive metrics are calculated on RETURNS.

    Actual return:
        actual_close(t) / actual_close(t-1) - 1

    Predicted return:
        predicted_close(t) / actual_close(t-1) - 1

    The same known previous actual close is used as the denominator
    for both actual and predicted returns.

    Therefore this is genuinely a one-step-ahead return evaluation.
    """

    metrics_df = results_df.copy()

    first_previous_close = float(
        full_df.iloc[start_position - 1]["close"]
    )

    previous_actual_close = np.concatenate(
        [
            [first_previous_close],
            metrics_df["actual_close"]
            .iloc[:-1]
            .to_numpy(dtype=float)
        ]
    )

    actual_close = (
        metrics_df["actual_close"]
        .to_numpy(dtype=float)
    )

    predicted_close = (
        metrics_df["predicted_close"]
        .to_numpy(dtype=float)
    )

    actual_returns = (
        actual_close / previous_actual_close
    ) - 1.0

    predicted_returns = (
        predicted_close / previous_actual_close
    ) - 1.0

    metrics_df["previous_actual_close"] = (
        previous_actual_close
    )

    metrics_df["actual_return"] = (
        actual_returns
    )

    metrics_df["predicted_return"] = (
        predicted_returns
    )

    metrics_df["close_difference"] = (
        metrics_df["actual_close"]
        - metrics_df["predicted_close"]
    )

    metrics_df["return_difference"] = (
        metrics_df["actual_return"]
        - metrics_df["predicted_return"]
    )

    metrics_df["actual_return_pct"] = (
        metrics_df["actual_return"] * 100.0
    )

    metrics_df["predicted_return_pct"] = (
        metrics_df["predicted_return"] * 100.0
    )

    # -------------------------------------------------------------------------
    # Remove invalid values
    # -------------------------------------------------------------------------

    evaluation_mask = (
        np.isfinite(actual_returns)
        & np.isfinite(predicted_returns)
    )

    y_true = actual_returns[evaluation_mask]
    y_pred = predicted_returns[evaluation_mask]

    if len(y_true) < 2:
        raise ValueError(
            "Not enough valid return observations "
            "to calculate metrics."
        )

    # -------------------------------------------------------------------------
    # RMSE ON RETURNS
    # -------------------------------------------------------------------------

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )

    # -------------------------------------------------------------------------
    # MAE ON RETURNS
    # -------------------------------------------------------------------------

    mae = float(
        mean_absolute_error(
            y_true,
            y_pred
        )
    )

    # -------------------------------------------------------------------------
    # MAPE ON RETURNS
    # Exclude returns extremely close to zero to avoid division explosion.
    # -------------------------------------------------------------------------

    epsilon = 1e-10

    nonzero_mask = np.abs(y_true) > epsilon

    if nonzero_mask.sum() > 0:

        mape = float(
            np.mean(
                np.abs(
                    (
                        y_true[nonzero_mask]
                        - y_pred[nonzero_mask]
                    )
                    /
                    y_true[nonzero_mask]
                )
            )
            * 100.0
        )

    else:
        mape = np.nan

    # -------------------------------------------------------------------------
    # R^2 ON RETURNS
    # -------------------------------------------------------------------------

    if np.std(y_true) > 0:
        r2 = float(
            r2_score(
                y_true,
                y_pred
            )
        )
    else:
        r2 = np.nan

    # -------------------------------------------------------------------------
    # DIRECTION ACCURACY ON RETURNS
    # -------------------------------------------------------------------------

    actual_direction = np.sign(y_true)
    predicted_direction = np.sign(y_pred)

    direction_accuracy = float(
        np.mean(
            actual_direction
            == predicted_direction
        )
        * 100.0
    )

    # -------------------------------------------------------------------------
    # TOTAL RETURN (% CHANGE)
    #
    # Actual:
    # (final actual close / close before first prediction - 1) * 100
    #
    # Predicted:
    # Compound predicted one-step returns.
    # -------------------------------------------------------------------------

    actual_total_return = float(
        (
            (
                actual_close[-1]
                / first_previous_close
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
    # PEARSON CORRELATION ON RETURNS
    # -------------------------------------------------------------------------

    if (
        len(y_true) > 1
        and np.std(y_true) > 0
        and np.std(y_pred) > 0
    ):

        pearson_correlation = float(
            np.corrcoef(
                y_true,
                y_pred
            )[0, 1]
        )

    else:
        pearson_correlation = np.nan

    metrics = {

        "evaluation_basis":
            "Daily one-step-ahead Close returns only",

        "number_of_predictions":
            int(len(metrics_df)),

        "RMSE":
            rmse,

        "MAE":
            mae,

        "MAPE_percent":
            mape,

        "R2":
            r2,

        "Direction_Accuracy_percent":
            direction_accuracy,

        "Actual_Return_percent":
            actual_total_return,

        "Predicted_Return_percent":
            predicted_total_return,

        "Pearson_Correlation":
            pearson_correlation,
    }

    return metrics_df, metrics


# =============================================================================
# CREATE DASHBOARD
# =============================================================================

def create_dashboard(
    results_df,
    metrics
):

    print("\n" + "=" * 80)
    print("CREATING INTERACTIVE DASHBOARD")
    print("=" * 80)

    plot_df = results_df.copy()

    plot_df["Date"] = pd.to_datetime(
        plot_df["Date"]
    )

    # -------------------------------------------------------------------------
    # SUBPLOT STRUCTURE
    #
    # LEFT:
    #   Row 1 = Actual Candlestick
    #   Row 2 = Predicted Candlestick
    #   Row 3 = Actual Close - Predicted Close
    #
    # RIGHT:
    #   One separate metrics table spanning all rows.
    #
    # This prevents the metrics from hiding any graph.
    # -------------------------------------------------------------------------

    fig = make_subplots(
        rows=3,
        cols=2,

        specs=[
            [
                {"type": "candlestick"},
                {
                    "type": "table",
                    "rowspan": 3
                }
            ],

            [
                {"type": "candlestick"},
                None
            ],

            [
                {"type": "scatter"},
                None
            ],
        ],

        column_widths=[
            0.80,
            0.20
        ],

        row_heights=[
            0.40,
            0.40,
            0.20
        ],

        vertical_spacing=0.08,

        horizontal_spacing=0.04,

        subplot_titles=[
            "1. Actual NIFTY 50 Daily OHLCV",
            "Held-Out Return Metrics",
            "2. Kronos Base Zero-Shot Predicted Daily OHLCV",
            "",
            "3. Close Price Difference (Actual Close − Predicted Close)",
            "",
        ]
    )

    # =========================================================================
    # 1. ACTUAL CANDLESTICK
    # =========================================================================

    fig.add_trace(

        go.Candlestick(

            x=plot_df["Date"],

            open=plot_df["actual_open"],
            high=plot_df["actual_high"],
            low=plot_df["actual_low"],
            close=plot_df["actual_close"],

            name="Actual OHLCV",

            increasing_line_color="#16a085",
            decreasing_line_color="#e74c3c",

            increasing_fillcolor="#16a085",
            decreasing_fillcolor="#e74c3c",

            showlegend=False
        ),

        row=1,
        col=1
    )

    # =========================================================================
    # 2. PREDICTED CANDLESTICK
    # =========================================================================

    fig.add_trace(

        go.Candlestick(

            x=plot_df["Date"],

            open=plot_df["predicted_open"],
            high=plot_df["predicted_high"],
            low=plot_df["predicted_low"],
            close=plot_df["predicted_close"],

            name="Predicted OHLCV",

            increasing_line_color="#2980b9",
            decreasing_line_color="#f39c12",

            increasing_fillcolor="#2980b9",
            decreasing_fillcolor="#f39c12",

            showlegend=False
        ),

        row=2,
        col=1
    )

    # =========================================================================
    # 3. ONLY ONE CLOSE PRICE DIFFERENCE GRAPH
    #
    # Actual Close - Predicted Close
    # =========================================================================

    fig.add_trace(

        go.Scatter(

            x=plot_df["Date"],

            y=plot_df["close_difference"],

            mode="lines",

            name="Actual Close - Predicted Close",

            line=dict(
                color="#8e44ad",
                width=1.5
            ),

            showlegend=False
        ),

        row=3,
        col=1
    )

    # Zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1,
        line_color="gray",
        row=3,
        col=1
    )

    # =========================================================================
    # METRICS TABLE
    #
    # IMPORTANT:
    # All predictive metrics below are calculated on RETURNS.
    # =========================================================================

    metric_names = [
        "<b>RMSE</b>",
        "<b>MAE</b>",
        "<b>MAPE</b>",
        "<b>R²</b>",
        "<b>Direction Accuracy</b>",
        "<b>Return (% change)</b>",
        "<b>Pearson Correlation</b>",
    ]

    metric_values = [

        f"{metrics['RMSE']:.8f}",

        f"{metrics['MAE']:.8f}",

        (
            f"{metrics['MAPE_percent']:.4f}%"
            if np.isfinite(metrics["MAPE_percent"])
            else "N/A"
        ),

        (
            f"{metrics['R2']:.6f}"
            if np.isfinite(metrics["R2"])
            else "N/A"
        ),

        (
            f"{metrics['Direction_Accuracy_percent']:.2f}%"
        ),

        (
            f"Actual: {metrics['Actual_Return_percent']:.2f}%"
            "<br>"
            f"Predicted: {metrics['Predicted_Return_percent']:.2f}%"
        ),

        (
            f"{metrics['Pearson_Correlation']:.6f}"
            if np.isfinite(
                metrics["Pearson_Correlation"]
            )
            else "N/A"
        ),
    ]

    fig.add_trace(

        go.Table(

            header=dict(

                values=[
                    "<b>METRIC</b>",
                    "<b>VALUE</b>"
                ],

                align=[
                    "left",
                    "right"
                ],

                font=dict(
                    size=15,
                    color="white"
                ),

                fill_color="#2c3e50",

                height=36
            ),

            cells=dict(

                values=[
                    metric_names,
                    metric_values
                ],

                align=[
                    "left",
                    "right"
                ],

                font=dict(
                    size=13,
                    color="#2c3e50"
                ),

                fill_color=[
                    ["#f8f9fa"] * len(metric_names),
                    ["#ffffff"] * len(metric_names)
                ],

                height=42
            ),

            columnwidth=[
                150,
                170
            ]
        ),

        row=1,
        col=2
    )

    # =========================================================================
    # AXIS TITLES
    # =========================================================================

    fig.update_yaxes(
        title_text="Actual Price",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="Predicted Price",
        row=2,
        col=1
    )

    fig.update_yaxes(
        title_text="Actual Close − Predicted Close",
        row=3,
        col=1
    )

    fig.update_xaxes(
        title_text="Trading Day",
        row=3,
        col=1
    )

    # =========================================================================
    # HIDE RANGE SLIDERS
    # =========================================================================

    fig.update_xaxes(
        rangeslider_visible=False,
        row=1,
        col=1
    )

    fig.update_xaxes(
        rangeslider_visible=False,
        row=2,
        col=1
    )

    # =========================================================================
    # DASHBOARD TITLE
    # =========================================================================

    fig.update_layout(

        title=dict(

            text=(
                "<b>NIFTY 50 — Kronos Base Zero-Shot Dashboard</b>"
                "<br>"
                "<sup>"
                "118-Day Sliding Window | "
                "Separate Actual and Predicted Candlestick Charts | "
                "One Close Difference Graph | "
                "Return-Based Metrics Only | "
                "No LoRA | No Fine-Tuning"
                "</sup>"
            ),

            x=0.01,
            xanchor="left"
        ),

        template="plotly_white",

        height=1500,

        width=1900,

        margin=dict(
            l=70,
            r=30,
            t=110,
            b=70
        ),

        showlegend=False,

        hovermode="x unified"
    )

    # Improve subplot title appearance
    for annotation in fig.layout.annotations:

        annotation.font.size = 17

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    html_path = (
        RESULTS_DIR
        / "nifty50_kronos_base_zero_shot_dashboard.html"
    )

    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True
    )

    print("\nDashboard saved:")
    print(html_path)

    return html_path


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    results_df,
    metrics
):

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    csv_path = (
        RESULTS_DIR
        / "nifty50_kronos_base_zero_shot_predictions.csv"
    )

    parquet_path = (
        RESULTS_DIR
        / "nifty50_kronos_base_zero_shot_predictions.parquet"
    )

    metrics_path = (
        RESULTS_DIR
        / "metrics.json"
    )

    results_df.to_csv(
        csv_path,
        index=False
    )

    results_df.to_parquet(
        parquet_path,
        index=False
    )

    # Convert NumPy types safely for JSON
    clean_metrics = {}

    for key, value in metrics.items():

        if isinstance(
            value,
            (np.floating, np.integer)
        ):
            clean_metrics[key] = value.item()

        else:
            clean_metrics[key] = value

    with open(
        metrics_path,
        "w"
    ) as f:

        json.dump(
            clean_metrics,
            f,
            indent=4
        )

    print("\nSaved prediction CSV:")
    print(csv_path)

    print("\nSaved prediction Parquet:")
    print(parquet_path)

    print("\nSaved metrics JSON:")
    print(metrics_path)


# =============================================================================
# PRINT METRICS
# =============================================================================

def print_metrics(metrics):

    print("\n" + "=" * 80)
    print("FINAL HELD-OUT METRICS")
    print("=" * 80)

    print(
        "\nIMPORTANT: "
        "RMSE, MAE, MAPE, R², Direction Accuracy, and Pearson "
        "Correlation are evaluated on DAILY CLOSE RETURNS."
    )

    print(
        "\nReturn definition:"
    )

    print(
        "Actual Return(t) = "
        "Actual Close(t) / Actual Close(t-1) - 1"
    )

    print(
        "Predicted Return(t) = "
        "Predicted Close(t) / Actual Close(t-1) - 1"
    )

    print(
        f"\nNumber of predictions: "
        f"{metrics['number_of_predictions']}"
    )

    print(
        f"\nRMSE (returns): "
        f"{metrics['RMSE']:.8f}"
    )

    print(
        f"MAE (returns): "
        f"{metrics['MAE']:.8f}"
    )

    if np.isfinite(
        metrics["MAPE_percent"]
    ):
        print(
            f"MAPE (returns): "
            f"{metrics['MAPE_percent']:.4f}%"
        )
    else:
        print(
            "MAPE (returns): N/A"
        )

    if np.isfinite(
        metrics["R2"]
    ):
        print(
            f"R² (returns): "
            f"{metrics['R2']:.6f}"
        )
    else:
        print(
            "R² (returns): N/A"
        )

    print(
        f"Direction Accuracy (returns): "
        f"{metrics['Direction_Accuracy_percent']:.2f}%"
    )

    print(
        f"Actual Return (% change): "
        f"{metrics['Actual_Return_percent']:.2f}%"
    )

    print(
        f"Predicted Return (% change): "
        f"{metrics['Predicted_Return_percent']:.2f}%"
    )

    if np.isfinite(
        metrics["Pearson_Correlation"]
    ):
        print(
            f"Pearson Correlation (returns): "
            f"{metrics['Pearson_Correlation']:.6f}"
        )
    else:
        print(
            "Pearson Correlation (returns): N/A"
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 80)
    print("ZERO-SHOT KRONOS BASELINE")
    print("=" * 80)

    print("\nConfiguration:")

    print(
        f"Requested prediction start: "
        f"{REQUESTED_START_DATE.date()}"
    )

    print(
        f"Sliding window size: "
        f"{WINDOW_SIZE} trading days"
    )

    print(
        f"Prediction length: "
        f"{PRED_LEN} day"
    )

    print(
        "Model: Pretrained Kronos Base"
    )

    print(
        "LoRA: NOT USED"
    )

    print(
        "Fine-tuning: NOT USED"
    )

    print(
        "Evaluation: ALL predictive metrics "
        "calculated on daily Close returns"
    )

    # -------------------------------------------------------------------------
    # 1. IMPORT KRONOS
    # -------------------------------------------------------------------------

    (
        Kronos,
        KronosTokenizer,
        KronosPredictor
    ) = import_kronos()

    # -------------------------------------------------------------------------
    # 2. LOAD DATA
    # -------------------------------------------------------------------------

    df = load_ohlcv_data(
        DATA_PATH
    )

    # -------------------------------------------------------------------------
    # 3. FIND FIRST VALID PREDICTION
    # -------------------------------------------------------------------------

    start_position = get_prediction_start_position(
        df=df,
        requested_start=REQUESTED_START_DATE,
        window_size=WINDOW_SIZE
    )

    # -------------------------------------------------------------------------
    # 4. LOAD MODEL
    # -------------------------------------------------------------------------

    predictor = load_kronos_predictor(
        Kronos=Kronos,
        KronosTokenizer=KronosTokenizer,
        KronosPredictor=KronosPredictor
    )

    # -------------------------------------------------------------------------
    # 5. TEST FIRST PREDICTION
    # -------------------------------------------------------------------------

    test_first_prediction(
        predictor=predictor,
        df=df,
        start_position=start_position
    )

    # -------------------------------------------------------------------------
    # 6. WALK-FORWARD PREDICTIONS
    # -------------------------------------------------------------------------

    raw_results_df = run_walk_forward_predictions(
        predictor=predictor,
        df=df,
        start_position=start_position
    )

    # -------------------------------------------------------------------------
    # 7. CALCULATE RETURN-BASED METRICS
    # -------------------------------------------------------------------------

    results_df, metrics = calculate_return_metrics(
        results_df=raw_results_df,
        full_df=df,
        start_position=start_position
    )

    # -------------------------------------------------------------------------
    # 8. PRINT METRICS
    # -------------------------------------------------------------------------

    print_metrics(
        metrics
    )

    # -------------------------------------------------------------------------
    # 9. SAVE DATA
    # -------------------------------------------------------------------------

    save_results(
        results_df=results_df,
        metrics=metrics
    )

    # -------------------------------------------------------------------------
    # 10. CREATE DASHBOARD
    # -------------------------------------------------------------------------

    dashboard_path = create_dashboard(
        results_df=results_df,
        metrics=metrics
    )

    # -------------------------------------------------------------------------
    # FINISHED
    # -------------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("ZERO-SHOT BASELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)

    print(
        "\nFinal dashboard:"
    )

    print(
        dashboard_path
    )

    print(
        "\nGraph layout:"
    )

    print(
        "1. Actual OHLCV Candlestick"
    )

    print(
        "2. Predicted OHLCV Candlestick"
    )

    print(
        "3. Actual Close - Predicted Close"
    )

    print(
        "4. Separate return-based metrics table on the right"
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()