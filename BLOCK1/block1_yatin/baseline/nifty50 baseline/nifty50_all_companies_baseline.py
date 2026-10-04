import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from scipy.stats import pearsonr

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# CONFIGURATION
# ============================================================

# Kronos repository
KRONOS_PATH = "/home/soq/Kronos"

# Kronos tokenizer
TOKENIZER_PATH = (
    "/home/soq/__shutupandbendover/het-uchiha/"
    "weights/Kronos-Tokenizer-base"
)

# Kronos Base model
MODEL_PATH = (
    "/home/soq/__shutupandbendover/het-uchiha/"
    "weights/Kronos-base"
)

# ------------------------------------------------------------
# YOUR NIFTY 50 COMPANY DATA
# ------------------------------------------------------------

DATA_DIR = Path(
    "/home/soq/__shutupandbendover/yatin/"
    "nifty50 baseline/NIFTY50_OHLCV"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    "/home/soq/__shutupandbendover/yatin/"
    "nifty50 baseline/results"
)

PREDICTION_DIR = OUTPUT_DIR / "predictions"
DASHBOARD_DIR = OUTPUT_DIR / "dashboards"
METRICS_DIR = OUTPUT_DIR / "metrics"

# ------------------------------------------------------------
# EXPERIMENT SETTINGS
# ------------------------------------------------------------

WINDOW_SIZE = 118
PRED_LEN = 1

# Optuna trials per company
OPTUNA_TRIALS = 10

# Number of validation predictions used by Optuna
OPTUNA_VALIDATION_DAYS = 20

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

warnings.filterwarnings("ignore")


# ============================================================
# KRONOS IMPORT
# ============================================================

sys.path.insert(0, KRONOS_PATH)

from model.kronos import (
    Kronos,
    KronosTokenizer,
    KronosPredictor
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DASHBOARD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD KRONOS BASE
# ============================================================

print()
print("=" * 80)
print("LOADING KRONOS BASE")
print("=" * 80)

print("Device:", DEVICE)

tokenizer = KronosTokenizer.from_pretrained(
    TOKENIZER_PATH
)

model = Kronos.from_pretrained(
    MODEL_PATH
)

model = model.to(DEVICE)

model.eval()

predictor = KronosPredictor(
    model,
    tokenizer,
    device=DEVICE,
    max_context=512
)

print("Kronos Base loaded successfully.")
print("No LoRA.")
print("No fine-tuning.")
print("Zero-shot baseline.")
print()


# ============================================================
# LOAD COMPANY DATA
# ============================================================

def load_company_data(file_path):
    """
    Load one company's OHLCV data.

    Supported:
        CSV
        Parquet

    Output columns:
        open
        high
        low
        close
        volume

    Index:
        DatetimeIndex
    """

    print(
        f"Loading data: {file_path.name}"
    )

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    suffix = file_path.suffix.lower()

    if suffix == ".csv":

        df = pd.read_csv(
            file_path
        )

    elif suffix in [".parquet", ".pq"]:

        df = pd.read_parquet(
            file_path
        )

    else:

        raise ValueError(
            f"Unsupported file type: "
            f"{file_path}"
        )

    # --------------------------------------------------------
    # Handle MultiIndex columns
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        # Usually OHLCV is on level 0
        df.columns = [
            str(x).strip()
            for x in df.columns.get_level_values(0)
        ]

    else:

        cleaned_columns = []

        for col in df.columns:

            col_string = str(col).strip()

            # Handle stringified tuple columns
            # such as:
            # "('Close', 'RELIANCE.NS')"

            if (
                col_string.startswith("(")
                and col_string.endswith(")")
            ):

                try:

                    import ast

                    parsed = ast.literal_eval(
                        col_string
                    )

                    if isinstance(
                        parsed,
                        tuple
                    ):

                        cleaned_columns.append(
                            str(parsed[0])
                        )

                    else:

                        cleaned_columns.append(
                            col_string
                        )

                except Exception:

                    cleaned_columns.append(
                        col_string
                    )

            else:

                cleaned_columns.append(
                    col_string
                )

        df.columns = cleaned_columns

    # --------------------------------------------------------
    # Find date column
    # --------------------------------------------------------

    date_candidates = [
        "Date",
        "date",
        "Datetime",
        "datetime",
        "Timestamp",
        "timestamp"
    ]

    date_col = None

    for col in date_candidates:

        if col in df.columns:

            date_col = col
            break

    # If no named date column,
    # use first column.

    if date_col is None:

        date_col = df.columns[0]

    # --------------------------------------------------------
    # Convert date
    # --------------------------------------------------------

    df[date_col] = pd.to_datetime(
        df[date_col],
        errors="coerce"
    )

    df = df.dropna(
        subset=[date_col]
    )

    df = df.set_index(
        date_col
    )

    # --------------------------------------------------------
    # Normalize OHLCV column names
    # --------------------------------------------------------

    rename_map = {}

    for col in df.columns:

        c = str(col).strip().lower()

        if c == "open":

            rename_map[col] = "open"

        elif c == "high":

            rename_map[col] = "high"

        elif c == "low":

            rename_map[col] = "low"

        elif c == "close":

            rename_map[col] = "close"

        elif c in [
            "volume",
            "vol"
        ]:

            rename_map[col] = "volume"

    df = df.rename(
        columns=rename_map
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "open",
        "high",
        "low",
        "close"
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:

        raise ValueError(
            f"{file_path.name}: "
            f"Missing columns: {missing}\n"
            f"Available columns: "
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    if "volume" not in df.columns:

        df["volume"] = 0.0

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Keep only OHLCV
    # --------------------------------------------------------

    df = df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]

    # --------------------------------------------------------
    # Remove bad rows
    # --------------------------------------------------------

    df = df.dropna()

    # Sort chronologically
    df = df.sort_index()

    # Remove duplicate dates
    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    return df


# ============================================================
# KRONOS ONE-DAY PREDICTION
# ============================================================

def predict_next_day(
    history_df,
    target_date,
    T,
    top_p
):
    """
    Predict exactly ONE next trading day.

    history_df contains exactly 118 previous
    ACTUAL trading days.

    target_date is the actual date being predicted.
    """

    # --------------------------------------------------------
    # Prepare OHLCV
    # --------------------------------------------------------

    x_df = history_df[
        [
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Kronos expects pandas Series for timestamps.
    # --------------------------------------------------------

    x_timestamp = pd.Series(
        x_df.index
    )

    y_timestamp = pd.Series(
        [target_date]
    )

    # --------------------------------------------------------
    # Kronos prediction
    # --------------------------------------------------------

    with torch.no_grad():

        prediction = predictor.predict(

            df=x_df,

            x_timestamp=x_timestamp,

            y_timestamp=y_timestamp,

            pred_len=1,

            T=T,

            top_p=top_p,

            sample_count=1,

            verbose=False
        )

    # First and only predicted row
    pred_row = prediction.iloc[0]

    return pred_row


# ============================================================
# OPTUNA TUNING
# ============================================================

def tune_parameters(df):
    """
    Tune Kronos inference parameters.

    Parameters:
        T
        top_p

    We use an early validation section.

    IMPORTANT:
    This validation section is separate from
    the final rolling evaluation period.
    """

    if len(df) < (
        WINDOW_SIZE +
        OPTUNA_VALIDATION_DAYS +
        1
    ):

        print(
            "Not enough data for Optuna."
        )

        return {
            "T": 1.0,
            "top_p": 0.7
        }

    # --------------------------------------------------------
    # Validation period
    #
    # Use the first 20 possible one-step
    # predictions after 118 days.
    # --------------------------------------------------------

    validation_start = WINDOW_SIZE

    validation_end = min(
        validation_start +
        OPTUNA_VALIDATION_DAYS,
        len(df)
    )

    validation_cases = []

    for target_idx in range(
        validation_start,
        validation_end
    ):

        history = df.iloc[
            target_idx - WINDOW_SIZE:
            target_idx
        ].copy()

        target_date = df.index[
            target_idx
        ]

        actual_close = float(
            df.iloc[
                target_idx
            ]["close"]
        )

        previous_actual_close = float(
            df.iloc[
                target_idx - 1
            ]["close"]
        )

        # Actual daily return
        actual_return = (
            actual_close /
            previous_actual_close
        ) - 1.0

        validation_cases.append(
            {
                "history": history,
                "target_date": target_date,
                "actual_return":
                    actual_return,
                "previous_close":
                    previous_actual_close
            }
        )

    # --------------------------------------------------------
    # Optuna objective
    # --------------------------------------------------------

    def objective(trial):

        T = trial.suggest_float(
            "T",
            0.5,
            1.5
        )

        top_p = trial.suggest_float(
            "top_p",
            0.5,
            0.95
        )

        squared_errors = []

        for case in validation_cases:

            try:

                pred = predict_next_day(

                    case["history"],

                    case["target_date"],

                    T,

                    top_p
                )

                predicted_close = float(
                    pred["close"]
                )

                previous_close = float(
                    case["previous_close"]
                )

                predicted_return = (
                    predicted_close /
                    previous_close
                ) - 1.0

                error = (
                    case["actual_return"]
                    -
                    predicted_return
                ) ** 2

                squared_errors.append(
                    error
                )

            except Exception:

                # Large penalty if prediction fails
                squared_errors.append(
                    1.0
                )

        return float(
            np.mean(
                squared_errors
            )
        )

    # --------------------------------------------------------
    # Create study
    # --------------------------------------------------------

    study = optuna.create_study(
        direction="minimize"
    )

    study.optimize(
        objective,
        n_trials=OPTUNA_TRIALS,
        show_progress_bar=False
    )

    print(
        "Optuna best parameters:",
        study.best_params
    )

    print(
        "Optuna best validation loss:",
        study.best_value
    )

    return study.best_params


# ============================================================
# ROLLING ONE-STEP PREDICTION
# ============================================================

def generate_predictions(
    df,
    T,
    top_p
):
    """
    Main prediction loop.

    EXACT LOGIC:

    118 actual days
        ↓
    predict next actual day
        ↓
    move window by 1 day
        ↓
    118 actual days
        ↓
    predict next actual day
        ↓
    repeat until final date

    NEVER use predicted values as future input.
    """

    predictions = []

    total_predictions = (
        len(df) -
        WINDOW_SIZE
    )

    print(
        f"Total possible predictions: "
        f"{total_predictions}"
    )

    # --------------------------------------------------------
    # Rolling prediction
    # --------------------------------------------------------

    for target_idx in range(
        WINDOW_SIZE,
        len(df)
    ):

        # ----------------------------------------------------
        # EXACTLY 118 PREVIOUS ACTUAL DAYS
        # ----------------------------------------------------

        history = df.iloc[
            target_idx - WINDOW_SIZE:
            target_idx
        ].copy()

        # ----------------------------------------------------
        # Actual target day
        # ----------------------------------------------------

        target_date = df.index[
            target_idx
        ]

        actual_row = df.iloc[
            target_idx
        ]

        # ----------------------------------------------------
        # Previous ACTUAL close
        # ----------------------------------------------------

        previous_actual_close = float(
            df.iloc[
                target_idx - 1
            ]["close"]
        )

        try:

            # ------------------------------------------------
            # Predict next day
            # ------------------------------------------------

            pred = predict_next_day(

                history,

                target_date,

                T,

                top_p
            )

            # ------------------------------------------------
            # Store actual OHLCV
            # ------------------------------------------------

            row = {

                "Date":
                    target_date,

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

                # --------------------------------------------
                # Store predicted OHLCV
                # --------------------------------------------

                "pred_open":
                    float(
                        pred["open"]
                    ),

                "pred_high":
                    float(
                        pred["high"]
                    ),

                "pred_low":
                    float(
                        pred["low"]
                    ),

                "pred_close":
                    float(
                        pred["close"]
                    ),

                "pred_volume":
                    float(
                        pred["volume"]
                    ),

                # --------------------------------------------
                # Previous actual close
                # --------------------------------------------

                "previous_actual_close":
                    previous_actual_close
            }

            predictions.append(
                row
            )

        except Exception as e:

            print(
                f"Prediction failed "
                f"for {target_date}: {e}"
            )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        completed = (
            target_idx -
            WINDOW_SIZE +
            1
        )

        if (
            completed == 1
            or completed % 50 == 0
            or completed == total_predictions
        ):

            print(
                f"Progress: "
                f"{completed}/"
                f"{total_predictions}"
            )

    # --------------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------------

    result = pd.DataFrame(
        predictions
    )

    if len(result) == 0:

        raise RuntimeError(
            "No predictions were generated."
        )

    result["Date"] = pd.to_datetime(
        result["Date"]
    )

    result = result.sort_values(
        "Date"
    ).reset_index(
        drop=True
    )

    # ========================================================
    # DAILY CLOSE RETURNS
    # ========================================================

    # Actual return:
    #
    # Actual Close(t) / Actual Close(t-1) - 1
    #

    result["actual_return"] = (
        result["actual_close"] /
        result["previous_actual_close"]
    ) - 1.0

    # --------------------------------------------------------
    # Predicted return
    #
    # IMPORTANT:
    #
    # We use the SAME previous actual close.
    #
    # This is the correct one-step evaluation:
    #
    # Predicted Close(t) /
    # Actual Close(t-1) - 1
    # --------------------------------------------------------

    result["predicted_return"] = (
        result["pred_close"] /
        result["previous_actual_close"]
    ) - 1.0

    # --------------------------------------------------------
    # Return error
    # --------------------------------------------------------

    result["return_error"] = (
        result["actual_return"] -
        result["predicted_return"]
    )

    # --------------------------------------------------------
    # Squared return error
    # --------------------------------------------------------

    result["squared_return_error"] = (
        result["return_error"] ** 2
    )

    return result


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(result):

    actual = result[
        "actual_return"
    ].to_numpy(
        dtype=float
    )

    predicted = result[
        "predicted_return"
    ].to_numpy(
        dtype=float
    )

    # ========================================================
    # RMSE
    # ========================================================

    rmse_decimal = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    # ========================================================
    # MAE
    # ========================================================

    mae_decimal = mean_absolute_error(
        actual,
        predicted
    )

    # ========================================================
    # MAPE
    #
    # Ignore actual returns extremely close to zero
    # to avoid division by zero.
    # ========================================================

    non_zero_mask = (
        np.abs(actual) > 1e-8
    )

    if np.any(non_zero_mask):

        mape_decimal = np.mean(
            np.abs(
                (
                    actual[non_zero_mask]
                    -
                    predicted[non_zero_mask]
                )
                /
                actual[non_zero_mask]
            )
        )

    else:

        mape_decimal = np.nan

    # ========================================================
    # R²
    # ========================================================

    r2 = r2_score(
        actual,
        predicted
    )

    # ========================================================
    # DIRECTION ACCURACY
    # ========================================================

    direction_accuracy = np.mean(
        np.sign(actual)
        ==
        np.sign(predicted)
    )

    # ========================================================
    # PEARSON CORRELATION
    # ========================================================

    if (
        len(actual) > 1
        and
        np.std(actual) > 0
        and
        np.std(predicted) > 0
    ):

        pearson_correlation, _ = pearsonr(
            actual,
            predicted
        )

    else:

        pearson_correlation = np.nan

    # ========================================================
    # CUMULATIVE RETURN
    # ========================================================

    actual_cumulative_return = (
        np.prod(
            1.0 + actual
        ) - 1.0
    )

    predicted_cumulative_return = (
        np.prod(
            1.0 + predicted
        ) - 1.0
    )

    # ========================================================
    # IMPORTANT:
    #
    # Convert errors from decimal to percentage.
    #
    # Example:
    #
    # 0.0116 -> 1.16%
    #
    # ========================================================

    rmse_percent = (
        rmse_decimal * 100.0
    )

    mae_percent = (
        mae_decimal * 100.0
    )

    if not np.isnan(
        mape_decimal
    ):

        mape_percent = (
            mape_decimal * 100.0
        )

    else:

        mape_percent = np.nan

    direction_accuracy_percent = (
        direction_accuracy * 100.0
    )

    actual_return_percent = (
        actual_cumulative_return *
        100.0
    )

    predicted_return_percent = (
        predicted_cumulative_return *
        100.0
    )

    # ========================================================
    # RETURN METRICS
    # ========================================================

    metrics = {

        # Decimal versions
        "RMSE_decimal":
            float(rmse_decimal),

        "MAE_decimal":
            float(mae_decimal),

        "MAPE_decimal":
            (
                float(mape_decimal)
                if not np.isnan(
                    mape_decimal
                )
                else None
            ),

        # Percentage versions
        "RMSE_percent":
            float(rmse_percent),

        "MAE_percent":
            float(mae_percent),

        "MAPE_percent":
            (
                float(mape_percent)
                if not np.isnan(
                    mape_percent
                )
                else None
            ),

        # Other metrics
        "R2":
            float(r2),

        "Direction_Accuracy_percent":
            float(
                direction_accuracy_percent
            ),

        "Actual_Return_percent":
            float(
                actual_return_percent
            ),

        "Predicted_Return_percent":
            float(
                predicted_return_percent
            ),

        "Pearson_Correlation":
            (
                float(
                    pearson_correlation
                )
                if not np.isnan(
                    pearson_correlation
                )
                else None
            ),

        # Experiment information
        "Number_of_predictions":
            int(len(result)),

        "First_prediction_date":
            str(
                result[
                    "Date"
                ].iloc[0].date()
            ),

        "Last_prediction_date":
            str(
                result[
                    "Date"
                ].iloc[-1].date()
            )
    }

    return metrics


# ============================================================
# CREATE DASHBOARD
# ============================================================

def create_dashboard(
    result,
    company,
    metrics,
    best_params,
    output_file
):

    dates = result[
        "Date"
    ]

    # ========================================================
    # SUBPLOTS
    # ========================================================

    fig = make_subplots(

        rows=3,
        cols=2,

        specs=[

            [
                {
                    "type":
                        "candlestick"
                },

                {
                    "type":
                        "table",
                    "rowspan": 3
                }
            ],

            [
                {
                    "type":
                        "candlestick"
                },

                None
            ],

            [
                {
                    "type":
                        "xy"
                },

                None
            ]
        ],

        row_heights=[
            0.38,
            0.38,
            0.24
        ],

        column_widths=[
            0.80,
            0.20
        ],

        vertical_spacing=0.08,

        horizontal_spacing=0.04,

        subplot_titles=[

            "1. Actual NIFTY 50 Company Daily OHLCV",

            "2. Kronos Base Zero-Shot Predicted Daily OHLCV",

            "3. Actual Close - Predicted Close"

        ]
    )

    # ========================================================
    # ACTUAL CANDLESTICK
    # ========================================================

    fig.add_trace(

        go.Candlestick(

            x=dates,

            open=result[
                "actual_open"
            ],

            high=result[
                "actual_high"
            ],

            low=result[
                "actual_low"
            ],

            close=result[
                "actual_close"
            ],

            name="Actual"

        ),

        row=1,
        col=1
    )

    # ========================================================
    # PREDICTED CANDLESTICK
    # ========================================================

    fig.add_trace(

        go.Candlestick(

            x=dates,

            open=result[
                "pred_open"
            ],

            high=result[
                "pred_high"
            ],

            low=result[
                "pred_low"
            ],

            close=result[
                "pred_close"
            ],

            name="Predicted"

        ),

        row=2,
        col=1
    )

    # ========================================================
    # ACTUAL CLOSE - PREDICTED CLOSE
    # ========================================================

    close_difference = (
        result[
            "actual_close"
        ]
        -
        result[
            "pred_close"
        ]
    )

    fig.add_trace(

        go.Scatter(

            x=dates,

            y=close_difference,

            mode="lines",

            name=(
                "Actual Close - "
                "Predicted Close"
            )

        ),

        row=3,
        col=1
    )

    # ========================================================
    # METRICS TABLE
    # ========================================================

    metric_names = [

        "RMSE",

        "MAE",

        "MAPE",

        "R²",

        "Direction Accuracy",

        "Return (% change)",

        "Pearson Correlation"

    ]

    metric_values = [

        # RMSE in %
        f"{metrics['RMSE_percent']:.4f}%",

        # MAE in %
        f"{metrics['MAE_percent']:.4f}%",

        # MAPE in %
        (
            f"{metrics['MAPE_percent']:.4f}%"
            if metrics["MAPE_percent"]
            is not None
            else "N/A"
        ),

        # R²
        f"{metrics['R2']:.6f}",

        # Direction accuracy
        (
            f"{metrics['Direction_Accuracy_percent']:.2f}%"
        ),

        # Cumulative return
        (
            f"Actual: "
            f"{metrics['Actual_Return_percent']:.2f}%"
            f"<br>"
            f"Predicted: "
            f"{metrics['Predicted_Return_percent']:.2f}%"
        ),

        # Pearson
        (
            f"{metrics['Pearson_Correlation']:.6f}"
            if metrics[
                "Pearson_Correlation"
            ] is not None
            else "N/A"
        )
    ]

    fig.add_trace(

        go.Table(

            header=dict(

                values=[
                    "<b>METRIC</b>",
                    "<b>VALUE</b>"
                ],

                align="left",

                height=35,

                font=dict(
                    size=13
                )
            ),

            cells=dict(

                values=[
                    metric_names,
                    metric_values
                ],

                align="left",

                height=42,

                font=dict(
                    size=12
                )
            )

        ),

        row=1,
        col=2
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(

        title=dict(

            text=(
                f"{company} — "
                f"Kronos Base Zero-Shot Baseline"
                f"<br>"
                f"<sup>"
                f"118-Day Sliding Window | "
                f"One-Day Ahead Prediction | "
                f"All Predictions Use Actual Historical Data | "
                f"Return-Based Metrics | "
                f"No LoRA | No Fine-Tuning"
                f"</sup>"
            ),

            x=0.01,

            xanchor="left"
        ),

        width=1900,

        height=1200,

        template="plotly_white",

        hovermode="x unified",

        margin=dict(
            l=70,
            r=40,
            t=110,
            b=60
        ),

        xaxis_rangeslider_visible=False,

        xaxis2_rangeslider_visible=False,

        showlegend=True
    )

    # ========================================================
    # AXIS TITLES
    # ========================================================

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

        title_text=(
            "Actual Close − "
            "Predicted Close"
        ),

        row=3,
        col=1
    )

    fig.update_xaxes(

        title_text="Trading Day",

        row=3,
        col=1
    )

    # ========================================================
    # SAVE HTML
    # ========================================================

    fig.write_html(

        str(output_file),

        include_plotlyjs=True
    )


# ============================================================
# PROCESS ONE COMPANY
# ============================================================

def process_company(
    file_path,
    company_number,
    total_companies
):

    company = file_path.stem

    print()
    print("=" * 80)
    print(
        f"[{company_number}/{total_companies}] "
        f"PROCESSING: {company}"
    )
    print("=" * 80)

    # ========================================================
    # LOAD DATA
    # ========================================================

    df = load_company_data(
        file_path
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        "Date range:",
        df.index.min().date(),
        "to",
        df.index.max().date()
    )

    # ========================================================
    # CHECK WINDOW
    # ========================================================

    if len(df) <= WINDOW_SIZE:

        print(
            f"SKIPPED: {company} does not "
            f"have more than {WINDOW_SIZE} rows."
        )

        return None

    # ========================================================
    # OPTUNA
    # ========================================================

    print()
    print(
        "Running Optuna tuning..."
    )

    best_params = tune_parameters(
        df
    )

    T = float(
        best_params["T"]
    )

    top_p = float(
        best_params["top_p"]
    )

    print(
        f"Selected T = {T:.6f}"
    )

    print(
        f"Selected top_p = {top_p:.6f}"
    )

    # ========================================================
    # FINAL ROLLING PREDICTIONS
    # ========================================================

    print()
    print(
        "Starting final rolling predictions..."
    )

    result = generate_predictions(

        df,

        T,

        top_p
    )

    print()
    print(
        f"Generated "
        f"{len(result)} predictions."
    )

    # ========================================================
    # METRICS
    # ========================================================

    metrics = calculate_metrics(
        result
    )

    # ========================================================
    # SAVE PREDICTIONS
    # ========================================================

    prediction_file = (
        PREDICTION_DIR /
        f"{company}_predictions.csv"
    )

    result.to_csv(
        prediction_file,
        index=False
    )

    print(
        f"Predictions saved:\n"
        f"{prediction_file}"
    )

    # ========================================================
    # SAVE METRICS JSON
    # ========================================================

    metrics_file = (
        METRICS_DIR /
        f"{company}_metrics.json"
    )

    metrics_json = {

        "company":
            company,

        "model":
            "Kronos Base",

        "experiment":
            "Zero-Shot Baseline",

        "fine_tuning":
            False,

        "lora":
            False,

        "window_size":
            WINDOW_SIZE,

        "prediction_length":
            PRED_LEN,

        "prediction_method":
            "118 actual previous trading days -> next actual trading day",

        "uses_predicted_values_as_input":
            False,

        "optuna":
            True,

        "optuna_best_parameters":
            best_params,

        "metrics":
            metrics
    }

    with open(
        metrics_file,
        "w"
    ) as f:

        json.dump(
            metrics_json,
            f,
            indent=4
        )

    print(
        f"Metrics saved:\n"
        f"{metrics_file}"
    )

    # ========================================================
    # DASHBOARD
    # ========================================================

    dashboard_file = (
        DASHBOARD_DIR /
        f"{company}_dashboard.html"
    )

    create_dashboard(

        result,

        company,

        metrics,

        best_params,

        dashboard_file
    )

    print(
        f"Dashboard saved:\n"
        f"{dashboard_file}"
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print(
        "-" * 60
    )

    print(
        f"{company} FINAL RESULTS"
    )

    print(
        "-" * 60
    )

    print(
        f"RMSE: "
        f"{metrics['RMSE_percent']:.4f}%"
    )

    print(
        f"MAE: "
        f"{metrics['MAE_percent']:.4f}%"
    )

    if metrics[
        "MAPE_percent"
    ] is not None:

        print(
            f"MAPE: "
            f"{metrics['MAPE_percent']:.4f}%"
        )

    else:

        print(
            "MAPE: N/A"
        )

    print(
        f"R²: "
        f"{metrics['R2']:.6f}"
    )

    print(
        f"Direction Accuracy: "
        f"{metrics['Direction_Accuracy_percent']:.2f}%"
    )

    print(
        f"Actual Return: "
        f"{metrics['Actual_Return_percent']:.2f}%"
    )

    print(
        f"Predicted Return: "
        f"{metrics['Predicted_Return_percent']:.2f}%"
    )

    if metrics[
        "Pearson_Correlation"
    ] is not None:

        print(
            f"Pearson Correlation: "
            f"{metrics['Pearson_Correlation']:.6f}"
        )

    else:

        print(
            "Pearson Correlation: N/A"
        )

    print(
        f"Predictions: "
        f"{metrics['Number_of_predictions']}"
    )

    print(
        f"Prediction period: "
        f"{metrics['First_prediction_date']} "
        f"to "
        f"{metrics['Last_prediction_date']}"
    )

    print(
        "-" * 60
    )

    return {

        "Company":
            company,

        "Rows":
            len(df),

        "Predictions":
            metrics[
                "Number_of_predictions"
            ],

        "First_Prediction":
            metrics[
                "First_prediction_date"
            ],

        "Last_Prediction":
            metrics[
                "Last_prediction_date"
            ],

        "RMSE_%":
            metrics[
                "RMSE_percent"
            ],

        "MAE_%":
            metrics[
                "MAE_percent"
            ],

        "MAPE_%":
            metrics[
                "MAPE_percent"
            ],

        "R2":
            metrics[
                "R2"
            ],

        "Direction_Accuracy_%":
            metrics[
                "Direction_Accuracy_percent"
            ],

        "Actual_Return_%":
            metrics[
                "Actual_Return_percent"
            ],

        "Predicted_Return_%":
            metrics[
                "Predicted_Return_percent"
            ],

        "Pearson_Correlation":
            metrics[
                "Pearson_Correlation"
            ],

        "Optuna_T":
            T,

        "Optuna_top_p":
            top_p
    }


# ============================================================
# FIND COMPANY FILES
# ============================================================

def find_company_files():

    files = []

    # CSV
    files.extend(
        DATA_DIR.glob("*.csv")
    )

    # Parquet
    files.extend(
        DATA_DIR.glob("*.parquet")
    )

    files.extend(
        DATA_DIR.glob("*.pq")
    )

    # Remove duplicates
    files = list(
        {
            str(file): file
            for file in files
        }.values()
    )

    # Sort
    files = sorted(
        files,
        key=lambda x: x.name.lower()
    )

    return files


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("NIFTY 50 — ALL COMPANY KRONOS BASELINE")
    print("=" * 80)

    print()
    print(
        "Data directory:"
    )

    print(
        DATA_DIR
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        f"Window size: "
        f"{WINDOW_SIZE} actual trading days"
    )

    print(
        "Prediction horizon: "
        "1 trading day"
    )

    print(
        "Fine-tuning: OFF"
    )

    print(
        "LoRA: OFF"
    )

    print(
        "Optuna: ON"
    )

    # ========================================================
    # FIND FILES
    # ========================================================

    files = find_company_files()

    print()
    print(
        f"Found {len(files)} company files."
    )

    if len(files) == 0:

        raise FileNotFoundError(

            "No CSV or Parquet files found in:\n"
            f"{DATA_DIR}"

        )

    # ========================================================
    # PRINT FILES
    # ========================================================

    print()

    for i, file_path in enumerate(
        files,
        start=1
    ):

        print(
            f"{i:02d}. "
            f"{file_path.name}"
        )

    # ========================================================
    # PROCESS ALL COMPANIES
    # ========================================================

    all_results = []

    total_companies = len(
        files
    )

    for company_number, file_path in enumerate(
        files,
        start=1
    ):

        try:

            result = process_company(

                file_path,

                company_number,

                total_companies
            )

            if result is not None:

                all_results.append(
                    result
                )

        except Exception as e:

            print()
            print(
                "!" * 80
            )

            print(
                f"ERROR processing "
                f"{file_path.name}"
            )

            print(
                str(e)
            )

            print(
                "!" * 80
            )

            # Continue with next company
            continue

    # ========================================================
    # SAVE COMBINED METRICS
    # ========================================================

    if len(all_results) > 0:

        summary_df = pd.DataFrame(
            all_results
        )

        summary_file = (
            OUTPUT_DIR /
            "all_companies_metrics.csv"
        )

        summary_df.to_csv(
            summary_file,
            index=False
        )

        print()
        print("=" * 80)
        print(
            "ALL COMPANIES COMPLETED"
        )
        print("=" * 80)

        print(
            f"Successfully processed: "
            f"{len(all_results)}"
        )

        print(
            f"Failed/skipped: "
            f"{total_companies - len(all_results)}"
        )

        print()
        print(
            "Combined metrics:"
        )

        print(
            summary_file
        )

        print()
        print(
            "Output folders:"
        )

        print(
            f"Predictions: "
            f"{PREDICTION_DIR}"
        )

        print(
            f"Dashboards: "
            f"{DASHBOARD_DIR}"
        )

        print(
            f"Metrics: "
            f"{METRICS_DIR}"
        )

    else:

        print()
        print(
            "No companies completed successfully."
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()