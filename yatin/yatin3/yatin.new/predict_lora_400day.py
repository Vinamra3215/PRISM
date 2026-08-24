import os
import sys
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "HCLTECH_clean.parquet"
)

LORA_PATH = os.path.join(
    BASE_DIR,
    "checkpoints",
    "lora_best.pt"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

PLOTS_DIR = os.path.join(
    BASE_DIR,
    "plots"
)

FORECAST_PATH = os.path.join(
    RESULTS_DIR,
    "HCLTECH_rolling_predictions.parquet"
)

PLOT_PATH = os.path.join(
    PLOTS_DIR,
    "HCLTECH_rolling_prediction.html"
)

# ============================================================
# KRONOS PATH
# ============================================================

KRONOS_DIR = "/home/soq/Kronos"

sys.path.insert(0, KRONOS_DIR)

from model import Kronos, KronosTokenizer
from model.kronos import calc_time_stamps

# Reuse the already-tested LoRA implementation
from het.predict_lora_ka_loda import (
    inject_lora,
    load_lora_weights,
    generate_full_forecast,
)

# ============================================================
# MODEL
# ============================================================

TOKENIZER_PATH = (
    "/home/soq/__shutupandbendover/"
    "het-uchiha/weights/Kronos-Tokenizer-base"
)

MODEL_PATH = (
    "/home/soq/__shutupandbendover/"
    "het-uchiha/weights/Kronos-base"
)

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)

# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

WINDOW_SIZE = 400

PREDICTION_START = pd.Timestamp(
    "2022-01-01"
)

CLIP = 5

TEMPERATURE = 1.0
TOP_K = 0
TOP_P = 0.9
SAMPLE_COUNT = 1

FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]

# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading HCLTECH data...")
    print(DATA_PATH)

    df = pd.read_parquet(
        DATA_PATH
    )

    # --------------------------------------------------------
    # If Date is a normal column
    # --------------------------------------------------------

    if "Date" in df.columns:

        df["Date"] = pd.to_datetime(
            df["Date"]
        )

        df = df.set_index(
            "Date"
        )

    # --------------------------------------------------------
    # Datetime index
    # --------------------------------------------------------

    df.index = pd.to_datetime(
        df.index
    )

    # --------------------------------------------------------
    # MultiIndex columns
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex,
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )

    # --------------------------------------------------------
    # Lowercase columns
    # --------------------------------------------------------

    df.columns = [
        str(c).lower()
        for c in df.columns
    ]

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_index()

    # --------------------------------------------------------
    # Remove duplicate dates
    # --------------------------------------------------------

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    # --------------------------------------------------------
    # Amount
    # --------------------------------------------------------

    if "amount" not in df.columns:

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

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    missing = [
        c
        for c in FEATURE_COLUMNS
        if c not in df.columns
    ]

    if missing:

        raise RuntimeError(
            f"Missing columns: {missing}"
        )

    # --------------------------------------------------------
    # Remove rows containing NaN
    # --------------------------------------------------------

    df = df.dropna(
        subset=FEATURE_COLUMNS
    )

    print(
        f"Rows      : {len(df)}"
    )

    print(
        f"Start     : {df.index.min()}"
    )

    print(
        f"End       : {df.index.max()}"
    )

    print(
        f"Columns   : {df.columns.tolist()}"
    )

    return df


# ============================================================
# PREPARE ONE 400-DAY WINDOW
# ============================================================

def prepare_context(
    context
):

    x = (
        context[
            FEATURE_COLUMNS
        ]
        .values
        .astype(np.float32)
    )

    # --------------------------------------------------------
    # SAME NORMALIZATION USED DURING TRAINING
    # --------------------------------------------------------

    x_mean = np.mean(
        x,
        axis=0,
    )

    x_std = np.std(
        x,
        axis=0,
    )

    x_norm = (
        x - x_mean
    ) / (
        x_std + 1e-5
    )

    x_norm = np.clip(
        x_norm,
        -CLIP,
        CLIP,
    )

    # --------------------------------------------------------
    # Timestamp
    # --------------------------------------------------------

    timestamps = pd.Series(
        context.index
    ).reset_index(
        drop=True
    )

    stamp_df = calc_time_stamps(
        timestamps
    )

    # --------------------------------------------------------
    # Batch dimension
    # --------------------------------------------------------

    x_tensor = torch.from_numpy(
        x_norm[
            np.newaxis,
            :,
            :
        ]
    ).to(DEVICE)

    stamp_tensor = torch.from_numpy(
        stamp_df.values
        .astype(np.float32)
        [
            np.newaxis,
            :,
            :
        ]
    ).to(DEVICE)

    return (
        x_tensor,
        stamp_tensor,
        x_mean,
        x_std,
    )


# ============================================================
# PREDICT ONE DAY
# ============================================================

def predict_one_day(
    tokenizer,
    model,
    context,
    prediction_date,
):

    (
        x,
        x_stamp,
        x_mean,
        x_std,
    ) = prepare_context(
        context
    )

    # --------------------------------------------------------
    # Future timestamp = ONLY one day
    # --------------------------------------------------------

    y_timestamp = pd.Series(
        [prediction_date]
    )

    y_stamp_df = calc_time_stamps(
        y_timestamp
    )

    y_stamp = torch.from_numpy(
        y_stamp_df.values
        .astype(np.float32)
        [
            np.newaxis,
            :,
            :
        ]
    ).to(DEVICE)

    # --------------------------------------------------------
    # Kronos prediction
    #
    # pred_len = 1
    #
    # Therefore only ONE future trading bar
    # is generated.
    # --------------------------------------------------------

    normalized_prediction = (
        generate_full_forecast(
            tokenizer=tokenizer,
            model=model,
            x=x,
            x_stamp=x_stamp,
            y_stamp=y_stamp,
            max_context=WINDOW_SIZE,
            pred_len=1,
            clip=CLIP,
            temperature=TEMPERATURE,
            top_k=TOP_K,
            top_p=TOP_P,
            sample_count=SAMPLE_COUNT,
            verbose=False,
        )
    )

    # --------------------------------------------------------
    # Shape:
    #
    # (1, 1, 6)
    # --------------------------------------------------------

    normalized_prediction = (
        normalized_prediction
        .reshape(
            1,
            6
        )
    )

    # --------------------------------------------------------
    # DENORMALIZE
    #
    # Use statistics of CURRENT 400-day window.
    # --------------------------------------------------------

    prediction = (
        normalized_prediction[0]
        * (x_std + 1e-5)
        + x_mean
    )

    return prediction


# ============================================================
# CREATE PLOT
# ============================================================

def create_plot(
    df,
    predictions,
):

    fig = go.Figure()

    # --------------------------------------------------------
    # Full actual HCLTECH close
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["close"],
            mode="lines",
            name="Actual Close",
            line=dict(
                width=1.5
            ),
        )
    )

    # --------------------------------------------------------
    # Rolling predictions
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=predictions.index,
            y=predictions["predicted_close"],
            mode="lines",
            name="Rolling LoRA Prediction",
            line=dict(
                width=2
            ),
        )
    )

    # --------------------------------------------------------
    # Prediction start
    # --------------------------------------------------------

    fig.add_vline(
        x=PREDICTION_START,
        line_dash="dash",
        annotation_text="Rolling prediction starts",
        annotation_position="top left",
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(
        title=(
            "HCLTECH - Kronos LoRA "
            "400-Day Rolling Prediction"
        ),
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
    )

    return fig


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 72
    )

    print(
        "HCLTECH KRONOS LoRA"
    )

    print(
        "400-DAY ROLLING-WINDOW PREDICTION"
    )

    print(
        "=" * 72
    )

    print(
        f"Device: {DEVICE}"
    )

    # ========================================================
    # 1. LOAD MODEL
    # ========================================================

    print(
        "\n[1/5] Loading pretrained Kronos..."
    )

    model = (
        Kronos
        .from_pretrained(
            MODEL_PATH
        )
        .to(DEVICE)
    )

    # ========================================================
    # 2. LOAD TOKENIZER
    # ========================================================

    print(
        "[2/5] Loading tokenizer..."
    )

    tokenizer = (
        KronosTokenizer
        .from_pretrained(
            TOKENIZER_PATH
        )
        .to(DEVICE)
    )

    tokenizer.eval()

    # ========================================================
    # 3. INJECT LoRA + LOAD CHECKPOINT
    # ========================================================

    print(
        "[3/5] Loading trained LoRA..."
    )

    model = inject_lora(
        model
    )

    model = model.to(
        DEVICE
    )

    load_lora_weights(
        model
    )

    model.eval()

    # ========================================================
    # 4. LOAD DATA
    # ========================================================

    print(
        "\n[4/5] Loading HCLTECH data..."
    )

    df = load_data()

    # --------------------------------------------------------
    # Find prediction start
    # --------------------------------------------------------

    prediction_dates = df.index[
        df.index >= PREDICTION_START
    ]

    if len(prediction_dates) == 0:

        raise RuntimeError(
            "No data found from "
            f"{PREDICTION_START}"
        )

    first_prediction_date = (
        prediction_dates[0]
    )

    # --------------------------------------------------------
    # Need 400 actual bars before
    # first prediction date
    # --------------------------------------------------------

    first_position = df.index.get_loc(
        first_prediction_date
    )

    if first_position < WINDOW_SIZE:

        raise RuntimeError(
            "Not enough historical data "
            "for the first 400-day window."
        )

    print(
        "\nRolling prediction setup:"
    )

    print(
        f"Window size       : {WINDOW_SIZE} days"
    )

    print(
        f"Prediction start  : {first_prediction_date}"
    )

    print(
        f"Prediction end    : {prediction_dates[-1]}"
    )

    print(
        f"Prediction days   : {len(prediction_dates)}"
    )

    # ========================================================
    # ROLLING PREDICTION
    # ========================================================

    print(
        "\n[5/5] Starting rolling prediction..."
    )

    prediction_rows = []

    total = len(
        prediction_dates
    )

    for i, prediction_date in enumerate(
        prediction_dates
    ):

        # ----------------------------------------------------
        # Position of prediction date
        # ----------------------------------------------------

        position = df.index.get_loc(
            prediction_date
        )

        # ----------------------------------------------------
        # EXACTLY PREVIOUS 400 ACTUAL BARS
        #
        # Prediction date itself is NOT included.
        # ----------------------------------------------------

        context = df.iloc[
            position - WINDOW_SIZE:
            position
        ].copy()

        if len(context) != WINDOW_SIZE:

            raise RuntimeError(
                f"Wrong context length "
                f"at {prediction_date}: "
                f"{len(context)}"
            )

        # ----------------------------------------------------
        # Predict ONE day
        # ----------------------------------------------------

        prediction = predict_one_day(
            tokenizer=tokenizer,
            model=model,
            context=context,
            prediction_date=prediction_date,
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        prediction_rows.append(
            [
                prediction_date,
                prediction[0],
                prediction[1],
                prediction[2],
                prediction[3],
                prediction[4],
                prediction[5],
            ]
        )

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (i + 1) % 25 == 0
            or i == 0
            or i == total - 1
        ):

            print(
                f"[{i + 1:4d}/{total}] "
                f"{prediction_date.date()} "
                f"| context "
                f"{context.index[0].date()} "
                f"-> "
                f"{context.index[-1].date()} "
                f"| predicted close "
                f"{prediction[3]:.2f}"
            )

    # ========================================================
    # PREDICTION DATAFRAME
    # ========================================================

    predictions = pd.DataFrame(
        prediction_rows,
        columns=[
            "Date",
            "predicted_open",
            "predicted_high",
            "predicted_low",
            "predicted_close",
            "predicted_volume",
            "predicted_amount",
        ],
    )

    predictions["Date"] = pd.to_datetime(
        predictions["Date"]
    )

    predictions = predictions.set_index(
        "Date"
    )

    # ========================================================
    # ADD ACTUAL VALUES
    # ========================================================

    comparison = predictions.copy()

    actual = df.loc[
        predictions.index
    ]

    comparison[
        "actual_open"
    ] = actual["open"]

    comparison[
        "actual_high"
    ] = actual["high"]

    comparison[
        "actual_low"
    ] = actual["low"]

    comparison[
        "actual_close"
    ] = actual["close"]

    comparison[
        "actual_volume"
    ] = actual["volume"]

    # ========================================================
    # REORDER COLUMNS
    # ========================================================

    comparison = comparison[
        [
            "actual_open",
            "actual_high",
            "actual_low",
            "actual_close",
            "actual_volume",
            "predicted_open",
            "predicted_high",
            "predicted_low",
            "predicted_close",
            "predicted_volume",
            "predicted_amount",
        ]
    ]

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    comparison.to_parquet(
        FORECAST_PATH
    )

    print(
        "\nResults saved:"
    )

    print(
        FORECAST_PATH
    )

    print(
        "\nRows:",
        len(comparison)
    )

    print(
        "\nFirst 3 predictions:"
    )

    print(
        comparison.head(3).to_string()
    )

    print(
        "\nLast 3 predictions:"
    )

    print(
        comparison.tail(3).to_string()
    )

    # ========================================================
    # PLOT
    # ========================================================

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    fig = create_plot(
        df=df,
        predictions=predictions,
    )

    fig.write_html(
        PLOT_PATH,
        include_plotlyjs=True,
    )

    print(
        "\nPlot saved:"
    )

    print(
        PLOT_PATH
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "ROLLING PREDICTION COMPLETE"
    )

    print(
        "=" * 72
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
