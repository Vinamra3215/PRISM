import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "HCLTECH_clean.parquet"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

PLOTS_DIR = os.path.join(
    BASE_DIR,
    "plots"
)

PLOT_PATH = os.path.join(
    PLOTS_DIR,
    "HCLTECH_40day_clear.html"
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

CUTOFF_DATE = "2022-01-01"

# EXACTLY 400 TRADING DAYS OF HISTORY
HISTORY_DAYS = 400


# ============================================================
# LOAD ORIGINAL HCLTECH DATA
# ============================================================

def load_data():

    print("\nLoading HCLTECH data:")
    print(DATA_PATH)

    df = pd.read_parquet(DATA_PATH)

    # --------------------------------------------------------
    # Handle MultiIndex columns
    # --------------------------------------------------------

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --------------------------------------------------------
    # Lowercase column names
    # --------------------------------------------------------

    df.columns = [
        str(c).lower()
        for c in df.columns
    ]

    # --------------------------------------------------------
    # Datetime index
    # --------------------------------------------------------

    df.index = pd.to_datetime(df.index)

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = df.sort_index()

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    if "close" not in df.columns:
        raise RuntimeError(
            "HCLTECH dataset does not contain "
            "'close' column."
        )

    return df


# ============================================================
# FIND FORECAST RESULT
# ============================================================

def find_forecast_file():

    parquet_files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "*.parquet"
        )
    )

    if not parquet_files:

        raise FileNotFoundError(
            "\nNo parquet forecast file found in:\n"
            + RESULTS_DIR
            + "\n\nRun the prediction first."
        )

    # Prefer files containing forecast/prediction
    preferred = [
        f for f in parquet_files
        if (
            "forecast" in os.path.basename(f).lower()
            or "predict" in os.path.basename(f).lower()
            or "rolling" in os.path.basename(f).lower()
        )
    ]

    if preferred:
        return preferred[0]

    return parquet_files[0]


# ============================================================
# LOAD FORECAST
# ============================================================

def load_forecast():

    forecast_path = find_forecast_file()

    print("\nLoading forecast result:")
    print(forecast_path)

    forecast = pd.read_parquet(
        forecast_path
    )

    # --------------------------------------------------------
    # Handle MultiIndex
    # --------------------------------------------------------

    if isinstance(
        forecast.columns,
        pd.MultiIndex
    ):
        forecast.columns = (
            forecast.columns
            .get_level_values(0)
        )

    forecast.columns = [
        str(c).lower()
        for c in forecast.columns
    ]

    forecast.index = pd.to_datetime(
        forecast.index
    )

    forecast = forecast.sort_index()

    required = [
        "actual_close",
        "predicted_close"
    ]

    missing = [
        c for c in required
        if c not in forecast.columns
    ]

    if missing:

        raise RuntimeError(
            "Forecast file is missing columns: "
            + str(missing)
        )

    return forecast


# ============================================================
# BUILD 400-DAY HISTORY
# ============================================================

def build_history(
    df,
    prediction_start
):

    # --------------------------------------------------------
    # Everything strictly before prediction
    # --------------------------------------------------------

    before_prediction = df[
        df.index < prediction_start
    ].copy()

    if len(before_prediction) < HISTORY_DAYS:

        raise RuntimeError(
            f"Only {len(before_prediction)} "
            f"historical bars are available before "
            f"{prediction_start}.\n"
            f"Need exactly {HISTORY_DAYS}."
        )

    # --------------------------------------------------------
    # EXACTLY LAST 400 TRADING DAYS
    # --------------------------------------------------------

    history = (
        before_prediction
        .tail(HISTORY_DAYS)
        .copy()
    )

    return history


# ============================================================
# CREATE INTERACTIVE GRAPH
# ============================================================

def create_graph(
    history,
    forecast
):

    prediction_start = forecast.index[0]

    # ========================================================
    # BASELINE
    # ========================================================

    # Baseline = mean close price of the
    # 400-day historical context.

    baseline = history["close"].mean()

    # ========================================================
    # DIVERGENCE
    # ========================================================

    # Positive error:
    # prediction is below actual

    # Negative error:
    # prediction is above actual

    forecast["divergence"] = (
        forecast["actual_close"]
        - forecast["predicted_close"]
    )

    # ========================================================
    # CREATE TWO-PANEL FIGURE
    # ========================================================

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[
            0.72,
            0.28
        ],
        subplot_titles=(
            "HCLTECH — 400-Day History + 40-Day Rolling LoRA Prediction",
            "Prediction Divergence (Actual − Predicted)"
        )
    )

    # ========================================================
    # 400-DAY HISTORICAL CLOSE
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history["close"],
            mode="lines",
            name="400-Day Historical Close",
            line=dict(
                width=2
            ),
            hovertemplate=(
                "<b>Date:</b> %{x|%d-%b-%Y}"
                "<br><b>Close:</b> %{y:.2f}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    # ========================================================
    # ACTUAL FUTURE
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["actual_close"],
            mode="lines",
            name="Actual Close",
            line=dict(
                width=2
            ),
            hovertemplate=(
                "<b>Date:</b> %{x|%d-%b-%Y}"
                "<br><b>Actual:</b> %{y:.2f}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    # ========================================================
    # ROLLING LoRA PREDICTION
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["predicted_close"],
            mode="lines",
            name="40-Day Rolling LoRA Prediction",
            line=dict(
                width=2
            ),
            hovertemplate=(
                "<b>Date:</b> %{x|%d-%b-%Y}"
                "<br><b>Prediction:</b> %{y:.2f}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    # ========================================================
    # BASELINE
    # ========================================================

    fig.add_hline(
        y=baseline,
        line_dash="dot",
        line_width=2,
        annotation_text=(
            f"400-Day Baseline = {baseline:.2f}"
        ),
        annotation_position="top left",
        row=1,
        col=1
    )

    # ========================================================
    # FORECAST START LINE
    # ========================================================

    fig.add_vline(
        x=prediction_start,
        line_dash="dash",
        line_width=2,
        annotation_text="Prediction starts",
        annotation_position="top right",
        row=1,
        col=1
    )

    # ========================================================
    # DIVERGENCE
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=forecast.index,
            y=forecast["divergence"],
            mode="lines",
            name="Prediction Divergence",
            line=dict(
                width=1.5
            ),
            hovertemplate=(
                "<b>Date:</b> %{x|%d-%b-%Y}"
                "<br><b>Error:</b> %{y:.2f}"
                "<extra></extra>"
            )
        ),
        row=2,
        col=1
    )

    # ========================================================
    # ZERO ERROR LINE
    # ========================================================

    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1.5,
        row=2,
        col=1
    )

    # ========================================================
    # FORECAST START LINE ON DIVERGENCE
    # ========================================================

    fig.add_vline(
        x=prediction_start,
        line_dash="dash",
        line_width=1.5,
        row=2,
        col=1
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(

        title=(
            "HCLTECH — Kronos + LoRA | "
            "400-Day Context + 40-Day Rolling Prediction"
        ),

        hovermode="x unified",

        template="plotly_white",

        height=900,

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        ),

        margin=dict(
            l=70,
            r=40,
            t=110,
            b=70
        ),

        # ----------------------------------------------------
        # IMPORTANT:
        # Plotly interactive toolbar
        # ----------------------------------------------------

        dragmode="zoom"
    )

    # ========================================================
    # AXIS LABELS
    # ========================================================

    fig.update_yaxes(
        title_text="HCLTECH Price",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="Divergence",
        row=2,
        col=1
    )

    fig.update_xaxes(
        title_text="Date",
        row=2,
        col=1
    )

    # ========================================================
    # RANGE SLIDER
    # ========================================================

    fig.update_xaxes(
        rangeslider=dict(
            visible=True,
            thickness=0.08
        ),
        row=2,
        col=1
    )

    # ========================================================
    # RANGE SELECTOR
    # ========================================================

    fig.update_xaxes(
        rangeselector=dict(
            buttons=[
                dict(
                    count=30,
                    label="30D",
                    step="day",
                    stepmode="backward"
                ),
                dict(
                    count=90,
                    label="90D",
                    step="day",
                    stepmode="backward"
                ),
                dict(
                    count=180,
                    label="180D",
                    step="day",
                    stepmode="backward"
                ),
                dict(
                    count=1,
                    label="1Y",
                    step="year",
                    stepmode="backward"
                ),
                dict(
                    step="all",
                    label="ALL"
                )
            ]
        ),
        row=2,
        col=1
    )

    # ========================================================
    # SAVE
    # ========================================================

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True
    )

    fig.write_html(
        PLOT_PATH,
        include_plotlyjs=True
    )

    return fig


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("HCLTECH — 400-DAY HISTORY + 40-DAY ROLLING LoRA")
    print("=" * 72)

    # --------------------------------------------------------
    # Load original data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # Load prediction
    # --------------------------------------------------------

    forecast = load_forecast()

    # --------------------------------------------------------
    # Prediction starts here
    # --------------------------------------------------------

    prediction_start = forecast.index[0]

    print(
        "\nPrediction starts:"
    )

    print(
        prediction_start
    )

    # --------------------------------------------------------
    # Build EXACTLY 400 historical days
    # --------------------------------------------------------

    history = build_history(
        df,
        prediction_start
    )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    print("\n" + "-" * 72)
    print("400-DAY HISTORICAL CONTEXT")
    print("-" * 72)

    print(
        "Start :",
        history.index[0]
    )

    print(
        "End   :",
        history.index[-1]
    )

    print(
        "Bars  :",
        len(history)
    )

    print("\n" + "-" * 72)
    print("ROLLING PREDICTION")
    print("-" * 72)

    print(
        "Start :",
        forecast.index[0]
    )

    print(
        "End   :",
        forecast.index[-1]
    )

    print(
        "Bars  :",
        len(forecast)
    )

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    create_graph(
        history,
        forecast
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("GRAPH CREATED SUCCESSFULLY")
    print("=" * 72)

    print(
        "\nGraph saved at:"
    )

    print(
        PLOT_PATH
    )

    print("\nGraph contains:")
    print("  ✓ Exactly 400 historical trading days")
    print("  ✓ Actual future price")
    print("  ✓ 40-day rolling LoRA prediction")
    print("  ✓ 400-day baseline")
    print("  ✓ Prediction divergence")
    print("  ✓ Interactive zoom")
    print("  ✓ Zoom out")
    print("  ✓ Pan")
    print("  ✓ Hover values")
    print("  ✓ Range slider")
    print("  ✓ 30D / 90D / 180D / 1Y / ALL buttons")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
