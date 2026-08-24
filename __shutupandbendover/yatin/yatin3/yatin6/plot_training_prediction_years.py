import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "HCLTECH_clean.parquet",
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results",
)

PLOTS_DIR = os.path.join(
    BASE_DIR,
    "plots",
)

OUTPUT_PATH = os.path.join(
    PLOTS_DIR,
    "HCLTECH_training_prediction_years.html",
)


# ------------------------------------------------------------
# Experiment settings
# ------------------------------------------------------------

TRAINING_DAYS = 400

PREDICTION_START = pd.Timestamp(
    "2022-01-01"
)

PREDICTION_END = pd.Timestamp(
    "2026-12-31"
)


# ============================================================
# LOAD HCLTECH DATA
# ============================================================

def load_data():

    print()
    print("=" * 70)
    print("LOADING HCLTECH DATA")
    print("=" * 70)

    print()
    print("Data path:")
    print(DATA_PATH)

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"\nHCLTECH data not found:\n{DATA_PATH}"
        )

    df = pd.read_parquet(
        DATA_PATH
    )

    # --------------------------------------------------------
    # Convert index to datetime
    # --------------------------------------------------------

    df.index = pd.to_datetime(
        df.index
    )

    # --------------------------------------------------------
    # Handle MultiIndex columns
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
    # Lowercase column names
    # --------------------------------------------------------

    df.columns = [
        str(c).lower()
        for c in df.columns
    ]

    # --------------------------------------------------------
    # Remove duplicate timestamps
    # --------------------------------------------------------

    if df.index.has_duplicates:

        print(
            "Removing duplicate timestamps..."
        )

        df = df[
            ~df.index.duplicated(
                keep="last"
            )
        ]

    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    df = df.sort_index()

    # --------------------------------------------------------
    # Check close column
    # --------------------------------------------------------

    if "close" not in df.columns:

        raise RuntimeError(
            "The HCLTECH data does not contain "
            "a 'close' column."
        )

    print()
    print(
        f"Total bars in dataset: {len(df)}"
    )

    print(
        f"Dataset start: {df.index[0]}"
    )

    print(
        f"Dataset end:   {df.index[-1]}"
    )

    return df


# ============================================================
# FIND PREDICTION RESULT
# ============================================================

def find_prediction_file():

    print()
    print("=" * 70)
    print("SEARCHING FOR PREDICTION RESULT")
    print("=" * 70)

    if not os.path.exists(
        RESULTS_DIR
    ):

        raise FileNotFoundError(
            f"\nResults directory not found:\n"
            f"{RESULTS_DIR}"
        )

    files = os.listdir(
        RESULTS_DIR
    )

    parquet_files = [
        f
        for f in files
        if f.lower().endswith(
            ".parquet"
        )
    ]

    if not parquet_files:

        raise FileNotFoundError(
            "\nNo parquet prediction file "
            "was found inside:\n"
            + RESULTS_DIR
        )

    print()
    print(
        "Parquet files found:"
    )

    for file in parquet_files:

        print(
            "  ",
            file
        )

    # --------------------------------------------------------
    # Prefer rolling prediction result
    # --------------------------------------------------------

    preferred = [
        f
        for f in parquet_files
        if "rolling" in f.lower()
    ]

    if preferred:

        prediction_file = preferred[0]

    else:

        prediction_file = parquet_files[0]

    prediction_path = os.path.join(
        RESULTS_DIR,
        prediction_file,
    )

    print()
    print(
        "Using prediction file:"
    )

    print(
        prediction_path
    )

    return prediction_path


# ============================================================
# LOAD PREDICTION RESULT
# ============================================================

def load_predictions(
    prediction_path
):

    prediction = pd.read_parquet(
        prediction_path
    )

    # --------------------------------------------------------
    # Datetime index
    # --------------------------------------------------------

    prediction.index = pd.to_datetime(
        prediction.index
    )

    # --------------------------------------------------------
    # MultiIndex columns
    # --------------------------------------------------------

    if isinstance(
        prediction.columns,
        pd.MultiIndex,
    ):

        prediction.columns = (
            prediction.columns
            .get_level_values(0)
        )

    # --------------------------------------------------------
    # Lowercase columns
    # --------------------------------------------------------

    prediction.columns = [
        str(c).lower()
        for c in prediction.columns
    ]

    prediction = prediction.sort_index()

    print()
    print(
        "Prediction columns:"
    )

    print(
        list(prediction.columns)
    )

    print()
    print(
        f"Prediction bars: "
        f"{len(prediction)}"
    )

    print(
        f"Prediction start: "
        f"{prediction.index[0]}"
    )

    print(
        f"Prediction end: "
        f"{prediction.index[-1]}"
    )

    return prediction


# ============================================================
# IDENTIFY ACTUAL AND PREDICTED CLOSE
# ============================================================

def get_close_columns(
    prediction
):

    columns = list(
        prediction.columns
    )

    # --------------------------------------------------------
    # Actual close
    # --------------------------------------------------------

    actual_candidates = [
        "actual_close",
        "close_actual",
        "actual",
        "close",
    ]

    actual_column = None

    for name in actual_candidates:

        if name in columns:

            actual_column = name

            break

    # --------------------------------------------------------
    # Predicted close
    # --------------------------------------------------------

    predicted_candidates = [
        "predicted_close",
        "prediction_close",
        "pred_close",
        "rolling_prediction",
        "predicted",
    ]

    predicted_column = None

    for name in predicted_candidates:

        if name in columns:

            predicted_column = name

            break

    # --------------------------------------------------------
    # More general search
    # --------------------------------------------------------

    if actual_column is None:

        actual_matches = [
            c
            for c in columns
            if (
                "actual" in c
                and "close" in c
            )
        ]

        if actual_matches:

            actual_column = (
                actual_matches[0]
            )

    if predicted_column is None:

        predicted_matches = [
            c
            for c in columns
            if (
                (
                    "pred" in c
                    or "forecast" in c
                )
                and "close" in c
            )
        ]

        if predicted_matches:

            predicted_column = (
                predicted_matches[0]
            )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if actual_column is None:

        raise RuntimeError(
            "\nCould not identify actual close "
            "column.\n"
            "Available columns:\n"
            + str(columns)
        )

    if predicted_column is None:

        raise RuntimeError(
            "\nCould not identify predicted close "
            "column.\n"
            "Available columns:\n"
            + str(columns)
        )

    print()
    print(
        "Actual close column:   ",
        actual_column,
    )

    print(
        "Predicted close column:",
        predicted_column,
    )

    return (
        actual_column,
        predicted_column,
    )


# ============================================================
# BUILD 400-DAY TRAINING CONTEXT
# ============================================================

def build_training_context(
    df,
    prediction,
):

    print()
    print("=" * 70)
    print("BUILDING 400-DAY TRAINING CONTEXT")
    print("=" * 70)

    # --------------------------------------------------------
    # Determine prediction start
    # --------------------------------------------------------

    prediction_start = max(
        prediction.index.min(),
        PREDICTION_START,
    )

    # --------------------------------------------------------
    # Historical data strictly before prediction
    # --------------------------------------------------------

    historical = df[
        df.index < prediction_start
    ].copy()

    print()
    print(
        "Prediction starts:",
        prediction_start,
    )

    print(
        "Historical bars available:",
        len(historical),
    )

    if len(historical) < TRAINING_DAYS:

        raise RuntimeError(
            f"\nOnly {len(historical)} "
            f"historical bars are available before "
            f"{prediction_start.date()}, but "
            f"{TRAINING_DAYS} are required."
        )

    # --------------------------------------------------------
    # EXACTLY last 400 trading days
    # --------------------------------------------------------

    training_context = (
        historical
        .tail(TRAINING_DAYS)
        .copy()
    )

    print()
    print(
        "400-DAY TRAINING CONTEXT"
    )

    print(
        "Start:",
        training_context.index[0],
    )

    print(
        "End:",
        training_context.index[-1],
    )

    print(
        "Bars:",
        len(training_context),
    )

    return (
        training_context,
        prediction_start,
    )


# ============================================================
# BUILD PREDICTION PERIOD
# ============================================================

def build_prediction_period(
    df,
    prediction,
    actual_column,
    predicted_column,
):

    # --------------------------------------------------------
    # Restrict prediction period
    # --------------------------------------------------------

    prediction_period = prediction[
        (
            prediction.index
            >= PREDICTION_START
        )
        &
        (
            prediction.index
            <= PREDICTION_END
        )
    ].copy()

    if prediction_period.empty:

        raise RuntimeError(
            "\nNo prediction data found "
            "between 2022 and 2026."
        )

    # --------------------------------------------------------
    # Actual close
    # --------------------------------------------------------

    actual_close = (
        prediction_period[
            actual_column
        ]
        .astype(float)
    )

    # --------------------------------------------------------
    # Predicted close
    # --------------------------------------------------------

    predicted_close = (
        prediction_period[
            predicted_column
        ]
        .astype(float)
    )

    # --------------------------------------------------------
    # Remove NaN rows
    # --------------------------------------------------------

    valid = (
        actual_close.notna()
        &
        predicted_close.notna()
    )

    actual_close = (
        actual_close[valid]
    )

    predicted_close = (
        predicted_close[valid]
    )

    # --------------------------------------------------------
    # Divergence
    #
    # Actual - Prediction
    #
    # Positive:
    # model predicted below actual
    #
    # Negative:
    # model predicted above actual
    # --------------------------------------------------------

    divergence = (
        actual_close
        - predicted_close
    )

    result = pd.DataFrame(
        {
            "actual_close":
                actual_close,

            "predicted_close":
                predicted_close,

            "divergence":
                divergence,
        }
    )

    return result


# ============================================================
# CREATE GRAPH
# ============================================================

def create_graph(
    training_context,
    prediction_period,
):

    print()
    print("=" * 70)
    print("CREATING GRAPH")
    print("=" * 70)

    # --------------------------------------------------------
    # Baseline
    #
    # Baseline = first actual close of prediction period.
    #
    # This gives a simple reference level for comparing
    # the predicted/actual movement after prediction starts.
    # --------------------------------------------------------

    baseline = (
        prediction_period[
            "actual_close"
        ].iloc[0]
    )

    # --------------------------------------------------------
    # Create two-panel figure
    # --------------------------------------------------------

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[
            0.72,
            0.28,
        ],
        subplot_titles=(
            "HCLTECH — 400-Day Training Context + 2022–2026 Prediction",
            "Prediction Divergence: Actual − Prediction",
        ),
    )

    # ========================================================
    # TOP GRAPH
    # ========================================================

    # --------------------------------------------------------
    # 400-day training close
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=training_context.index,
            y=training_context["close"],
            mode="lines",
            name="400-Day Training Close",
            line=dict(
                width=2,
            ),
            hovertemplate=(
                "Date: %{x|%d %b %Y}"
                "<br>"
                "Training Close: %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # Actual future
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=prediction_period.index,
            y=prediction_period[
                "actual_close"
            ],
            mode="lines",
            name="Actual Close",
            line=dict(
                width=2,
            ),
            hovertemplate=(
                "Date: %{x|%d %b %Y}"
                "<br>"
                "Actual: %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # Rolling LoRA prediction
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=prediction_period.index,
            y=prediction_period[
                "predicted_close"
            ],
            mode="lines",
            name="Rolling LoRA Prediction",
            line=dict(
                width=2,
            ),
            hovertemplate=(
                "Date: %{x|%d %b %Y}"
                "<br>"
                "Prediction: %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    fig.add_hline(
        y=baseline,
        row=1,
        col=1,
        line_dash="dot",
        line_width=1.5,
        annotation_text=(
            f"Baseline = {baseline:.2f}"
        ),
        annotation_position="top right",
    )

    # --------------------------------------------------------
    # Prediction start
    # --------------------------------------------------------

    prediction_start = (
        prediction_period.index[0]
    )

    fig.add_vline(
        x=prediction_start,
        row=1,
        col=1,
        line_dash="dash",
        line_width=1.5,
        annotation_text=(
            "Prediction starts"
        ),
        annotation_position="top left",
    )

    # ========================================================
    # BOTTOM GRAPH — DIVERGENCE
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=prediction_period.index,
            y=prediction_period[
                "divergence"
            ],
            mode="lines",
            name="Divergence",
            line=dict(
                width=1.5,
            ),
            hovertemplate=(
                "Date: %{x|%d %b %Y}"
                "<br>"
                "Divergence: %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # --------------------------------------------------------
    # Zero divergence line
    # --------------------------------------------------------

    fig.add_hline(
        y=0,
        row=2,
        col=1,
        line_dash="dot",
        line_width=1,
    )

    # ========================================================
    # AXES
    # ========================================================

    # --------------------------------------------------------
    # Top y-axis
    # --------------------------------------------------------

    fig.update_yaxes(
        title_text="HCLTECH Price",
        row=1,
        col=1,
        showgrid=True,
        zeroline=False,
    )

    # --------------------------------------------------------
    # Divergence y-axis
    # --------------------------------------------------------

    fig.update_yaxes(
        title_text="Divergence",
        row=2,
        col=1,
        showgrid=True,
        zeroline=True,
    )

    # --------------------------------------------------------
    # X-axis
    #
    # Year ticks only.
    # --------------------------------------------------------

    fig.update_xaxes(
        title_text="Year",
        row=2,
        col=1,
        dtick="M12",
        tickformat="%Y",
        showgrid=True,
    )

    fig.update_xaxes(
        row=1,
        col=1,
        dtick="M12",
        tickformat="%Y",
        showgrid=True,
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(
        title=(
            "HCLTECH — Kronos + LoRA | "
            "400-Day Training Context → "
            "2022–2026 Prediction"
        ),

        template="plotly_white",

        height=850,

        hovermode="x unified",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),

        margin=dict(
            l=80,
            r=40,
            t=110,
            b=70,
        ),

        # ----------------------------------------------------
        # IMPORTANT:
        # Remove Plotly zoom/toolbar panel.
        # ----------------------------------------------------

        modebar_remove=[
            "zoom2d",
            "pan2d",
            "select2d",
            "lasso2d",
            "zoomIn2d",
            "zoomOut2d",
            "autoScale2d",
            "resetScale2d",
        ],
    )

    # --------------------------------------------------------
    # Save graph
    # --------------------------------------------------------

    os.makedirs(
        PLOTS_DIR,
        exist_ok=True,
    )

    fig.write_html(
        OUTPUT_PATH,
        include_plotlyjs=True,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
        },
    )

    print()
    print(
        "Graph saved:"
    )

    print(
        OUTPUT_PATH
    )

    return fig


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 75)
    print(
        "HCLTECH — 400-DAY TRAINING + "
        "2022–2026 PREDICTION GRAPH"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # 1. Load original HCLTECH data
    # --------------------------------------------------------

    df = load_data()

    # --------------------------------------------------------
    # 2. Find prediction result
    # --------------------------------------------------------

    prediction_path = (
        find_prediction_file()
    )

    # --------------------------------------------------------
    # 3. Load prediction result
    # --------------------------------------------------------

    prediction = (
        load_predictions(
            prediction_path
        )
    )

    # --------------------------------------------------------
    # 4. Identify columns
    # --------------------------------------------------------

    (
        actual_column,
        predicted_column,
    ) = get_close_columns(
        prediction
    )

    # --------------------------------------------------------
    # 5. Build exactly 400-day training context
    # --------------------------------------------------------

    (
        training_context,
        prediction_start,
    ) = build_training_context(
        df,
        prediction,
    )

    # --------------------------------------------------------
    # 6. Build 2022–2026 prediction period
    # --------------------------------------------------------

    prediction_period = (
        build_prediction_period(
            df,
            prediction,
            actual_column,
            predicted_column,
        )
    )

    # --------------------------------------------------------
    # 7. Print experiment information
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL GRAPH PERIODS")
    print("=" * 70)

    print()
    print("TRAINING CONTEXT")
    print(
        "Start:",
        training_context.index[0],
    )
    print(
        "End:  ",
        training_context.index[-1],
    )
    print(
        "Bars: ",
        len(training_context),
    )

    print()
    print("PREDICTION PERIOD")
    print(
        "Start:",
        prediction_period.index[0],
    )
    print(
        "End:  ",
        prediction_period.index[-1],
    )
    print(
        "Bars: ",
        len(prediction_period),
    )

    # --------------------------------------------------------
    # 8. Create graph
    # --------------------------------------------------------

    create_graph(
        training_context,
        prediction_period,
    )

    # --------------------------------------------------------
    # 9. Complete
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PLOT CREATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(
        "Open:"
    )

    print(
        OUTPUT_PATH
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()