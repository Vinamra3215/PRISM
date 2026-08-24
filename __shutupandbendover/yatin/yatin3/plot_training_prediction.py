import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "HCLTECH_clean.parquet"
)

PREDICTION_PATH = os.path.join(
    BASE_DIR,
    "results",
    "HCLTECH_rolling_predictions.parquet"
)

PLOTS_DIR = os.path.join(
    BASE_DIR,
    "plots"
)

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)

PLOT_PATH = os.path.join(
    PLOTS_DIR,
    "HCLTECH_training_prediction_2022_2026.html"
)


# ============================================================
# SETTINGS
# ============================================================

TRAINING_WINDOW = 400

PREDICTION_START = pd.Timestamp(
    "2022-01-01"
)

PREDICTION_END = pd.Timestamp(
    "2026-12-31"
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

def load_historical_data():

    print("\nLoading historical HCLTECH data...")

    print(DATA_PATH)

    df = pd.read_parquet(
        DATA_PATH
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
        pd.MultiIndex
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
    # Remove duplicate timestamps
    # --------------------------------------------------------

    df = df[
        ~df.index.duplicated(
            keep="last"
        )
    ]

    print("\nHistorical columns:")

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# LOAD ROLLING PREDICTIONS
# ============================================================

def load_prediction_data():

    print("\nLoading rolling prediction data...")

    print(PREDICTION_PATH)

    df = pd.read_parquet(
        PREDICTION_PATH
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
        pd.MultiIndex
    ):

        df.columns = [
            "_".join(
                str(x)
                for x in column
                if str(x) != "nan"
            ).lower()
            for column in df.columns
        ]

    else:

        df.columns = [
            str(c).lower()
            for c in df.columns
        ]

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

    print("\nPrediction columns:")

    print(
        df.columns.tolist()
    )

    return df


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    df,
    possible_names
):

    for name in possible_names:

        if name in df.columns:

            return name

    return None


# ============================================================
# PREPARE PREDICTION DATA
# ============================================================

def prepare_prediction_data(
    prediction_df
):

    # --------------------------------------------------------
    # Actual close
    # --------------------------------------------------------

    actual_col = find_column(
        prediction_df,
        [
            "actual_close",
            "actual",
            "close_actual",
            "close"
        ]
    )

    # --------------------------------------------------------
    # Predicted close
    # --------------------------------------------------------

    predicted_col = find_column(
        prediction_df,
        [
            "predicted_close",
            "prediction_close",
            "pred_close",
            "rolling_prediction",
            "prediction"
        ]
    )

    if actual_col is None:

        raise RuntimeError(
            "Could not find ACTUAL CLOSE "
            "column in prediction parquet.\n"
            f"Available columns:\n"
            f"{prediction_df.columns.tolist()}"
        )

    if predicted_col is None:

        raise RuntimeError(
            "Could not find PREDICTED CLOSE "
            "column in prediction parquet.\n"
            f"Available columns:\n"
            f"{prediction_df.columns.tolist()}"
        )

    print(
        "\nActual column:",
        actual_col
    )

    print(
        "Predicted column:",
        predicted_col
    )

    result = pd.DataFrame(
        index=prediction_df.index
    )

    result["actual"] = pd.to_numeric(
        prediction_df[actual_col],
        errors="coerce"
    )

    result["prediction"] = pd.to_numeric(
        prediction_df[predicted_col],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Remove NaN
    # --------------------------------------------------------

    result = result.dropna(
        subset=[
            "actual",
            "prediction"
        ]
    )

    return result


# ============================================================
# BUILD 400-DAY TRAINING CONTEXT
# ============================================================

def build_training_context(
    historical_df,
    prediction_df
):

    # --------------------------------------------------------
    # Prediction starts on the first available prediction date
    # in the requested 2022-2026 period.
    # --------------------------------------------------------

    prediction_period = prediction_df[
        (prediction_df.index >= PREDICTION_START)
        &
        (prediction_df.index <= PREDICTION_END)
    ].copy()

    if prediction_period.empty:

        raise RuntimeError(
            "No prediction data found "
            "between 2022-01-01 and 2026-12-31."
        )

    prediction_start = (
        prediction_period.index[0]
    )

    # --------------------------------------------------------
    # Historical data strictly BEFORE prediction start
    # --------------------------------------------------------

    historical_before_prediction = (
        historical_df[
            historical_df.index
            < prediction_start
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Need exactly 400 bars
    # --------------------------------------------------------

    if len(
        historical_before_prediction
    ) < TRAINING_WINDOW:

        raise RuntimeError(
            f"Only "
            f"{len(historical_before_prediction)} "
            f"historical bars are available before "
            f"{prediction_start.date()}, "
            f"but {TRAINING_WINDOW} are required."
        )

    training = (
        historical_before_prediction
        .tail(TRAINING_WINDOW)
        .copy()
    )

    # --------------------------------------------------------
    # Make sure close exists
    # --------------------------------------------------------

    if "close" not in training.columns:

        raise RuntimeError(
            "The historical parquet does not contain "
            "'close' column."
        )

    return (
        training,
        prediction_period
    )


# ============================================================
# ADD BASELINE + DIVERGENCE
# ============================================================

def add_metrics(
    training,
    prediction
):

    prediction = prediction.copy()

    # --------------------------------------------------------
    # Baseline
    #
    # Persistence baseline:
    #
    # Today's baseline prediction =
    # previous actual close.
    #
    # For first prediction day, use the last close
    # of the 400-day training context.
    # --------------------------------------------------------

    actual_previous = (
        prediction["actual"]
        .shift(1)
    )

    prediction["baseline"] = (
        actual_previous
    )

    prediction.loc[
        prediction.index[0],
        "baseline"
    ] = training["close"].iloc[-1]

    # --------------------------------------------------------
    # Prediction error
    #
    # Predicted - Actual
    # --------------------------------------------------------

    prediction["error"] = (
        prediction["prediction"]
        - prediction["actual"]
    )

    # --------------------------------------------------------
    # Divergence
    #
    # Actual - Prediction
    # --------------------------------------------------------

    prediction["divergence"] = (
        prediction["actual"]
        - prediction["prediction"]
    )

    return prediction


# ============================================================
# CREATE GRAPH
# ============================================================

def create_graph(
    training,
    prediction
):

    # ========================================================
    # TWO PANELS
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

        subplot_titles=[
            (
                "HCLTECH — 400-Day Training Context "
                "+ 2022–2026 Rolling Prediction"
            ),
            "Prediction Divergence"
        ]
    )

    # ========================================================
    # 1. TRAINING CONTEXT
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=training.index,

            y=training["close"],

            mode="lines",

            name="Training Context (400 days)",

            line=dict(
                width=2
            ),

            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}"
                "<br>"
                "<b>Training Close:</b> %{y:.2f}"
                "<extra></extra>"
            )
        ),

        row=1,

        col=1
    )

    # ========================================================
    # 2. ACTUAL PRICE — PREDICTION PERIOD
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=prediction.index,

            y=prediction["actual"],

            mode="lines",

            name="Actual Close",

            line=dict(
                width=2
            ),

            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}"
                "<br>"
                "<b>Actual:</b> %{y:.2f}"
                "<extra></extra>"
            )
        ),

        row=1,

        col=1
    )

    # ========================================================
    # 3. 40-DAY ROLLING LoRA PREDICTION
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=prediction.index,

            y=prediction["prediction"],

            mode="lines",

            name="40-Day Rolling LoRA Prediction",

            line=dict(
                width=2
            ),

            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}"
                "<br>"
                "<b>Prediction:</b> %{y:.2f}"
                "<extra></extra>"
            )
        ),

        row=1,

        col=1
    )

    # ========================================================
    # 4. BASELINE
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=prediction.index,

            y=prediction["baseline"],

            mode="lines",

            name="Baseline",

            line=dict(
                width=1.5,
                dash="dot"
            ),

            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}"
                "<br>"
                "<b>Baseline:</b> %{y:.2f}"
                "<extra></extra>"
            )
        ),

        row=1,

        col=1
    )

    # ========================================================
    # 5. DIVERGENCE
    # ========================================================

    fig.add_trace(

        go.Scatter(

            x=prediction.index,

            y=prediction["divergence"],

            mode="lines",

            name="Divergence",

            line=dict(
                width=1.5
            ),

            hovertemplate=(
                "<b>Date:</b> %{x|%d %b %Y}"
                "<br>"
                "<b>Divergence:</b> %{y:.2f}"
                "<extra></extra>"
            )
        ),

        row=2,

        col=1
    )

    # ========================================================
    # ZERO LINE FOR DIVERGENCE
    # ========================================================

    fig.add_hline(

        y=0,

        line_dash="dash",

        line_width=1,

        row=2,

        col=1
    )

    # ========================================================
    # PREDICTION START LINE
    # ========================================================

    prediction_start = (
        prediction.index[0]
    )

    fig.add_vline(

        x=prediction_start,

        line_dash="dash",

        line_width=2,

        row=1,

        col=1
    )

    # ========================================================
    # PREDICTION START LABEL
    # ========================================================

    fig.add_annotation(

        x=prediction_start,

        y=1,

        yref="paper",

        text=(
            "Prediction starts: "
            + prediction_start.strftime(
                "%d-%b-%Y"
            )
        ),

        showarrow=False,

        xanchor="left",

        yanchor="bottom"
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    valid = prediction.dropna(
        subset=[
            "actual",
            "prediction"
        ]
    )

    errors = (
        valid["prediction"]
        - valid["actual"]
    )

    mae = np.mean(
        np.abs(errors)
    )

    rmse = np.sqrt(
        np.mean(
            errors ** 2
        )
    )

    correlation = (
        valid["actual"]
        .corr(
            valid["prediction"]
        )
    )

    # ========================================================
    # TITLE
    # ========================================================

    fig.update_layout(

        title=dict(

            text=(
                "HCLTECH — Kronos + LoRA"
                "<br>"
                "<sup>"
                "400-Day Training Context → "
                "2022–2026 Rolling Prediction"
                " &nbsp; | &nbsp; "
                f"MAE: {mae:.2f}"
                " &nbsp; | &nbsp; "
                f"RMSE: {rmse:.2f}"
                " &nbsp; | &nbsp; "
                f"Correlation: {correlation:.3f}"
                "</sup>"
            ),

            x=0.5,

            xanchor="center"
        ),

        height=850,

        template="plotly_white",

        hovermode="x unified",

        # ----------------------------------------------------
        # No zoom
        # ----------------------------------------------------

        dragmode=False,

        # ----------------------------------------------------
        # Legend
        # ----------------------------------------------------

        legend=dict(

            orientation="h",

            yanchor="bottom",

            y=1.02,

            xanchor="left",

            x=0
        )
    )

    # ========================================================
    # X AXIS
    #
    # Show years only
    # ========================================================

    fig.update_xaxes(

        type="date",

        tickformat="%Y",

        dtick="M12",

        showgrid=True,

        row=1,

        col=1
    )

    fig.update_xaxes(

        type="date",

        tickformat="%Y",

        dtick="M12",

        showgrid=True,

        title_text="Year",

        row=2,

        col=1
    )

    # ========================================================
    # Y AXIS
    # ========================================================

    fig.update_yaxes(

        title_text="HCLTECH Price",

        showgrid=True,

        row=1,

        col=1
    )

    fig.update_yaxes(

        title_text="Divergence",

        showgrid=True,

        row=2,

        col=1
    )

    # ========================================================
    # COMPLETELY STATIC GRAPH
    # ========================================================

    config = {

        "staticPlot": True,

        "displayModeBar": False,

        "responsive": True
    }

    return (
        fig,
        config
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 75
    )

    print(
        "HCLTECH — 400-DAY TRAINING + "
        "2022–2026 ROLLING PREDICTION"
    )

    print(
        "=" * 75
    )

    # ========================================================
    # LOAD ORIGINAL HISTORICAL DATA
    # ========================================================

    historical_df = (
        load_historical_data()
    )

    # ========================================================
    # LOAD PREDICTION RESULTS
    # ========================================================

    prediction_raw = (
        load_prediction_data()
    )

    prediction_df = (
        prepare_prediction_data(
            prediction_raw
        )
    )

    # ========================================================
    # BUILD PERIODS
    # ========================================================

    (
        training,
        prediction
    ) = build_training_context(

        historical_df,

        prediction_df
    )

    # ========================================================
    # ADD BASELINE + DIVERGENCE
    # ========================================================

    prediction = add_metrics(

        training,

        prediction
    )

    # ========================================================
    # PRINT TRAINING PERIOD
    # ========================================================

    print(
        "\n" + "-" * 75
    )

    print(
        "400-DAY TRAINING / CONTEXT PERIOD"
    )

    print(
        "-" * 75
    )

    print(
        "Start :",
        training.index[0]
    )

    print(
        "End   :",
        training.index[-1]
    )

    print(
        "Bars  :",
        len(training)
    )

    # ========================================================
    # PRINT PREDICTION PERIOD
    # ========================================================

    print(
        "\n" + "-" * 75
    )

    print(
        "PREDICTION PERIOD"
    )

    print(
        "-" * 75
    )

    print(
        "Start :",
        prediction.index[0]
    )

    print(
        "End   :",
        prediction.index[-1]
    )

    print(
        "Bars  :",
        len(prediction)
    )

    # ========================================================
    # CREATE GRAPH
    # ========================================================

    fig, config = create_graph(

        training,

        prediction
    )

    # ========================================================
    # SAVE
    # ========================================================

    fig.write_html(

        PLOT_PATH,

        include_plotlyjs=True,

        config=config
    )

    # ========================================================
    # FINAL
    # ========================================================

    print(
        "\n" + "=" * 75
    )

    print(
        "GRAPH CREATED SUCCESSFULLY"
    )

    print(
        "=" * 75
    )

    print(
        "\nSaved at:"
    )

    print(
        PLOT_PATH
    )

    print(
        "\nGraph contains:"
    )

    print(
        "  • Exactly 400 historical training/context bars"
    )

    print(
        "  • 2022-01-01 to 2026 prediction period"
    )

    print(
        "  • Actual Close"
    )

    print(
        "  • 40-Day Rolling LoRA Prediction"
    )

    print(
        "  • Baseline"
    )

    print(
        "  • Divergence"
    )

    print(
        "  • Year-based x-axis"
    )

    print(
        "  • No zoom controls"
    )

    print(
        "  • No range slider"
    )

    print(
        "  • No mode bar"
    )

    print(
        "=" * 75
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()