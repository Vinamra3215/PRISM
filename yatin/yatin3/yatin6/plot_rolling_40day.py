import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# SETTINGS
# ============================================================

ROLLING_WINDOW = 40

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(PLOTS_DIR, exist_ok=True)

PLOT_PATH = os.path.join(
    PLOTS_DIR,
    "HCLTECH_40day_interactive.html"
)


# ============================================================
# FIND RESULT FILE
# ============================================================

def find_result_file():

    files = glob.glob(
        os.path.join(
            RESULTS_DIR,
            "*.parquet"
        )
    )

    if not files:
        raise FileNotFoundError(
            f"No parquet result file found in:\n{RESULTS_DIR}"
        )

    # Prefer files containing forecast/prediction
    preferred = [
        f for f in files
        if any(
            word in os.path.basename(f).lower()
            for word in [
                "forecast",
                "prediction",
                "rolling",
                "result",
            ]
        )
    ]

    if preferred:
        return preferred[0]

    return files[0]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    result_path = find_result_file()

    print("=" * 70)
    print("LOADING RESULT")
    print("=" * 70)

    print(result_path)

    df = pd.read_parquet(result_path)

    # --------------------------------------------------------
    # Datetime index
    # --------------------------------------------------------

    if not isinstance(
        df.index,
        pd.DatetimeIndex
    ):
        df.index = pd.to_datetime(df.index)

    df = df.sort_index()

    # --------------------------------------------------------
    # Handle MultiIndex columns
    # --------------------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):
        df.columns = [
            "_".join(
                str(x)
                for x in col
                if str(x) != "nan"
            ).lower()
            for col in df.columns
        ]

    else:
        df.columns = [
            str(c).lower()
            for c in df.columns
        ]

    print("\nColumns found:")
    print(df.columns.tolist())

    return df


# ============================================================
# FIND ACTUAL / PREDICTED COLUMNS
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
# PREPARE COLUMNS
# ============================================================

def prepare_data(df):

    # --------------------------------------------------------
    # Actual close
    # --------------------------------------------------------

    actual_col = find_column(
        df,
        [
            "actual_close",
            "close_actual",
            "actual",
            "close",
        ]
    )

    # --------------------------------------------------------
    # Predicted close
    # --------------------------------------------------------

    predicted_col = find_column(
        df,
        [
            "predicted_close",
            "prediction_close",
            "pred_close",
            "rolling_prediction",
            "prediction",
        ]
    )

    if actual_col is None:
        raise RuntimeError(
            "Could not find actual close column."
        )

    if predicted_col is None:
        raise RuntimeError(
            "Could not find predicted close column."
        )

    print("\nUsing columns:")
    print("Actual    :", actual_col)
    print("Prediction:", predicted_col)

    actual = pd.to_numeric(
        df[actual_col],
        errors="coerce"
    )

    prediction = pd.to_numeric(
        df[predicted_col],
        errors="coerce"
    )

    result = pd.DataFrame(
        index=df.index
    )

    result["actual"] = actual
    result["prediction"] = prediction

    result = result.dropna()

    # --------------------------------------------------------
    # Prediction error
    #
    # Positive:
    # prediction is above actual
    #
    # Negative:
    # prediction is below actual
    # --------------------------------------------------------

    result["error"] = (
        result["prediction"]
        - result["actual"]
    )

    # --------------------------------------------------------
    # Baseline
    #
    # Baseline = previous actual closing price.
    #
    # This is a simple persistence baseline:
    # tomorrow's price = today's price.
    # --------------------------------------------------------

    result["baseline"] = (
        result["actual"]
        .shift(1)
    )

    # --------------------------------------------------------
    # Divergence
    #
    # Difference between actual and prediction.
    # --------------------------------------------------------

    result["divergence"] = (
        result["actual"]
        - result["prediction"]
    )

    return result


# ============================================================
# CREATE INTERACTIVE GRAPH
# ============================================================

def create_plot(df):

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[
            0.72,
            0.28,
        ],
        subplot_titles=[
            "HCLTECH — Actual vs 40-Day Rolling LoRA Prediction",
            "Prediction Divergence / Error",
        ],
    )

    # ========================================================
    # MAIN GRAPH
    # ========================================================

    # Actual
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["actual"],
            mode="lines",
            name="Actual Close",
            line=dict(
                width=2,
            ),
            hovertemplate=(
                "<b>Date:</b> %{x}<br>"
                "<b>Actual:</b> %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Rolling prediction
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["prediction"],
            mode="lines",
            name="40-Day Rolling LoRA Prediction",
            line=dict(
                width=1.8,
            ),
            hovertemplate=(
                "<b>Date:</b> %{x}<br>"
                "<b>Prediction:</b> %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # Baseline
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["baseline"],
            mode="lines",
            name="Baseline",
            line=dict(
                width=1.5,
                dash="dot",
            ),
            hovertemplate=(
                "<b>Date:</b> %{x}<br>"
                "<b>Baseline:</b> %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    # ========================================================
    # DIVERGENCE
    # ========================================================

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["divergence"],
            mode="lines",
            name="Divergence",
            line=dict(
                width=1.5,
            ),
            hovertemplate=(
                "<b>Date:</b> %{x}<br>"
                "<b>Divergence:</b> %{y:.2f}"
                "<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    # Zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_width=1,
        row=2,
        col=1,
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    error = df["error"].dropna()

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    correlation = (
        df["actual"]
        .corr(
            df["prediction"]
        )
    )

    # ========================================================
    # LAYOUT
    # ========================================================

    fig.update_layout(

        title=dict(
            text=(
                "HCLTECH — Kronos + LoRA "
                "40-Day Rolling Prediction"
                "<br>"
                "<sup>"
                f"MAE: {mae:.2f} | "
                f"RMSE: {rmse:.2f} | "
                f"Correlation: {correlation:.3f}"
                "</sup>"
            ),
            x=0.5,
        ),

        height=850,

        hovermode="x unified",

        template="plotly_white",

        # ----------------------------------------------------
        # IMPORTANT:
        # Keep useful controls but REMOVE aggressive + / -
        # zoom buttons.
        # ----------------------------------------------------

        dragmode="pan",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),

        # ----------------------------------------------------
        # RANGE SLIDER
        #
        # This gives smooth control over how much of the
        # graph is visible.
        # ----------------------------------------------------

        xaxis=dict(
            title="Date",

            type="date",

            rangeslider=dict(
                visible=True,
                thickness=0.08,
            ),

            rangeselector=dict(
                buttons=[
                    dict(
                        count=6,
                        label="6M",
                        step="month",
                        stepmode="backward",
                    ),
                    dict(
                        count=1,
                        label="1Y",
                        step="year",
                        stepmode="backward",
                    ),
                    dict(
                        count=2,
                        label="2Y",
                        step="year",
                        stepmode="backward",
                    ),
                    dict(
                        count=5,
                        label="5Y",
                        step="year",
                        stepmode="backward",
                    ),
                    dict(
                        step="all",
                        label="ALL",
                    ),
                ]
            ),
        ),

        yaxis=dict(
            title="HCLTECH Price",
        ),

        xaxis2=dict(
            title="Date",
        ),

        yaxis2=dict(
            title="Divergence",
        ),
    )

    # ========================================================
    # INTERACTIVE CONFIGURATION
    # ========================================================

    config = {

        # Mouse wheel gives gradual zooming
        "scrollZoom": True,

        # Don't show Plotly logo
        "displaylogo": False,

        # Remove aggressive + and - zoom buttons
        "modeBarButtonsToRemove": [
            "zoomIn2d",
            "zoomOut2d",
            "lasso2d",
            "select2d",
        ],

        # Responsive graph
        "responsive": True,

        # Useful when opening in browser
        "doubleClick": "reset",
    }

    return fig, config


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HCLTECH 40-DAY ROLLING INTERACTIVE PLOT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    raw_df = load_data()

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    df = prepare_data(
        raw_df
    )

    # --------------------------------------------------------
    # Print information
    # --------------------------------------------------------

    print("\n" + "-" * 70)
    print("DATA INFORMATION")
    print("-" * 70)

    print(
        "Start:",
        df.index[0]
    )

    print(
        "End  :",
        df.index[-1]
    )

    print(
        "Bars :",
        len(df)
    )

    print(
        "Rolling window:",
        ROLLING_WINDOW,
        "days"
    )

    # --------------------------------------------------------
    # Create graph
    # --------------------------------------------------------

    fig, config = create_plot(
        df
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    fig.write_html(
        PLOT_PATH,
        include_plotlyjs=True,
        config=config,
    )

    print("\n" + "=" * 70)
    print("GRAPH CREATED SUCCESSFULLY")
    print("=" * 70)

    print("\nSaved at:")
    print(PLOT_PATH)

    print("\nOpen using:")
    print(
        f"http://localhost:8002/"
        f"{os.path.basename(PLOT_PATH)}"
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
