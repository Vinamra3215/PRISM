import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# CONFIG
# ============================================================

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

OUTPUT_FILE = os.path.join(
    PLOTS_DIR,
    "HCLTECH_rolling_clear.html"
)


# ============================================================
# FIND FORECAST FILE
# ============================================================

parquet_files = glob.glob(
    os.path.join(RESULTS_DIR, "*.parquet")
)

if not parquet_files:
    raise FileNotFoundError(
        "No parquet file found inside results/"
    )


# Prefer a rolling forecast file
rolling_files = [
    f for f in parquet_files
    if "rolling" in os.path.basename(f).lower()
]


if rolling_files:
    FORECAST_FILE = rolling_files[0]
else:
    FORECAST_FILE = parquet_files[0]


print("=" * 70)
print("ROLLING PREDICTION PLOT")
print("=" * 70)

print("Using forecast file:")
print(FORECAST_FILE)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(
    FORECAST_FILE
)

print("\nColumns:")
print(df.columns.tolist())

print("\nRows:", len(df))


# ============================================================
# HANDLE DATE INDEX
# ============================================================

if "Date" in df.columns:

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    df = df.set_index("Date")

else:

    df.index = pd.to_datetime(
        df.index
    )


df = df.sort_index()


# ============================================================
# FIND ACTUAL CLOSE COLUMN
# ============================================================

actual_candidates = [
    "actual_close",
    "Actual Close",
    "actual_Close",
    "close_actual",
]


predicted_candidates = [
    "predicted_close",
    "Predicted Close",
    "predicted_Close",
    "prediction_close",
]


actual_col = None
predicted_col = None


for col in actual_candidates:

    if col in df.columns:
        actual_col = col
        break


for col in predicted_candidates:

    if col in df.columns:
        predicted_col = col
        break


# ------------------------------------------------------------
# If your parquet uses different names, handle them here.
# ------------------------------------------------------------

if actual_col is None:

    if "close" in df.columns:
        actual_col = "close"

    elif "Actual" in df.columns:
        actual_col = "Actual"


if predicted_col is None:

    if "prediction" in df.columns:
        predicted_col = "prediction"

    elif "predicted" in df.columns:
        predicted_col = "predicted"


if actual_col is None:

    raise RuntimeError(
        "Could not find actual close column.\n"
        f"Available columns: {df.columns.tolist()}"
    )


if predicted_col is None:

    raise RuntimeError(
        "Could not find predicted close column.\n"
        f"Available columns: {df.columns.tolist()}"
    )


print("\nActual column   :", actual_col)
print("Prediction column:", predicted_col)


# ============================================================
# CLEAN DATA
# ============================================================

plot_df = pd.DataFrame(
    index=df.index
)

plot_df["actual"] = pd.to_numeric(
    df[actual_col],
    errors="coerce"
)

plot_df["prediction"] = pd.to_numeric(
    df[predicted_col],
    errors="coerce"
)


plot_df = plot_df.dropna(
    subset=["actual"]
)


# ============================================================
# PREDICTION ERROR
# ============================================================

plot_df["error"] = (
    plot_df["prediction"]
    - plot_df["actual"]
)


plot_df["absolute_error"] = (
    plot_df["error"].abs()
)


# ============================================================
# BASIC METRICS
# ============================================================

valid = plot_df.dropna(
    subset=["actual", "prediction"]
)


if len(valid) > 0:

    mae = np.mean(
        np.abs(
            valid["prediction"]
            - valid["actual"]
        )
    )

    rmse = np.sqrt(
        np.mean(
            (
                valid["prediction"]
                - valid["actual"]
            ) ** 2
        )
    )

    correlation = (
        valid["actual"]
        .corr(valid["prediction"])
    )

else:

    mae = np.nan
    rmse = np.nan
    correlation = np.nan


print("\nMetrics")
print("-" * 40)
print(f"MAE         : {mae:.2f}")
print(f"RMSE        : {rmse:.2f}")
print(f"Correlation : {correlation:.4f}")


# ============================================================
# FIND FIRST PREDICTION DATE
# ============================================================

prediction_available = (
    plot_df["prediction"]
    .notna()
)


if prediction_available.any():

    prediction_start = (
        plot_df.index[
            prediction_available
        ][0]
    )

else:

    prediction_start = plot_df.index[0]


# ============================================================
# CREATE FIGURE
# ============================================================

fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.72, 0.28],
    subplot_titles=(
        "HCLTECH — Actual vs Rolling LoRA Prediction",
        "Prediction Error: Predicted − Actual",
    ),
)


# ============================================================
# ACTUAL PRICE
# ============================================================

fig.add_trace(
    go.Scatter(
        x=plot_df.index,
        y=plot_df["actual"],
        mode="lines",
        name="Actual Close",
        line=dict(
            width=2
        ),
        hovertemplate=(
            "<b>Date:</b> %{x|%Y-%m-%d}"
            "<br><b>Actual:</b> %{y:.2f}"
            "<extra></extra>"
        ),
    ),
    row=1,
    col=1,
)


# ============================================================
# ROLLING LoRA PREDICTION
# ============================================================

fig.add_trace(
    go.Scatter(
        x=plot_df.index,
        y=plot_df["prediction"],
        mode="lines",
        name="Rolling LoRA Prediction",
        line=dict(
            width=2
        ),
        connectgaps=False,
        hovertemplate=(
            "<b>Date:</b> %{x|%Y-%m-%d}"
            "<br><b>Prediction:</b> %{y:.2f}"
            "<extra></extra>"
        ),
    ),
    row=1,
    col=1,
)


# ============================================================
# ERROR / DIVERGENCE
# ============================================================

fig.add_trace(
    go.Scatter(
        x=plot_df.index,
        y=plot_df["error"],
        mode="lines",
        name="Prediction Error",
        line=dict(
            width=1.5
        ),
        hovertemplate=(
            "<b>Date:</b> %{x|%Y-%m-%d}"
            "<br><b>Error:</b> %{y:.2f}"
            "<extra></extra>"
        ),
    ),
    row=2,
    col=1,
)


# ============================================================
# ZERO ERROR LINE
# ============================================================

fig.add_hline(
    y=0,
    line_dash="dot",
    row=2,
    col=1,
)


# ============================================================
# PREDICTION START LINE
# ============================================================

fig.add_vline(
    x=prediction_start,
    line_dash="dash",
    line_width=2,
    annotation_text="Rolling prediction starts",
    annotation_position="top left",
    row=1,
    col=1,
)


# ============================================================
# TITLE
# ============================================================

fig.update_layout(

    title=dict(
        text=(
            "HCLTECH — Kronos + LoRA<br>"
            "<sup>"
            "400-Day Rolling Window Prediction"
            f" | MAE: {mae:.2f}"
            f" | RMSE: {rmse:.2f}"
            f" | Correlation: {correlation:.3f}"
            "</sup>"
        ),
        x=0.5,
        xanchor="center",
    ),

    template="plotly_white",

    hovermode="x unified",

    height=850,

    margin=dict(
        l=80,
        r=50,
        t=120,
        b=80,
    ),

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
    ),

    # --------------------------------------------------------
    # Plotly interactive buttons
    # --------------------------------------------------------

    xaxis=dict(
        rangeslider=dict(
            visible=True,
            thickness=0.08,
        ),

        rangeselector=dict(

            buttons=[

                dict(
                    count=1,
                    label="1Y",
                    step="year",
                    stepmode="backward",
                ),

                dict(
                    count=3,
                    label="3Y",
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

            ],

            x=0,
            y=1.12,
        ),
    ),

    # --------------------------------------------------------
    # Interactive toolbar
    # --------------------------------------------------------

    modebar=dict(
        orientation="h",
    ),
)


# ============================================================
# AXIS LABELS
# ============================================================

fig.update_yaxes(
    title_text="HCLTECH Price",
    row=1,
    col=1,
)


fig.update_yaxes(
    title_text="Error",
    row=2,
    col=1,
)


fig.update_xaxes(
    title_text="Date",
    row=2,
    col=1,
)


# ============================================================
# GRID / APPEARANCE
# ============================================================

fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    row=1,
    col=1,
)


fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    row=2,
    col=1,
)


fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    row=1,
    col=1,
)


fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    row=2,
    col=1,
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    PLOTS_DIR,
    exist_ok=True
)


fig.write_html(
    OUTPUT_FILE,
    include_plotlyjs=True,
)


print("\n" + "=" * 70)
print("PLOT CREATED SUCCESSFULLY")
print("=" * 70)

print("\nSaved to:")
print(
    os.path.abspath(
        OUTPUT_FILE
    )
)

print("\nOpen with:")
print(
    f"open {OUTPUT_FILE}"
)
