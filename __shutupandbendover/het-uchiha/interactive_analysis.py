import pandas as pd
import plotly.graph_objects as go

# ============================================================
# PATHS
# ============================================================

DATA_PATH = "/home/soq/__shutupandbendover/het-uchiha/RELIANCE_2016_2021.parquet"

PREDICTION_PATH = (
    "/home/soq/__shutupandbendover/het-uchiha/"
    "reliance_2020_jan2021_predictions.csv"
)

OUTPUT_HTML = (
    "/home/soq/__shutupandbendover/het-uchiha/"
    "reliance_kronos_full_graph.html"
)


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

df = pd.read_parquet(DATA_PATH)

# Handle MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df.columns
    ]

df.columns = [str(c).lower() for c in df.columns]

# Make sure index is datetime
df.index = pd.to_datetime(df.index)
df = df.sort_index()


# ============================================================
# EXTRACT 2020 DATA THAT WAS FED TO KRONOS
# ============================================================

input_data = df.loc["2020-01-01":"2020-12-31"].copy()


# ============================================================
# LOAD KRONOS PREDICTIONS
# ============================================================

pred = pd.read_csv(PREDICTION_PATH)

# Find date column
date_col = None

for col in pred.columns:
    if "date" in col.lower() or "time" in col.lower():
        date_col = col
        break

if date_col is None:
    raise ValueError("Could not find date column in prediction CSV.")

pred[date_col] = pd.to_datetime(pred[date_col])
pred = pred.sort_values(date_col)


# Find prediction column
prediction_col = "Kronos_Predicted_Close"



if prediction_col is None:
    # Show available columns so we know what the CSV contains
    print("Prediction CSV columns:", pred.columns.tolist())
    raise ValueError("Could not identify prediction column.")


# ============================================================
# ACTUAL JANUARY 2021 DATA
# ============================================================

actual_jan = df.loc["2021-01-01":"2021-01-31"].copy()


# ============================================================
# METRICS
# ============================================================

# Match prediction dates with actual dates
comparison = pd.merge(
    pred[[date_col, prediction_col]],
    actual_jan[["close"]].reset_index(),
    left_on=date_col,
    right_on=actual_jan.index.name if actual_jan.index.name else "index",
    how="inner"
)

if len(comparison) > 0:

    y_pred = comparison[prediction_col].astype(float)
    y_actual = comparison["close"].astype(float)

    mae = (y_actual - y_pred).abs().mean()

    rmse = ((y_actual - y_pred) ** 2).mean() ** 0.5

    mape = (
        ((y_actual - y_pred).abs() / y_actual.abs()).mean()
        * 100
    )

else:
    mae = rmse = mape = None


# ============================================================
# CREATE GRAPH
# ============================================================

fig = go.Figure()


# ------------------------------------------------------------
# 2020 DATA FED INTO KRONOS
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=input_data.index,
        y=input_data["close"],
        mode="lines",
        name="2020 Input Data",
        line=dict(width=2),
        hovertemplate=
            "<b>%{x|%Y-%m-%d}</b><br>"
            "Actual Close: ₹%{y:.2f}"
            "<extra></extra>"
    )
)


# ------------------------------------------------------------
# KRONOS PREDICTION
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=pred[date_col],
        y=pred[prediction_col],
        mode="lines+markers",
        name="Kronos Prediction",
        line=dict(width=2, dash="dash"),
        hovertemplate=
            "<b>%{x|%Y-%m-%d}</b><br>"
            "Kronos: ₹%{y:.2f}"
            "<extra></extra>"
    )
)


# ------------------------------------------------------------
# ACTUAL JANUARY 2021
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=actual_jan.index,
        y=actual_jan["close"],
        mode="lines+markers",
        name="January 2021 Actual",
        line=dict(width=2),
        hovertemplate=
            "<b>%{x|%Y-%m-%d}</b><br>"
            "Actual: ₹%{y:.2f}"
            "<extra></extra>"
    )
)


# ============================================================
# METRICS BOX
# ============================================================

if mae is not None:

    metrics_text = (
        "<b>Model Performance</b><br>"
        f"MAE: ₹{mae:.2f}<br>"
        f"RMSE: ₹{rmse:.2f}<br>"
        f"MAPE: {mape:.2f}%"
    )

else:

    metrics_text = (
        "<b>Model Performance</b><br>"
        "No matching actual/prediction dates"
    )


fig.add_annotation(
    x=0.01,
    y=0.98,
    xref="paper",
    yref="paper",
    text=metrics_text,
    showarrow=False,
    align="left",
    bgcolor="rgba(20,30,45,0.9)",
    bordercolor="rgba(255,255,255,0.3)",
    borderwidth=1,
    font=dict(color="white", size=14)
)


# ============================================================
# FORECAST START MARKER
# ============================================================

fig.add_vline(
    x=pd.Timestamp("2021-01-01"),
    line_dash="dot",
    line_width=2,
    annotation_text="Forecast begins",
    annotation_position="top"
)


# ============================================================
# LAYOUT
# ============================================================

fig.update_layout(

    title=dict(
        text=(
            "RELIANCE.NS — Kronos Model "
            "Input Data vs Prediction vs Ground Truth"
        ),
        x=0.5
    ),

    xaxis_title="Date",

    yaxis_title="RELIANCE.NS Close Price (₹)",

    template="plotly_dark",

    hovermode="x unified",

    height=750,

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),

    margin=dict(
        l=70,
        r=40,
        t=110,
        b=70
    )
)


# ============================================================
# SAVE
# ============================================================

fig.write_html(
    OUTPUT_HTML,
    include_plotlyjs=True
)

print()
print("==============================================")
print("GRAPH CREATED")
print("==============================================")
print(f"2020 input bars : {len(input_data)}")
print(f"Prediction bars : {len(pred)}")
print(f"Actual Jan bars : {len(actual_jan)}")

if mae is not None:
    print(f"MAE             : ₹{mae:.2f}")
    print(f"RMSE            : ₹{rmse:.2f}")
    print(f"MAPE            : {mape:.2f}%")

print()
print("HTML saved to:")
print(OUTPUT_HTML)