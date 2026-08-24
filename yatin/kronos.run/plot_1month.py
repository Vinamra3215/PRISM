import os
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# SETTINGS
# ============================================================

RESULTS_DIR = "_results"

# Existing Kronos CSV
CSV_FILE = os.path.join(
    RESULTS_DIR,
    "yatin-kronos-2021-120-results.csv"
)

# New graph
GRAPH_FILE = os.path.join(
    RESULTS_DIR,
    "yatin-kronos-2021-1month-plot.html"
)

# Number of trading days to display
DISPLAY_DAYS = 22


# ============================================================
# 1. LOAD EXISTING KRONOS RESULTS
# ============================================================

if not os.path.exists(CSV_FILE):
    raise FileNotFoundError(
        f"CSV file not found:\n{CSV_FILE}"
    )

df = pd.read_csv(CSV_FILE)

print("CSV columns:")
print(df.columns.tolist())


# ============================================================
# 2. FIND DATE COLUMN
# ============================================================

date_column = None

for col in df.columns:
    col_lower = str(col).lower()

    if (
        "date" in col_lower
        or "time" in col_lower
        or "timestamp" in col_lower
    ):
        date_column = col
        break

if date_column is None:
    raise ValueError(
        "No date/timestamp column found in the CSV."
    )


df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)

df = df.dropna(
    subset=[date_column]
).sort_values(
    date_column
).reset_index(drop=True)


# ============================================================
# 3. FIND PRICE COLUMNS
# ============================================================

def find_column(possible_names):

    for col in df.columns:

        clean = (
            str(col)
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )

        for name in possible_names:

            if name in clean:
                return col

    return None


actual_close = find_column([
    "actualclose",
    "close"
])

predicted_close = find_column([
    "predictedclose",
    "forecastclose",
    "predclose"
])


if actual_close is None:
    raise ValueError(
        "Could not find actual Close column."
    )

if predicted_close is None:
    raise ValueError(
        "Could not find predicted Close column."
    )


print()
print("Date column       :", date_column)
print("Actual close      :", actual_close)
print("Predicted close   :", predicted_close)


# ============================================================
# 4. CONVERT TO NUMERIC
# ============================================================

df[actual_close] = pd.to_numeric(
    df[actual_close],
    errors="coerce"
)

df[predicted_close] = pd.to_numeric(
    df[predicted_close],
    errors="coerce"
)

df = df.dropna(
    subset=[
        actual_close,
        predicted_close
    ]
).reset_index(drop=True)


# ============================================================
# 5. SELECT ONLY 1 MONTH
# ============================================================

plot_df = df.head(DISPLAY_DAYS).copy()

if len(plot_df) == 0:
    raise ValueError(
        "No data available for plotting."
    )


print()
print(
    f"Plotting {len(plot_df)} trading days "
    f"(approximately 1 month)."
)

print(
    "Start:",
    plot_df[date_column].iloc[0]
)

print(
    "End  :",
    plot_df[date_column].iloc[-1]
)


# ============================================================
# 6. CALCULATE DIVERGENCE
# ============================================================

plot_df["divergence"] = (
    plot_df[predicted_close]
    - plot_df[actual_close]
)


# ============================================================
# 7. FIND MAX DIVERGENCE POINT
# ============================================================

max_divergence_index = (
    plot_df["divergence"]
    .abs()
    .idxmax()
)

divergence_date = plot_df.loc[
    max_divergence_index,
    date_column
]

divergence_actual = plot_df.loc[
    max_divergence_index,
    actual_close
]

divergence_predicted = plot_df.loc[
    max_divergence_index,
    predicted_close
]

divergence_value = plot_df.loc[
    max_divergence_index,
    "divergence"
]


# ============================================================
# 8. CREATE PLOTLY GRAPH
# ============================================================

fig = go.Figure()


# ------------------------------------------------------------
# Actual price
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=plot_df[date_column],
        y=plot_df[actual_close],
        mode="lines",
        name="Actual Close",
        line=dict(
            width=2
        )
    )
)


# ------------------------------------------------------------
# Kronos predicted price
# ------------------------------------------------------------

fig.add_trace(
    go.Scatter(
        x=plot_df[date_column],
        y=plot_df[predicted_close],
        mode="lines",
        name="Kronos Predicted Close",
        line=dict(
            width=2,
            dash="dash"
        )
    )
)


# ============================================================
# 9. DIVERGENCE VERTICAL LINE
# ============================================================

fig.add_vline(
    x=divergence_date,
    line_width=2,
    line_dash="dot",
    line_color="red"
)


# ============================================================
# 10. DIVERGENCE ANNOTATION
# ============================================================

fig.add_annotation(
    x=divergence_date,
    y=max(
        divergence_actual,
        divergence_predicted
    ),
    text=(
        f"Divergence<br>"
        f"Actual: {divergence_actual:.2f}<br>"
        f"Predicted: {divergence_predicted:.2f}<br>"
        f"Difference: {divergence_value:.2f}"
    ),
    showarrow=True,
    arrowhead=2
)


# ============================================================
# 11. GRAPH LAYOUT
# ============================================================

fig.update_layout(
    title=(
        "RELIANCE.NS — Kronos "
        "Next 1 Month Forecast"
    ),

    xaxis_title="Date",

    yaxis_title="Price",

    hovermode="x unified",

    template="plotly_white",

    height=800,

    xaxis=dict(
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)


# ============================================================
# 12. SAVE GRAPH
# ============================================================

fig.write_html(
    GRAPH_FILE,
    include_plotlyjs=True,
    full_html=True
)


# ============================================================
# 13. FINAL OUTPUT
# ============================================================

print()
print("=" * 70)
print("1-MONTH GRAPH CREATED")
print("=" * 70)

print()
print("Graph:")
print(GRAPH_FILE)

print()
print("Days plotted :", len(plot_df))

print(
    "Start date   :",
    plot_df[date_column].iloc[0]
)

print(
    "End date     :",
    plot_df[date_column].iloc[-1]
)

print(
    "Divergence   :",
    divergence_date
)

print(
    "Difference   :",
    f"{divergence_value:.2f}"
)

print()
print("=" * 70)

