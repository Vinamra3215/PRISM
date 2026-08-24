import os
import sys
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# SETTINGS
# ============================================================

LOCAL_FILE = "/home/soq/yatin/yatin2/data/parquet_data/HCLTECH-EQ_1min.parquet"

OUTPUT_DIR = "/home/soq/yatin/kronos.run/hcltech_api"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(
    OUTPUT_DIR,
    "HCLTECH_1min_local.csv"
)

HTML_FILE = os.path.join(
    OUTPUT_DIR,
    "HCLTECH_1min_interactive.html"
)


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 80)
print("HCLTECH 1-MINUTE DATA PLOT")
print("=" * 80)

print()
print("Source:")
print(LOCAL_FILE)

print()


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(LOCAL_FILE):

    print("ERROR: HCLTECH file does not exist.")
    print(LOCAL_FILE)

    sys.exit(1)


# ============================================================
# READ PARQUET
# ============================================================

print("Reading HCLTECH parquet...")

try:

    df = pd.read_parquet(LOCAL_FILE)

except Exception as e:

    print()
    print("ERROR READING PARQUET:")
    print(e)

    print()
    print("If this says pyarrow is missing, run:")
    print()
    print("python -m pip install pyarrow")

    sys.exit(1)


print("Data loaded.")


# ============================================================
# DISPLAY ORIGINAL INFORMATION
# ============================================================

print()
print("=" * 80)
print("ORIGINAL DATA")
print("=" * 80)

print()
print("Rows:", len(df))

print()
print("Columns:")
print(df.columns.tolist())

print()


# ============================================================
# FIND TIMESTAMP COLUMN
# ============================================================

timestamp_column = None

for col in ["timestamp", "Timestamp", "datetime", "Datetime", "date", "Date"]:

    if col in df.columns:

        timestamp_column = col
        break


if timestamp_column is None:

    if isinstance(df.index, pd.DatetimeIndex):

        df = df.reset_index()

        timestamp_column = df.columns[0]

    else:

        print("ERROR: Cannot find timestamp column.")

        sys.exit(1)


print("Timestamp column:", timestamp_column)


# ============================================================
# CONVERT TIMESTAMP
# ============================================================

df[timestamp_column] = pd.to_datetime(
    df[timestamp_column],
    errors="coerce"
)


# Remove invalid timestamps
df = df.dropna(
    subset=[timestamp_column]
)


# ============================================================
# TIMEZONE
# ============================================================

if df[timestamp_column].dt.tz is None:

    df[timestamp_column] = (
        df[timestamp_column]
        .dt.tz_localize(
            "Asia/Kolkata"
        )
    )

else:

    df[timestamp_column] = (
        df[timestamp_column]
        .dt.tz_convert(
            "Asia/Kolkata"
        )
    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    timestamp_column
).reset_index(drop=True)


# Remove duplicate timestamps
df = df.drop_duplicates(
    subset=[timestamp_column]
).reset_index(drop=True)


# ============================================================
# STANDARDIZE COLUMN NAMES
# ============================================================

rename_map = {}

for col in df.columns:

    lower = str(col).lower()

    if lower == "open":
        rename_map[col] = "Open"

    elif lower == "high":
        rename_map[col] = "High"

    elif lower == "low":
        rename_map[col] = "Low"

    elif lower == "close":
        rename_map[col] = "Close"

    elif lower == "volume":
        rename_map[col] = "Volume"


df = df.rename(
    columns=rename_map
)


# ============================================================
# CHECK OHLCV
# ============================================================

required_columns = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    print()
    print("ERROR: Missing columns:")
    print(missing)

    print()
    print("Available columns:")
    print(df.columns.tolist())

    sys.exit(1)


# ============================================================
# REMOVE INVALID OHLCV
# ============================================================

df = df.dropna(
    subset=required_columns
)


# ============================================================
# FINAL DATA INFORMATION
# ============================================================

print()
print("=" * 80)
print("FINAL DATA")
print("=" * 80)

print()

print("Rows:", len(df))

print()

print("Start:")
print(df[timestamp_column].min())

print()

print("End:")
print(df[timestamp_column].max())

print()

print("Trading dates:")

print(
    df[timestamp_column]
    .dt.date
    .nunique()
)


# ============================================================
# SAVE CLEAN DATA
# ============================================================

df.to_csv(
    CSV_FILE,
    index=False
)

print()
print("Clean CSV saved:")
print(CSV_FILE)


# ============================================================
# PLOT
# ============================================================

print()
print("Creating interactive graph...")


fig = go.Figure()


# ------------------------------------------------------------
# CANDLESTICK
# ------------------------------------------------------------

fig.add_trace(
    go.Candlestick(

        x=df[timestamp_column],

        open=df["Open"],

        high=df["High"],

        low=df["Low"],

        close=df["Close"],

        name="HCLTECH"
    )
)


# ------------------------------------------------------------
# VOLUME
# ------------------------------------------------------------

fig.add_trace(
    go.Bar(

        x=df[timestamp_column],

        y=df["Volume"],

        name="Volume",

        opacity=0.30,

        yaxis="y2"
    )
)


# ============================================================
# LAYOUT
# ============================================================

fig.update_layout(

    title={
        "text": "HCLTECH — 1 Minute OHLCV",
        "x": 0.5
    },

    xaxis_title="Time",

    yaxis_title="Price (₹)",

    yaxis2=dict(

        title="Volume",

        overlaying="y",

        side="right",

        showgrid=False
    ),

    height=800,

    hovermode="x unified",

    xaxis=dict(

        rangeslider=dict(
            visible=True
        ),

        rangeselector=dict(

            buttons=[

                dict(
                    count=1,
                    label="1H",
                    step="hour",
                    stepmode="backward"
                ),

                dict(
                    count=3,
                    label="3H",
                    step="hour",
                    stepmode="backward"
                ),

                dict(
                    count=1,
                    label="1D",
                    step="day",
                    stepmode="backward"
                ),

                dict(
                    count=3,
                    label="3D",
                    step="day",
                    stepmode="backward"
                ),

                dict(
                    count=7,
                    label="7D",
                    step="day",
                    stepmode="backward"
                ),

                dict(
                    count=1,
                    label="1M",
                    step="month",
                    stepmode="backward"
                ),

                dict(
                    label="ALL",
                    step="all"
                )
            ]
        )
    ),

    template="plotly_white"
)


# ============================================================
# SAVE HTML
# ============================================================

fig.write_html(
    HTML_FILE,
    include_plotlyjs=True
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 80)
print("SUCCESS")
print("=" * 80)

print()

print("Interactive HTML:")
print(HTML_FILE)

print()

print("CSV:")
print(CSV_FILE)

print()

print("You can now:")
print("  - Zoom")
print("  - Pan")
print("  - Hover over individual minutes")
print("  - Use 1H / 3H / 1D / 3D / 7D / 1M")
print("  - Drag the range slider")

print()
print("=" * 80)