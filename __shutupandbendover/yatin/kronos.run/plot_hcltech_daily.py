import os
import pandas as pd
import plotly.graph_objects as go

# ============================================================
# SETTINGS
# ============================================================

TICKER = "HCLTECH.NS"

START_DATE = "2022-01-01"
END_DATE = "2026-08-14"

LOCAL_FILE = "/home/soq/yatin/yatin2/data/parquet_data/HCLTECH-EQ_1min.parquet"

OUTPUT_DIR = "/home/soq/yatin/kronos.run/hcltech_daily"

DAILY_CSV = os.path.join(
    OUTPUT_DIR,
    "HCLTECH_daily_2022_2026.csv"
)

HTML_FILE = os.path.join(
    OUTPUT_DIR,
    "HCLTECH_daily_interactive.html"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# STEP 1: LOAD LOCAL 1-MINUTE DATA
# ============================================================

print("=" * 70)
print("HCLTECH DAILY DATA + PLOTLY")
print("=" * 70)

print()
print("STEP 1: Loading local HCLTECH data...")
print("File:")
print(LOCAL_FILE)

if not os.path.exists(LOCAL_FILE):
    raise FileNotFoundError(
        f"\nLocal HCLTECH file not found:\n{LOCAL_FILE}"
    )

df = pd.read_parquet(LOCAL_FILE)

print()
print("Rows loaded:", len(df))
print("Columns:", df.columns.tolist())


# ============================================================
# STEP 2: FIND TIMESTAMP COLUMN
# ============================================================

possible_timestamp_columns = [
    "timestamp",
    "datetime",
    "Date",
    "date",
    "Datetime"
]

timestamp_column = None

for col in possible_timestamp_columns:
    if col in df.columns:
        timestamp_column = col
        break

if timestamp_column is None:

    # Sometimes timestamp is the index
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()

        for col in possible_timestamp_columns:
            if col in df.columns:
                timestamp_column = col
                break

        if timestamp_column is None:
            timestamp_column = df.columns[0]

    else:
        raise ValueError(
            "Could not find timestamp column."
        )

print()
print("Timestamp column:", timestamp_column)


# ============================================================
# STEP 3: CONVERT TIMESTAMP
# ============================================================

df[timestamp_column] = pd.to_datetime(
    df[timestamp_column],
    errors="coerce"
)

df = df.dropna(
    subset=[timestamp_column]
)

# Convert timezone correctly
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

df = df.sort_values(
    timestamp_column
)

print()
print(
    "Local data range:"
)

print(
    df[timestamp_column].min()
)

print(
    df[timestamp_column].max()
)


# ============================================================
# STEP 4: IDENTIFY OHLCV COLUMNS
# ============================================================

def find_column(possible_names):

    for name in possible_names:

        if name in df.columns:
            return name

    return None


open_col = find_column(
    ["open", "Open", "OPEN"]
)

high_col = find_column(
    ["high", "High", "HIGH"]
)

low_col = find_column(
    ["low", "Low", "LOW"]
)

close_col = find_column(
    ["close", "Close", "CLOSE"]
)

volume_col = find_column(
    ["volume", "Volume", "VOLUME"]
)


print()
print("Detected columns:")

print("Open   :", open_col)
print("High   :", high_col)
print("Low    :", low_col)
print("Close  :", close_col)
print("Volume :", volume_col)


if any(
    x is None
    for x in [
        open_col,
        high_col,
        low_col,
        close_col
    ]
):

    raise ValueError(
        "Required OHLC columns are missing."
    )


# ============================================================
# STEP 5: RENAME COLUMNS
# ============================================================

rename_dict = {

    timestamp_column: "timestamp",

    open_col: "open",

    high_col: "high",

    low_col: "low",

    close_col: "close"
}

if volume_col is not None:

    rename_dict[volume_col] = "volume"

df = df.rename(
    columns=rename_dict
)


# ============================================================
# STEP 6: KEEP ONLY REQUIRED COLUMNS
# ============================================================

columns = [
    "timestamp",
    "open",
    "high",
    "low",
    "close"
]

if "volume" in df.columns:
    columns.append("volume")

df = df[columns]


# ============================================================
# STEP 7: REMOVE BAD VALUES
# ============================================================

df = df.dropna(
    subset=[
        "open",
        "high",
        "low",
        "close"
    ]
)

df = df[
    (df["close"] > 0)
]


# ============================================================
# STEP 8: CONVERT 1-MINUTE → DAILY
# ============================================================

print()
print("=" * 70)
print("STEP 2: CONVERTING 1-MINUTE DATA TO DAILY DATA")
print("=" * 70)

df = df.set_index(
    "timestamp"
)

daily_dict = {

    "open": "first",

    "high": "max",

    "low": "min",

    "close": "last"
}

if "volume" in df.columns:

    daily_dict["volume"] = "sum"


daily = (
    df
    .resample("1D")
    .agg(daily_dict)
    .dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )
)


# ============================================================
# STEP 9: KEEP REQUESTED PERIOD
# ============================================================

start_ts = pd.Timestamp(
    START_DATE,
    tz="Asia/Kolkata"
)

end_ts = pd.Timestamp(
    END_DATE,
    tz="Asia/Kolkata"
)

daily = daily[
    (daily.index >= start_ts)
    &
    (daily.index <= end_ts)
]


# ============================================================
# STEP 10: RESET INDEX
# ============================================================

daily = daily.reset_index()

daily = daily.rename(
    columns={
        "timestamp": "datetime"
    }
)

daily["datetime"] = (
    daily["datetime"]
    .dt.tz_localize(None)
)


# ============================================================
# STEP 11: SHOW DATA INFORMATION
# ============================================================

print()
print("=" * 70)
print("DAILY DATA")
print("=" * 70)

print(
    "Rows:",
    len(daily)
)

if len(daily) > 0:

    print(
        "Start:",
        daily["datetime"].min()
    )

    print(
        "End:",
        daily["datetime"].max()
    )

print()
print(
    daily.head()
)

print()
print(
    daily.tail()
)


# ============================================================
# STEP 12: SAVE DAILY CSV
# ============================================================

daily.to_csv(
    DAILY_CSV,
    index=False
)

print()
print("Daily CSV saved:")
print(DAILY_CSV)


# ============================================================
# STEP 13: CREATE PLOTLY GRAPH
# ============================================================

print()
print("=" * 70)
print("STEP 3: CREATING INTERACTIVE PLOTLY GRAPH")
print("=" * 70)

fig = go.Figure()


# ------------------------------------------------------------
# Candlestick
# ------------------------------------------------------------

fig.add_trace(
    go.Candlestick(

        x=daily["datetime"],

        open=daily["open"],

        high=daily["high"],

        low=daily["low"],

        close=daily["close"],

        name="HCLTECH",

        increasing_line_color="#00aa00",

        decreasing_line_color="#dd0000"
    )
)


# ============================================================
# 20-DAY MOVING AVERAGE
# ============================================================

daily["MA20"] = (
    daily["close"]
    .rolling(20)
    .mean()
)

fig.add_trace(
    go.Scatter(

        x=daily["datetime"],

        y=daily["MA20"],

        mode="lines",

        name="MA20",

        line=dict(
            width=1.5
        )
    )
)


# ============================================================
# 50-DAY MOVING AVERAGE
# ============================================================

daily["MA50"] = (
    daily["close"]
    .rolling(50)
    .mean()
)

fig.add_trace(
    go.Scatter(

        x=daily["datetime"],

        y=daily["MA50"],

        mode="lines",

        name="MA50",

        line=dict(
            width=1.5
        )
    )
)


# ============================================================
# LAYOUT
# ============================================================

fig.update_layout(

    title=(
        "HCLTECH Daily OHLCV "
        "2022-01-01 to 2026-08-14"
    ),

    xaxis_title="Date",

    yaxis_title="Price",

    template="plotly_white",

    hovermode="x unified",

    xaxis=dict(

        rangeslider=dict(
            visible=True
        ),

        rangeselector=dict(

            buttons=[

                dict(
                    count=7,
                    label="1W",
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
                    count=3,
                    label="3M",
                    step="month",
                    stepmode="backward"
                ),

                dict(
                    count=6,
                    label="6M",
                    step="month",
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
        )
    ),

    height=750
)


# ============================================================
# SAVE HTML
# ============================================================

fig.write_html(
    HTML_FILE,
    include_plotlyjs=True
)

print()
print("=" * 70)

print(
    "INTERACTIVE PLOT CREATED SUCCESSFULLY"
)

print("=" * 70)

print()
print("HTML:")
print(HTML_FILE)

print()
print("CSV:")
print(DAILY_CSV)

print()
print("You can:")
print("  - Zoom")
print("  - Pan")
print("  - Hover over dates")
print("  - Use 1W / 1M / 3M / 6M / 1Y")
print("  - Drag the range slider")

print()
print("=" * 70)