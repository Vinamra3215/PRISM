import pandas as pd
import plotly.graph_objects as go

# ============================================================
# PATHS
# ============================================================

DATA_FILE = "data/NIFTY50_5Y_OHLCV.parquet"
OUTPUT_FILE = "plots/baseline_1year.html"


# ============================================================
# LOAD DATA
# ============================================================

print("========================================")
print("LOADING NIFTY50 DATA")
print("========================================")

df = pd.read_parquet(DATA_FILE)

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# ============================================================
# HANDLE MULTIINDEX
# ============================================================

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)


# ============================================================
# GET DATE FROM INDEX
# ============================================================

if isinstance(df.index, pd.DatetimeIndex):

    df = df.reset_index()

    # Rename index column
    df.rename(
        columns={df.columns[0]: "timestamp"},
        inplace=True
    )

else:

    # Find date column
    date_column = None

    for col in df.columns:

        name = str(col).lower()

        if "date" in name or "time" in name:
            date_column = col
            break

    if date_column is None:

        print("ERROR: Date column not found.")
        print(df.columns.tolist())

        raise SystemExit

    df.rename(
        columns={date_column: "timestamp"},
        inplace=True
    )


# ============================================================
# STANDARDIZE CLOSE COLUMN
# ============================================================

close_column = None

for col in df.columns:

    name = str(col).lower()

    if name == "close":

        close_column = col
        break


if close_column is None:

    print("ERROR: Close column not found.")
    print(df.columns.tolist())

    raise SystemExit


df.rename(
    columns={close_column: "close"},
    inplace=True
)


# ============================================================
# CLEAN DATA
# ============================================================

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

df["close"] = pd.to_numeric(
    df["close"],
    errors="coerce"
)

df = df.dropna(
    subset=["timestamp", "close"]
)

df = df.sort_values(
    "timestamp"
)

df = df.reset_index(drop=True)


# ============================================================
# SELECT ONE YEAR
# ============================================================

# CHANGE THESE DATES IF YOUR ASSIGNED YEAR IS DIFFERENT

START_DATE = "2024-01-01"
END_DATE = "2024-12-31"

year_df = df[
    (df["timestamp"] >= START_DATE)
    &
    (df["timestamp"] <= END_DATE)
].copy()


print("\n========================================")
print("ONE YEAR DATA")
print("========================================")

print("Start:", year_df["timestamp"].min())
print("End  :", year_df["timestamp"].max())
print("Rows :", len(year_df))


# ============================================================
# CREATE GRAPH
# ============================================================

fig = go.Figure()


fig.add_trace(
    go.Scatter(
        x=year_df["timestamp"],
        y=year_df["close"],
        mode="lines",
        name="NIFTY50 Close",
        line=dict(width=2)
    )
)


# ============================================================
# GRAPH SETTINGS
# ============================================================

fig.update_layout(

    title="NIFTY50 - 1 Year Baseline",

    xaxis_title="Date",

    yaxis_title="NIFTY50 Price",

    template="plotly_white",

    hovermode="x unified",

    width=1400,

    height=700,

    xaxis=dict(

        rangeslider=dict(
            visible=True
        ),

        type="date"
    ),

    margin=dict(
        l=80,
        r=80,
        t=80,
        b=80
    )
)


# ============================================================
# SAVE
# ============================================================

fig.write_html(
    OUTPUT_FILE
)


print("\n========================================")
print("GRAPH CREATED SUCCESSFULLY")
print("========================================")

print("File:")
print(OUTPUT_FILE)