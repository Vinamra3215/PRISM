import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go


# ============================================================
# 1. FILE PATH
# ============================================================

FILE_PATH = "/home/soq/yatin/baseline/NIFTY50_2022_to_today.parquet"


# ============================================================
# 2. LOAD DATA
# ============================================================

df = pd.read_parquet(FILE_PATH)

print("\nOriginal columns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))


# ============================================================
# 3. CLEAN THE DATA
#
# Your parquet structure is:
#
# Row 0 -> Ticker information
# Row 1 -> Date information
# Row 2 onwards -> Actual data
#
# Column 'Price' contains the dates
# ============================================================

df = df.iloc[2:].copy()

# Rename Price column to Date
df = df.rename(columns={"Price": "Date"})


# Convert Date
df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


# Convert Close to numeric
df["Close"] = pd.to_numeric(
    df["Close"],
    errors="coerce"
)


# Remove invalid rows
df = df.dropna(
    subset=["Date", "Close"]
).copy()


# Sort by date
df = df.sort_values("Date").reset_index(drop=True)


print("\n" + "=" * 60)
print("CLEANED DATA")
print("=" * 60)

print("\nFirst 5 rows:")
print(df[["Date", "Close"]].head())

print("\nLast 5 rows:")
print(df[["Date", "Close"]].tail())

print("\nData range:")
print("Start date:", df["Date"].min())
print("End date:", df["Date"].max())

print("\nTotal rows:", len(df))


# ============================================================
# 4. CREATE LAG FEATURES
#
# Previous 5 days are used to predict today's Close price
# ============================================================

for lag in range(1, 6):
    df[f"Lag_{lag}"] = df["Close"].shift(lag)


df = df.dropna().copy()


feature_columns = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "Lag_4",
    "Lag_5"
]


# ============================================================
# 5. SPLIT DATA
#
# TRAIN:
# 1 Jan 2025 -> 31 Dec 2025
#
# PREDICT / TEST:
# 1 Jan 2026 -> Last available date
# ============================================================

train_start = pd.Timestamp("2025-01-01")
train_end = pd.Timestamp("2025-12-31")

test_start = pd.Timestamp("2026-01-01")


train_df = df[
    (df["Date"] >= train_start) &
    (df["Date"] <= train_end)
].copy()


test_df = df[
    df["Date"] >= test_start
].copy()


print("\n" + "=" * 60)
print("TRAINING DATA")
print("=" * 60)

print("Start:", train_df["Date"].min())
print("End:", train_df["Date"].max())
print("Number of samples:", len(train_df))


print("\n" + "=" * 60)
print("PREDICTION / TEST DATA")
print("=" * 60)

print("Start:", test_df["Date"].min())
print("End:", test_df["Date"].max())
print("Number of samples:", len(test_df))


# Check training data
if len(train_df) == 0:
    raise ValueError(
        "No training data found between "
        "1 Jan 2025 and 31 Dec 2025."
    )


# Check test data
if len(test_df) == 0:
    raise ValueError(
        "No data found from 1 Jan 2026 onward."
    )


# ============================================================
# 6. PREPARE TRAINING DATA
# ============================================================

X_train = train_df[feature_columns]
y_train = train_df["Close"]


X_test = test_df[feature_columns]
y_test = test_df["Close"]


# ============================================================
# 7. TRAIN BASELINE MODEL
# ============================================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nBaseline model trained successfully.")


# ============================================================
# 8. MAKE PREDICTIONS
# ============================================================

test_df["Predicted_Close"] = model.predict(X_test)


# ============================================================
# 9. CALCULATE ERRORS
# ============================================================

mae = np.mean(
    np.abs(
        test_df["Close"] -
        test_df["Predicted_Close"]
    )
)


rmse = np.sqrt(
    np.mean(
        (
            test_df["Close"] -
            test_df["Predicted_Close"]
        ) ** 2
    )
)


print("\n" + "=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")


# ============================================================
# 10. CREATE ACTUAL VS PREDICTED PLOT
# ============================================================

fig = go.Figure()


# Actual Close Price
fig.add_trace(
    go.Scatter(
        x=test_df["Date"],
        y=test_df["Close"],
        mode="lines",
        name="Actual Close Price",
        line=dict(width=2)
    )
)


# Predicted Close Price
fig.add_trace(
    go.Scatter(
        x=test_df["Date"],
        y=test_df["Predicted_Close"],
        mode="lines",
        name="Predicted Close Price",
        line=dict(width=2)
    )
)


# ============================================================
# 11. GRAPH LAYOUT
# ============================================================

fig.update_layout(
    title=(
        "NIFTY50 Baseline Model: Actual vs Predicted Price"
        "<br><sup>"
        "Training: 1 Jan 2025 - 31 Dec 2025 | "
        "Prediction: 1 Jan 2026 - Last Available Date"
        "</sup>"
    ),
    xaxis_title="Date",
    yaxis_title="NIFTY50 Close Price",
    hovermode="x unified",
    template="plotly_white",
    height=700,
    width=1300
)


# Interactive date slider
fig.update_xaxes(
    rangeslider_visible=True
)


# ============================================================
# 12. SAVE GRAPH
# ============================================================

OUTPUT_FILE = (
    "/home/soq/yatin/baseline/"
    "nifty_baseline_actual_vs_predicted.html"
)

fig.write_html(OUTPUT_FILE)


print("\n" + "=" * 60)
print("GRAPH SAVED SUCCESSFULLY")
print("=" * 60)

print(OUTPUT_FILE)


# ============================================================
# 13. SHOW GRAPH
# ============================================================

fig.show()