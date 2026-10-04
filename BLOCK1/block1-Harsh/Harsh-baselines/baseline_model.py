import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

TICKER = "HCLTECH.NS"

TRAIN_START = "2025-01-01"
TRAIN_END = "2025-12-31"

TEST_START = "2026-01-01"

# Number of previous days used as input
LOOKBACK = 5


# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)


# ============================================================
# 1. DOWNLOAD DATA FROM YAHOO FINANCE
# ============================================================

print(f"Downloading data for {TICKER}...")

df = yf.download(
    TICKER,
    start=TRAIN_START,
    auto_adjust=False,
    progress=False
)

# Keep only Close price
df = df[["Close"]].copy()

# Handle MultiIndex columns if returned by yfinance
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna()

print(f"Data downloaded successfully.")
print(f"First date: {df.index.min().date()}")
print(f"Latest date: {df.index.max().date()}")
print(f"Total rows: {len(df)}")


# ============================================================
# 2. SAVE RAW DATA
# ============================================================

df.to_csv("data/hcltech_yahoo_data.csv")


# ============================================================
# 3. CREATE LAG FEATURES
#
# Example:
# Lag_1 = yesterday's close
# Lag_2 = 2 days ago close
# ...
# Lag_5 = 5 days ago close
#
# Target = today's Close
# ============================================================

for lag in range(1, LOOKBACK + 1):
    df[f"Lag_{lag}"] = df["Close"].shift(lag)

df = df.dropna()


# ============================================================
# 4. SPLIT DATA
#
# TRAIN:
# 1 Jan 2025 to 31 Dec 2025
#
# TEST:
# 1 Jan 2026 to latest Yahoo Finance date
# ============================================================

train_df = df.loc[
    (df.index >= TRAIN_START) &
    (df.index <= TRAIN_END)
].copy()

test_df = df.loc[
    df.index >= TEST_START
].copy()


print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print(
    f"Training period: "
    f"{train_df.index.min().date()} to "
    f"{train_df.index.max().date()}"
)

print(f"Training samples: {len(train_df)}")

print(
    f"\nTesting period: "
    f"{test_df.index.min().date()} to "
    f"{test_df.index.max().date()}"
)

print(f"Testing samples: {len(test_df)}")


# ============================================================
# 5. PREPARE INPUT AND TARGET
# ============================================================

feature_columns = [
    f"Lag_{lag}" for lag in range(1, LOOKBACK + 1)
]

X_train = train_df[feature_columns]
y_train = train_df["Close"]

X_test = test_df[feature_columns]
y_test = test_df["Close"]


# ============================================================
# 6. TRAIN BASELINE MODEL
# ============================================================

print("\nTraining Linear Regression baseline model...")

model = LinearRegression()

model.fit(X_train, y_train)

print("Training completed.")


# ============================================================
# 7. PREDICT ON 2026 TO LATEST DATA
# ============================================================

predictions = model.predict(X_test)


# ============================================================
# 8. CREATE RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
}, index=test_df.index)

results.to_csv("results/actual_vs_predicted.csv")


# ============================================================
# 9. CALCULATE METRICS
# ============================================================

mae = mean_absolute_error(
    results["Actual"],
    results["Predicted"]
)

rmse = np.sqrt(
    mean_squared_error(
        results["Actual"],
        results["Predicted"]
    )
)

print("\n" + "=" * 60)
print("BASELINE MODEL RESULTS")
print("=" * 60)

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

print(f"\nResults saved to:")
print("results/actual_vs_predicted.csv")


# ============================================================
# 10. PLOT ACTUAL VS PREDICTED PRICE
# ============================================================

plt.figure(figsize=(16, 7))

plt.plot(
    results.index,
    results["Actual"],
    label="Actual Price",
    linewidth=2
)

plt.plot(
    results.index,
    results["Predicted"],
    label="Predicted Price",
    linewidth=2,
    alpha=0.8
)

plt.title(
    f"{TICKER} Baseline Model: Actual vs Predicted Price\n"
    f"Train: Jan 2025-Dec 2025 | "
    f"Test: Jan 2026-Latest"
)

plt.xlabel("Date")
plt.ylabel("Closing Price (INR)")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plot_path = "results/actual_vs_predicted.png"

plt.savefig(
    plot_path,
    dpi=300,
    bbox_inches="tight"
)

print(f"Plot saved to: {plot_path}")

plt.show()
