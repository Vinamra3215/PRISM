import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import Kronos, KronosTokenizer, KronosPredictor


# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "./data/RELIANCE_2016_2021.parquet"
TOKENIZER_PATH = "./weights/Kronos-Tokenizer-base"
MODEL_PATH = "./weights/Kronos-base"

OUTPUT_CSV = "./results/reliance_2020_jan2021_predictions.csv"
OUTPUT_PLOT = "./results/reliance_2020_jan2021_forecast.png"


# ============================================================
# 2. LOAD KRONOS
# ============================================================

print("Loading Kronos tokenizer...")
tokenizer = KronosTokenizer.from_pretrained(TOKENIZER_PATH)

print("Loading Kronos model...")
model = Kronos.from_pretrained(MODEL_PATH)

print("Creating predictor...")
predictor = KronosPredictor(
    model,
    tokenizer,
    max_context=512
)


# ============================================================
# 3. LOAD RELIANCE DATA
# ============================================================

print("\nLoading Reliance data...")

df = pd.read_parquet(DATA_PATH)

# Yahoo Finance can produce MultiIndex columns such as:
# ('Close', 'RELIANCE.NS')
# Convert them to:
# Close, High, Low, Open, Volume

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [
        col[0] if isinstance(col, tuple) else col
        for col in df.columns
    ]

df.columns = [str(col).lower() for col in df.columns]

# Make sure the index is datetime
df.index = pd.to_datetime(df.index)

# Sort chronologically
df = df.sort_index()

print("Available data:")
print(df.index.min(), "->", df.index.max())
print("Columns:", df.columns.tolist())


# ============================================================
# 4. SPLIT THE DATA
# ============================================================

# IMPORTANT:
# Kronos gets ONLY 2020 data.
#
# January 2021 is completely hidden from Kronos and is used
# only AFTER prediction for evaluation.

train = df.loc["2020-01-01":"2020-12-31"].copy()

actual = df.loc["2021-01-01":"2021-01-31"].copy()

print("\nExperiment split:")
print("2020 input:", train.index.min(), "->", train.index.max())
print("2020 bars:", len(train))

print(
    "January 2021 target:",
    actual.index.min(),
    "->",
    actual.index.max()
)
print("January 2021 bars:", len(actual))


# ============================================================
# 5. PREPARE DATA FOR KRONOS
# ============================================================

# Kronos expects these columns.
required_columns = ["open", "high", "low", "close", "volume"]

missing = [
    col for col in required_columns
    if col not in train.columns
]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Historical timestamps that Kronos is allowed to see
x_timestamp = train.index.to_series()

# Future timestamps that Kronos should predict
y_timestamp = actual.index.to_series()

pred_len = len(actual)


# ============================================================
# 6. RUN KRONOS
# ============================================================

print("\n" + "=" * 60)
print("RUNNING KRONOS")
print("=" * 60)

print(f"Historical context : {len(train)} bars")
print(f"Forecast horizon   : {pred_len} bars")
print(
    f"Forecast period    : "
    f"{y_timestamp[0].date()} -> {y_timestamp[-1].date()}"
)

print("\nKronos is now generating the January 2021 forecast...")
print("January 2021 prices are NOT provided to the model.")

predictions = predictor.predict(
    df=train,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,
    top_k=0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)


# ============================================================
# 7. EXTRACT CLOSE PREDICTIONS
# ============================================================

# Kronos returns predictions for Open, High, Low, Close,
# Volume, etc. We only need Close for our main evaluation.

pred_close = predictions["close"].astype(float)

actual_close = actual["close"].astype(float)


# ============================================================
# 8. CALCULATE METRICS
# ============================================================

errors = pred_close.values - actual_close.values

mae = np.mean(np.abs(errors))

rmse = np.sqrt(np.mean(errors ** 2))

mape = np.mean(
    np.abs(errors / actual_close.values)
) * 100


# Directional accuracy
#
# Compare whether Kronos correctly predicted the direction
# of movement relative to the previous day's actual close.

previous_close = train["close"].iloc[-1]

actual_direction = np.sign(
    actual_close.values - previous_close
)

predicted_direction = np.sign(
    pred_close.values - previous_close
)

directional_accuracy = np.mean(
    actual_direction == predicted_direction
) * 100


# ============================================================
# 9. PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("KRONOS JANUARY 2021 FORECAST RESULTS")
print("=" * 60)

print(f"MAE                  : ₹{mae:.2f}")
print(f"RMSE                 : ₹{rmse:.2f}")
print(f"MAPE                 : {mape:.2f}%")
print(f"Directional Accuracy : {directional_accuracy:.2f}%")

print("\nSample predictions:")
print("-" * 70)

for date, pred, real in zip(
    actual.index,
    pred_close.values,
    actual_close.values
):
    error = pred - real

    print(
        f"{date.date()} | "
        f"Actual: ₹{real:.2f} | "
        f"Kronos: ₹{pred:.2f} | "
        f"Error: ₹{error:.2f}"
    )


# ============================================================
# 10. SAVE PREDICTIONS
# ============================================================

results = pd.DataFrame({
    "Actual_Close": actual_close.values,
    "Kronos_Predicted_Close": pred_close.values,
})

results["Error"] = (
    results["Kronos_Predicted_Close"]
    - results["Actual_Close"]
)

results["Absolute_Error"] = results["Error"].abs()

results.index = actual.index
results.index.name = "Date"

results.to_csv(OUTPUT_CSV)

print(f"\nPrediction table saved to:")
print(OUTPUT_CSV)


# ============================================================
# 11. CREATE PLOT
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    actual.index,
    actual_close.values,
    label="Actual Close"
)

plt.plot(
    actual.index,
    pred_close.values,
    label="Kronos Prediction"
)

plt.title(
    "RELIANCE.NS — Kronos Forecast "
    "vs Actual, January 2021"
)

plt.xlabel("Date")
plt.ylabel("Price (₹)")

plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(OUTPUT_PLOT, dpi=200)

plt.close()

print("Forecast plot saved to:")
print(OUTPUT_PLOT)

print("\nExperiment complete.")