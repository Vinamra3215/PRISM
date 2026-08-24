import pandas as pd
import numpy as np


# ==========================================
# 1. LOAD NIFTY DATA
# ==========================================

file_path = "/home/soq/NIFTY50_5Y_OHLCV.parquet"

df = pd.read_parquet(file_path)

print("Data loaded successfully!")
print("Original columns:")
print(df.columns)


# ==========================================
# 2. PREPARE DATE
# ==========================================

df.index = pd.to_datetime(df.index)
df = df.sort_index()


# ==========================================
# 3. GET NIFTY CLOSE PRICE
# ==========================================

close = df[("Close", "^NSEI")]


# ==========================================
# 4. CALCULATE LOG RETURN
# ==========================================

df["Log_Return"] = np.log(
    close / close.shift(1)
)


# ==========================================
# 5. CALCULATE ROLLING VOLATILITY
# ==========================================

df["Volatility"] = (
    df["Log_Return"]
    .rolling(window=20)
    .std()
)


# ==========================================
# 6. REMOVE FIRST ROWS WITH NaN
# ==========================================

df = df.dropna()

# ==========================================
# 7. ASK FOR DATE
# ==========================================

date_input = input(
    "\nEnter date (YYYY-MM-DD): "
)

try:
    selected_date = pd.to_datetime(date_input)

except:
    print("Invalid date format!")
    exit()


# ==========================================
# 8. CHECK DATE
# ==========================================

if selected_date not in df.index:

    print("\nDate not available in dataset.")
    print("Please enter a trading day.")

    print(
        "\nAvailable range:",
        df.index.min().date(),
        "to",
        df.index.max().date()
    )

    exit()


# ==========================================
# 9. GET SELECTED DAY
# ==========================================

row = df.loc[selected_date]


# ==========================================
# 10. PRINT OHLCV
# ==========================================

print("\n================================")
print("NIFTY OHLCV")
print("================================")

print("Date   :", selected_date.date())

print(
    "Open   :",
    row[("Open", "^NSEI")]
)

print(
    "High   :",
    row[("High", "^NSEI")]
)

print(
    "Low    :",
    row[("Low", "^NSEI")]
)

print(
    "Close  :",
    row[("Close", "^NSEI")]
)

print(
    "Volume :",
    row[("Volume", "^NSEI")]
)


# ==========================================
# 11. PRINT FEATURES
# ==========================================

print("\n================================")
print("FEATURES")
print("================================")

print(
    "Log Return :",
    row["Log_Return"]
)

print(
    "Volatility :",
    row["Volatility"]
)


# ==========================================
# 12. CREATE FEATURE VECTOR
# ==========================================

feature_vector = np.array([
    row[("Close", "^NSEI")],
    row["Log_Return"],
    row["Volatility"]
])


# ==========================================
# 13. PRINT FEATURE VECTOR
# ==========================================

print("\n================================")
print("FEATURE VECTOR")
print("================================")

print(feature_vector)

print("\nFeature Names:")
print([
    "Price",
    "Log_Return",
    "Volatility"
])