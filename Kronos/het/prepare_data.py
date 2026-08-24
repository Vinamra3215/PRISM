import pandas as pd
import os

RAW_PATH = "data/RIL_19-26.parquet"
OUTPUT_PATH = "data/RIL_400bars.parquet"

# Load raw data
df = pd.read_parquet(RAW_PATH)

# Remove Yahoo Finance MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Standardize column names
df.columns = [str(col).lower() for col in df.columns]

# Make sure Date is the index
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Keep only the columns Kronos needs
df = df[["open", "high", "low", "close", "volume"]]

# Remove missing values
df = df.dropna()

# Everything strictly before 2021-01-01
before_2021 = df[df.index < "2021-01-01"]

# Take EXACTLY the final 400 bars
experiment_data = before_2021.tail(400)

# Save
experiment_data.to_parquet(OUTPUT_PATH)

print("========================================")
print("DATA PREPARATION COMPLETE")
print("========================================")
print(f"Total bars before 2021 : {len(before_2021)}")
print(f"Experiment bars        : {len(experiment_data)}")
print(f"Start                  : {experiment_data.index.min()}")
print(f"End                    : {experiment_data.index.max()}")
print(f"Columns                : {experiment_data.columns.tolist()}")
print(f"Saved to               : {OUTPUT_PATH}")