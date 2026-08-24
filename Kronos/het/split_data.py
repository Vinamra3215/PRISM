import pandas as pd
import os

INPUT_PATH = "data/RIL_400bars.parquet"

TRAIN_PATH = "data/train.parquet"
VAL_PATH = "data/validation.parquet"

# Load the 400-bar experiment dataset
df = pd.read_parquet(INPUT_PATH)

# Make sure chronological order is preserved
df.index = pd.to_datetime(df.index)
df = df.sort_index()

# Safety check
if len(df) != 400:
    raise ValueError(f"Expected exactly 400 bars, found {len(df)}")

# Chronological split
train = df.iloc[:320].copy()
validation = df.iloc[320:].copy()

# Save
train.to_parquet(TRAIN_PATH)
validation.to_parquet(VAL_PATH)

print("========================================")
print("TRAIN / VALIDATION SPLIT")
print("========================================")

print(f"Total bars      : {len(df)}")

print("\nTRAINING SET")
print(f"Bars            : {len(train)}")
print(f"Start           : {train.index.min()}")
print(f"End             : {train.index.max()}")

print("\nVALIDATION SET")
print(f"Bars            : {len(validation)}")
print(f"Start           : {validation.index.min()}")
print(f"End             : {validation.index.max()}")

print("\nSaved:")
print(TRAIN_PATH)
print(VAL_PATH)