import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path("/home/soq/__shutupandbendover/tanishq")
CALIB_PATH = Path("/home/soq/BLOCK 2/phase2_tanishq/nifty50_phase2_all_features_full.parquet")
OUTPUT_DIR = DATA_DIR / "phase3_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Assembling structured features and macro indicators...")
df = pd.read_parquet(CALIB_PATH)
df['date'] = pd.to_datetime(df['date'])

# 1. Segregate Stock Features vs. Macro Features
market_cols = [
    'nifty_daily_return', 'nifty_5d_return', 'nifty_vol_20d', 
    'pharma_return'
]

stock_cols = [
    'ma_ratio_50_200', 'rolling_volatility_10d', 'intraday_range',
    'keltner_channel_position', 'trend_strength', 'cs_return_z',
    'volume_price_trend', 'volatility_ratio_5d_20d', 'sector_relative_return',
    'rolling_skewness_20d'
]

# 2. Daily Cross-Sectional Z-Score for Stock Features
print("Computing daily cross-sectional z-score for stock features...")
for col in stock_cols:
    df[col] = df.groupby('date')[col].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))

# 3. Rolling 60-Day Time-Series Z-Score for Market Features
print("Computing rolling 60-day z-score for market indicators...")
market_df = df[['date'] + market_cols].drop_duplicates('date').sort_values('date').set_index('date')
for col in market_cols:
    rolling_mean = market_df[col].rolling(window=60, min_periods=20).mean()
    rolling_std = market_df[col].rolling(window=60, min_periods=20).std() + 1e-8
    market_df[col] = (market_df[col] - rolling_mean) / rolling_std

# Merge normalized market metrics back
df = df.drop(columns=market_cols).merge(market_df.reset_index(), on='date', how='left')

# Save prepared tabular matrix
matrix_path = OUTPUT_DIR / "prepared_features_matrix.parquet"
df.to_parquet(matrix_path, index=False)
print(f"Saved matrix: {df.shape} -> {matrix_path}")