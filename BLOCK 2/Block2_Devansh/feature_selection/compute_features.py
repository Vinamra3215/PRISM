import numpy as np
import pandas as pd


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
  """Vectorized Wilder's Exponential Relative Strength Index (RSI)."""
  delta = series.diff()
  gain = delta.clip(lower=0)
  loss = -delta.clip(upper=0)
  avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
  avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
  rs = avg_gain / (avg_loss + 1e-9)
  return 100.0 - (100.0 / (1.0 + rs))


def main():
  print(
      "[Step P1] Loading raw price data (Jan 2022 -> Present)..."
  )  #
  df = pd.read_parquet("nifty50_raw_data.parquet")
  df["date"] = pd.to_datetime(df["date"])
  df = df.sort_values(by=["symbol", "date"]).reset_index(drop=True)

  print("[Step P1] Computing Base Technical Indicators per stock...")
  grouped = df.groupby("symbol", group_keys=False)

  # Target: 1-day forward return (Close[t+1] / Close[t] - 1) for tomorrow's prediction
  df["target_return_1d"] = grouped["close"].shift(-1) / df["close"] - 1.0

  # Returns & Volatility
  df["return_1d"] = grouped["close"].pct_change()
  df["return_5d"] = grouped["close"].pct_change(5)
  df["return_20d"] = grouped["close"].pct_change(20)
  df["volatility_20d"] = grouped["return_1d"].transform(
      lambda s: s.rolling(20, min_periods=20).std()
  )

  # Volume Ratios & Moving Averages
  df["volume_sma20"] = grouped["volume"].transform(
      lambda s: s.rolling(20, min_periods=20).mean()
  )
  df["volume_ratio"] = df["volume"] / (df["volume_sma20"] + 1e-9)
  df["ma_50"] = grouped["close"].transform(
      lambda s: s.rolling(50, min_periods=50).mean()
  )
  df["ma_ratio"] = df["close"] / (df["ma_50"] + 1e-9)

  # Gaps & Candlestick Lower Shadow Ratio
  df["prev_close"] = grouped["close"].shift(1)
  df["gap"] = (df["open"] - df["prev_close"]) / (df["prev_close"] + 1e-9)
  candle_range = df["high"] - df["low"]
  lower_shadow = np.minimum(df["open"], df["close"]) - df["low"]
  df["lower_shadow_ratio"] = np.where(
      candle_range > 0, lower_shadow / candle_range, 0.0
  )

  # Bollinger Bands & RSI
  bb_mean = grouped["close"].transform(
      lambda s: s.rolling(20, min_periods=20).mean()
  )
  bb_std = grouped["close"].transform(
      lambda s: s.rolling(20, min_periods=20).std()
  )
  df["bollinger_upper"] = bb_mean + (2.0 * bb_std)
  df["rsi"] = grouped["close"].transform(calculate_rsi)
  df["rsi_direction"] = np.sign(grouped["rsi"].diff()).fillna(0)
  df["vol_direction"] = np.sign(grouped["volume"].diff()).fillna(0)

  # Sector momentum alignment
  sector_avg_ret20 = df.groupby(["sector", "date"])["return_20d"].transform(
      "mean"
  )
  df["stock_momentum_dir"] = np.sign(df["return_20d"]).fillna(0)
  df["sector_momentum_dir"] = np.sign(sector_avg_ret20).fillna(0)

  print("[Step P1] Generating 10 Category 8 Derived Features...")
  df["vol_weight_mom"] = (
      df["return_20d"] * df["volume_ratio"]
  )  #[cite: 2] 1. Volume-Weighted Momentum
  df["vol_adj_return"] = df["return_20d"] / (
      df["volatility_20d"] + 1e-9
  )  #[cite: 2] 2. Volatility-Adjusted Return
  df["mean_reversion"] = -df["return_5d"] / (
      df["volatility_20d"] + 1e-9
  )  #[cite: 2] 3. Mean Reversion Signal
  df["trend_strength"] = (
      np.abs(df["ma_ratio"] - 1.0) * df["volume_ratio"]
  )  #[cite: 2] 4. Trend Strength
  df["vix_adj_mom"] = df["return_20d"] / (
      df["vix"] + 1e-9
  )  #[cite: 2] 5. VIX-Adjusted Momentum
  df["gap_vol_inter"] = (
      df["gap"] * df["volume_ratio"]
  )  #[cite: 2] 6. Gap-Volume Interaction
  df["rsi_vol_diverg"] = (
      df["rsi_direction"] - df["vol_direction"]
  )  #[cite: 2] 7. RSI-Volume Divergence
  df["breakout_signal"] = (
      (df["close"] > df["bollinger_upper"]) & (df["volume_ratio"] > 1.5)
  ).astype(
      float
  )  #[cite: 2] 8. Breakout Signal
  df["reversal_signal"] = (
      (df["rsi"] < 30.0) & (df["lower_shadow_ratio"] > 0.5)
  ).astype(
      float
  )  #[cite: 2] 9. Reversal Signal
  df["sector_mom_align"] = np.where(
      df["stock_momentum_dir"] == df["sector_momentum_dir"], 1.0, -1.0
  )  #[cite: 2] 10. Sector Momentum Alignment

  output_parquet = "category8_computed_features.parquet"
  df.to_parquet(output_parquet, engine="pyarrow", index=False)
  print(
      f"[Step P1 Complete] Generated {len(df):,} rows saved to"
      f" {output_parquet}\n"
  )


if __name__ == "__main__":
  main()