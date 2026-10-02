import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller

FEATURE_COLS = [
    'vol_weight_mom',
    'vol_adj_return',
    'mean_reversion',
    'trend_strength',
    'vix_adj_mom',
    'gap_vol_inter',
    'rsi_vol_diverg',
    'breakout_signal',
    'reversal_signal',
    'sector_mom_align',
]


def check_stationarity(series: pd.Series) -> bool:
  """Augmented Dickey-Fuller test. Returns True if stationary (p < 0.05)."""
  clean = series.dropna()
  if len(clean) < 50 or clean.nunique() <= 1:
    return False
  if len(clean) > 5000:
    clean = clean.sample(5000, random_state=42)
  p_val = adfuller(clean, autolag='AIC')[1]
  return bool(p_val < 0.05)


def main():
  print('[Step P2] Loading computed dataset...')
  df = pd.read_parquet('category8_computed_features.parquet')
  df['date'] = pd.to_datetime(df['date'])

  # Strict chronological slice: Jan 2022 to Dec 2023 ONLY[cite: 1]
  train_df = df[
      (df['date'] >= '2022-01-01') & (df['date'] <= '2023-12-31')
  ].copy()
  train_df = train_df.dropna(subset=['target_return_1d'])
  unique_dates = np.sort(train_df['date'].unique())
  print(
      f'[Step P2] Processing {len(train_df):,} rows over {len(unique_dates)}'
      ' trading days (2022-2023)...'
  )

  results = []
  for feat in FEATURE_COLS:
    print(f'-> Evaluating {feat}...')
    is_stationary = check_stationarity(train_df[feat])

    daily_ics = []
    for d in unique_dates:
      day_slice = train_df[train_df['date'] == d][[feat, 'target_return_1d']]
      valid_slice = day_slice.dropna()

      # Valid cross-sectional rank: min 15 stocks and non-zero variance
      if (
          len(valid_slice) >= 15
          and valid_slice[feat].nunique() > 1
          and valid_slice['target_return_1d'].nunique() > 1
      ):
        ic, _ = spearmanr(valid_slice[feat], valid_slice['target_return_1d'])
        if not np.isnan(ic):
          daily_ics.append(ic)

    daily_ics = np.array(daily_ics)
    if len(daily_ics) > 0:
      mean_ic = float(np.mean(daily_ics))
      std_ic = float(np.std(daily_ics, ddof=1))
      icir = mean_ic / (std_ic + 1e-9) if std_ic > 0 else 0.0
    else:
      mean_ic, icir = 0.0, 0.0

    # Verdict rules for Step P2
    if not is_stationary:
      verdict = 'DISCARD (Non-stationary)'
    elif abs(icir) < 0.07:
      verdict = 'DISCARD (Low ICIR)'
    else:
      verdict = 'KEEP'

    results.append({
        'Feature': feat,
        'Stationary?': 'Yes' if is_stationary else 'No',
        'RankIC': round(mean_ic, 4),
        'ICIR': round(icir, 4),
        'abs_ICIR': round(abs(icir), 4),
        'Verdict_P2': verdict,
    })

  summary_df = pd.DataFrame(results).sort_values(by='abs_ICIR', ascending=False)
  summary_df.to_csv('step_p2_results.csv', index=False)
  print('\n[Step P2 Results Table]')
  print(
      summary_df[
          ['Feature', 'Stationary?', 'RankIC', 'ICIR', 'Verdict_P2']
      ].to_string(index=False)
  )
  print('\n[Step P2 Complete] Results saved to: step_p2_results.csv\n')


if __name__ == '__main__':
  main()