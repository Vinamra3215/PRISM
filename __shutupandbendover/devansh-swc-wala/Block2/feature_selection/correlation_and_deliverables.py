import json
import numpy as np
import pandas as pd

DIVIDER = '=' * 160
SUBDIVIDER = '-' * 160

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


def main():
  print('[Step P3] Loading Phase 1 data...')
  p2_df = pd.read_csv('step_p2_results.csv')

  if 'abs_ICIR' not in p2_df.columns:
    p2_df['abs_ICIR'] = p2_df['ICIR'].abs()

  icir_lookup = dict(zip(p2_df['Feature'], p2_df['abs_ICIR']))
  p2_verdict_lookup = dict(zip(p2_df['Feature'], p2_df['Verdict_P2']))

  # 1. STRICT FILTER: Only consider features that PASSED Step P2
  surviving_p2 = p2_df[p2_df['Verdict_P2'] == 'KEEP']['Feature'].tolist()
  dropped_in_p2 = p2_df[p2_df['Verdict_P2'] != 'KEEP']['Feature'].tolist()

  print(
      f'[Step P2 Carry-Over] {len(surviving_p2)} features passed Step P2:'
      f' {surviving_p2}'
  )
  print(
      f'[Step P2 Carry-Over] {len(dropped_in_p2)} features previously discarded'
      f' in Step P2: {dropped_in_p2}'
  )

  df = pd.read_parquet('category8_computed_features.parquet')
  df['date'] = pd.to_datetime(df['date'])

  # Strict chronological slice: Jan 2022 to Dec 2023 ONLY[cite: 1]
  train_df = df[
      (df['date'] >= '2022-01-01') & (df['date'] <= '2023-12-31')
  ].copy()

  # 2. Compute full pairwise Spearman correlation on all features for full visibility
  clean_stacked_all = train_df[FEATURE_COLS].dropna()
  corr_matrix = clean_stacked_all.corr(method='spearman')

  # Print SPEARMAN CORRELATION MATRIX Table
  print('\n' + DIVIDER)
  print('SPEARMAN CORRELATION MATRIX')
  print(DIVIDER)

  short_col_headers = [c[:8] for c in FEATURE_COLS]
  header_line = f"{'Feature':<20} " + ' '.join(
      f'{c:>8}' for c in short_col_headers
  )
  print(f'\n{header_line}')
  print(SUBDIVIDER)

  for feat in FEATURE_COLS:
    row_vals = ' '.join(f'{corr_matrix.loc[feat, c]:>8.3f}' for c in FEATURE_COLS)
    print(f'{feat:<20} {row_vals}')

  # 3. Detect and Print High-Correlation Pairs (|corr| > 0.70)
  print('\n' + SUBDIVIDER)
  print('HIGH-CORRELATION PAIRS |corr| > 0.70')
  print(SUBDIVIDER + '\n')

  for i in range(len(FEATURE_COLS)):
    for j in range(i + 1, len(FEATURE_COLS)):
      f1 = FEATURE_COLS[i]
      f2 = FEATURE_COLS[j]
      coeff = corr_matrix.loc[f1, f2]
      if abs(coeff) > 0.70:
        print(f'{f1:<20} <-> {f2:<20} corr =  {coeff:.6f}')

  # 4. Redundancy Filtering: Run ONLY on surviving_p2 sorted by highest |ICIR|
  sorted_surviving_p2 = sorted(
      surviving_p2, key=lambda f: icir_lookup.get(f, 0.0), reverse=True
  )

  kept_features = []
  dropped_correlated_map = {}

  for feat in sorted_surviving_p2:
    redundant = False
    for accepted in kept_features:
      coeff = abs(corr_matrix.loc[feat, accepted])
      if coeff > 0.70:
        redundant = True
        dropped_correlated_map[feat] = (
            f'{accepted} (corr={coeff:.2f}, kept higher |ICIR|'
            f' {icir_lookup[accepted]:.4f} vs {icir_lookup[feat]:.4f})'
        )
        break

    if not redundant:
      kept_features.append(feat)

  # 5. Print Result Box
  print('\n' + DIVIDER)
  print(
      f'RESULT: {len(FEATURE_COLS)} original -> {len(surviving_p2)} after Step'
      f' P2 -> {len(kept_features)} final survivors'
  )
  print(
      'Selection basis: Discard Step P2 failures + Higher |ICIR| kept between'
      ' redundant pairs'
  )
  print('Threshold: |corr| > 0.70')
  print(DIVIDER + '\n')

  # 6. Print FINAL PHASE 1 FEATURE SELECTION Table
  print(DIVIDER)
  print('FINAL PHASE 1 FEATURE SELECTION')
  print(DIVIDER)
  print(
      f"\n{'Feature':<20} {'|ICIR|':<10} {'Dropped Correlated With':<65}"
      f" {'Verdict':<25}"
  )
  print(SUBDIVIDER)

  for feat in FEATURE_COLS:
    score = f'{icir_lookup.get(feat, 0.0):.4f}'
    p2_verdict = p2_verdict_lookup.get(feat, '')

    if p2_verdict != 'KEEP':
      # Retain the exact failure reason from Step P2
      dropped_info = '—'
      verdict = p2_verdict
    elif feat in kept_features:
      dropped_info = '—'
      verdict = 'KEEP'
    else:
      dropped_info = dropped_correlated_map.get(feat, '')
      verdict = 'DROP - CORRELATED'

    print(f'{feat:<20} {score:<10} {dropped_info:<65} {verdict:<25}')

  # 7. Print FINAL FEATURES List
  print('\n' + SUBDIVIDER)
  print('FINAL FEATURES')
  print(SUBDIVIDER + '\n')
  for idx, f in enumerate(sorted(kept_features), 1):
    print(f'{idx}. {f} (|ICIR| = {icir_lookup.get(f, 0.0):.4f})')

  print(f'\nTotal features: {len(FEATURE_COLS)}')
  print(f'Final kept features: {len(kept_features)}')
  print(f'Dropped features: {len(FEATURE_COLS) - len(kept_features)}')
  print(DIVIDER)

  with open('category8_selected_features.json', 'w') as f:
    json.dump(kept_features, f, indent=4)
  print('\n[Step P3 Complete] Saved survivors to: category8_selected_features.json')


if __name__ == '__main__':
  main()