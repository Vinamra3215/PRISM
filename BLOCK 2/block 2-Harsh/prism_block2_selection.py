#!/usr/bin/env python3
"""PRISM P2/P3/C1/C2/C3 selection on precomputed features; no preset winners.

Feature names/categories come from a manifest. All scores come from the panel.
The guide's dates and rules are configurable experiment settings, not results.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True,
                   help='CSV: Feature, Category; optional Simplicity and Stationary')
    p.add_argument('--category-policy', type=Path, required=True,
                   help='CSV: Category, Min, Max; PRISM category coverage settings')
    p.add_argument('--output-dir', type=Path, default=Path('prism_block2_results'))
    p.add_argument('--target', default='target_return_1d')
    p.add_argument('--label-date-column', default=None)
    p.add_argument('--label-horizon-days', type=int, default=1)
    p.add_argument('--selection-start', default='2022-01-01')
    p.add_argument('--selection-end', default='2023-12-31')
    p.add_argument('--train-end', default='2022-12-31')
    p.add_argument('--validation-start', default='2023-01-01')
    p.add_argument('--expected-features', type=int, default=93)
    p.add_argument('--expected-categories', type=int, default=8)
    p.add_argument('--icir-cutoff', type=float, default=0.15)
    p.add_argument('--correlation-cutoff', type=float, default=0.7)
    p.add_argument('--icir-tie-tolerance', type=float, default=0.01,
                   help='Prefer simpler manifest feature within this absolute-ICIR difference')
    p.add_argument('--final-count', type=int, default=19)
    p.add_argument('--minimum-final-count', type=int, default=15)
    p.add_argument('--maximum-final-count', type=int, default=20)
    p.add_argument('--icir-weight', type=float, default=0.5,
                   help='Consensus weight; gain weight is 1 minus this value')
    p.add_argument('--minimum-gain-fraction', type=float, default=0.0)
    p.add_argument('--market-category', default='Market-Level')
    p.add_argument('--market-ic-window', type=int, default=20)
    p.add_argument('--minimum-stocks', type=int, default=20)
    p.add_argument('--minimum-ic-observations', type=int, default=20)
    p.add_argument('--minimum-ic-coverage', type=float, default=0.5)
    p.add_argument('--num-boost-round', type=int, default=300)
    p.add_argument('--learning-rate', type=float, default=0.03)
    p.add_argument('--num-leaves', type=int, default=15)
    p.add_argument('--max-depth', type=int, default=5)
    p.add_argument('--min-data-in-leaf', type=int, default=100)
    p.add_argument('--feature-fraction', type=float, default=0.8)
    p.add_argument('--bagging-fraction', type=float, default=0.8)
    p.add_argument('--lambda-l1', type=float, default=0.1)
    p.add_argument('--lambda-l2', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num-threads', type=int, default=4)
    a = p.parse_args()
    start, end, train_end, valid_start = map(pd.Timestamp,
        [a.selection_start, a.selection_end, a.train_end, a.validation_start])
    if not start <= train_end < valid_start <= end:
        p.error('Require selection-start <= train-end < validation-start <= selection-end')
    if not 0 < a.correlation_cutoff <= 1 or not 0 <= a.icir_weight <= 1:
        p.error('Invalid correlation cutoff or consensus weight')
    if not 0 < a.minimum_ic_coverage <= 1 or a.icir_cutoff < 0 or a.icir_tie_tolerance < 0:
        p.error('Invalid IC screening settings')
    if not 0 <= a.minimum_gain_fraction < 1:
        p.error('Invalid minimum gain fraction')
    if not 1 <= a.minimum_final_count <= a.final_count <= a.maximum_final_count:
        p.error('Final count must lie inside the configured deliverable range')
    if min(a.expected_features, a.expected_categories, a.minimum_stocks,
           a.minimum_ic_observations, a.market_ic_window, a.label_horizon_days,
           a.num_boost_round, a.num_threads) < 1:
        p.error('Counts must be positive')
    if a.market_ic_window < 3:
        p.error('Market IC window must contain at least three dates')
    return a


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    return value


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def load_schema(args, panel):
    manifest = pd.read_csv(args.manifest)
    policy = pd.read_csv(args.category_policy)
    if not {'Feature', 'Category'}.issubset(manifest) or not {'Category', 'Min', 'Max'}.issubset(policy):
        raise ValueError('Manifest needs Feature/Category; policy needs Category/Min/Max')
    if manifest[['Feature', 'Category']].isna().any().any() or policy.Category.isna().any():
        raise ValueError('Missing schema names')
    if manifest.Feature.duplicated().any() or policy.Category.duplicated().any():
        raise ValueError('Duplicate feature or policy category')
    if len(manifest) != args.expected_features or manifest.Category.nunique() != args.expected_categories:
        raise ValueError('Manifest count does not match the configured candidate/category counts')
    if set(manifest.Category) != set(policy.Category):
        raise ValueError('Manifest categories must exactly match the category-policy names')
    for column in ['Min', 'Max']:
        values = pd.to_numeric(policy[column], errors='raise')
        if values.isna().any() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError('Category bounds must be nonnegative integers')
        policy[column] = values.astype(int)
    if (policy.Min > policy.Max).any() or policy.Min.sum() > args.final_count or policy.Max.sum() < args.final_count:
        raise ValueError('Category policy cannot satisfy the requested final count')
    features = manifest.Feature.tolist()
    missing = [f for f in features if f not in panel]
    if missing:
        raise ValueError(f'Missing manifest features: {missing}')
    forbidden = {'date', 'symbol', args.target, args.label_date_column}
    if forbidden.intersection(features):
        raise ValueError('Identifiers/target/label dates cannot be candidates')
    if 'Simplicity' not in manifest:
        manifest['Simplicity'] = 0.0
    manifest.Simplicity = pd.to_numeric(manifest.Simplicity, errors='raise')
    if not np.isfinite(manifest.Simplicity).all():
        raise ValueError('Simplicity must be finite; smaller means simpler')
    if 'Stationary' in manifest:
        flags = manifest.Stationary.astype(str).str.strip().str.lower()
        if not flags.isin(['true', 'false', 'yes', 'no', '1', '0']).all():
            raise ValueError('Stationary must explicitly be Yes/No or True/False')
        manifest['stationary_eligible'] = flags.isin(['true', 'yes', '1'])
    else:
        manifest['stationary_eligible'] = True
    return manifest, policy


def prepare_panel(raw, args, features):
    df = raw.copy()
    if df.columns.duplicated().any() or not {'date', 'symbol', args.target}.issubset(df):
        raise ValueError('Missing required identifiers/target or duplicate columns')
    df.date = pd.to_datetime(df.date, errors='raise').dt.tz_localize(None)
    if df[['date', 'symbol']].isna().any().any() or df.duplicated(['date', 'symbol']).any():
        raise ValueError('Missing identifiers or duplicate date/symbol records')
    df = df.sort_values(['symbol', 'date'])
    # Only date metadata is used to establish label maturity; future target values
    # are never used in selection. Explicit actual label dates are preferred.
    if args.label_date_column:
        df['_label_date'] = pd.to_datetime(df[args.label_date_column], errors='raise').dt.tz_localize(None)
    else:
        df['_label_date'] = df.groupby('symbol').date.shift(-args.label_horizon_days)
    start, end = pd.Timestamp(args.selection_start), pd.Timestamp(args.selection_end)
    df = df[df.date.between(start, end) & df._label_date.notna()
            & (df._label_date > df.date) & (df._label_date <= end)].copy()
    if df.empty:
        raise ValueError('No calibration observations with labels inside selection boundaries')
    for feature in features:
        if not (pd.api.types.is_numeric_dtype(df[feature]) or pd.api.types.is_bool_dtype(df[feature])):
            raise ValueError(f'Candidate is not numeric: {feature}')
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).astype('float32')
    df['_target'] = pd.to_numeric(df[args.target], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return df[df._target.notna()].sort_values(['date', 'symbol']).reset_index(drop=True)


def feature_statistics(df, feature, is_market, args):
    values = []
    if is_market:
        counts = df.groupby('date')[feature].nunique(dropna=True)
        if (counts > 1).any():
            raise ValueError(f'Market candidate {feature} has conflicting values on the same date')
        daily = df.groupby('date').agg(value=(feature, 'first'), target=('_target', 'mean')).dropna()
        rank_ic = (float(spearmanr(daily.value, daily.target).statistic)
                   if len(daily) >= args.minimum_ic_observations and daily.value.nunique() > 1
                   and daily.target.nunique() > 1 else np.nan)
        for stop in range(args.market_ic_window, len(daily)+1):
            window = daily.iloc[stop-args.market_ic_window:stop]
            if window.value.nunique() > 1 and window.target.nunique() > 1:
                rho = spearmanr(window.value, window.target).statistic
                if np.isfinite(rho):
                    values.append(float(rho))
        possible = max(0, df.date.nunique()-args.market_ic_window+1)
        method = 'market time-series Spearman; ICIR from overlapping rolling-window correlations'
    else:
        for _, day in df.groupby('date', sort=True):
            pairs = day[[feature, '_target']].dropna()
            if len(pairs) >= args.minimum_stocks and pairs[feature].nunique() > 1 and pairs._target.nunique() > 1:
                rho = spearmanr(pairs[feature], pairs._target).statistic
                if np.isfinite(rho):
                    values.append(float(rho))
        rank_ic = float(np.mean(values)) if values else np.nan
        possible = df.date.nunique()
        method = 'daily cross-sectional Spearman'
    std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
    icir = float(np.mean(values))/std if np.isfinite(std) and std > 0 else np.nan
    coverage = len(values)/possible if possible else 0.0
    return {'Feature': feature, 'RankIC': rank_ic, 'ICIR': icir, 'AbsICIR': abs(icir),
            'IC Observations': len(values), 'IC Coverage': coverage, 'IC Method': method}


def prefer_feature(a, b, evidence, tolerance):
    difference = evidence.loc[a, 'AbsICIR']-evidence.loc[b, 'AbsICIR']
    if abs(difference) > tolerance:
        return a if difference > 0 else b
    # A manifest supplies interpretability, rather than inventing it from names.
    if evidence.loc[a, 'Simplicity'] != evidence.loc[b, 'Simplicity']:
        return a if evidence.loc[a, 'Simplicity'] < evidence.loc[b, 'Simplicity'] else b
    if difference != 0:
        return a if difference > 0 else b
    return min(a, b)  # Deterministic tie, not a fixed winning feature.


def prune_correlations(features, evidence, matrix, cutoff, tolerance, within_category):
    remaining = set(features)
    drops = []
    while True:
        pairs = []
        ordered = sorted(remaining)
        for i, a in enumerate(ordered):
            for b in ordered[i+1:]:
                same_category = evidence.loc[a, 'Category'] == evidence.loc[b, 'Category']
                if same_category != within_category:
                    continue
                corr = matrix.loc[a, b]
                if np.isfinite(corr) and abs(corr) > cutoff:
                    pairs.append((abs(float(corr)), a, b, float(corr)))
        if not pairs:
            break
        _, a, b, corr = sorted(pairs, key=lambda x: (-x[0], x[1], x[2]))[0]
        keep = prefer_feature(a, b, evidence, tolerance)
        drop = b if keep == a else a
        remaining.remove(drop)
        drops.append({'Feature': drop, 'Kept Feature': keep, 'Spearman Correlation': corr,
                      'Reason': 'Lower absolute ICIR, or manifest simplicity for close ICIR'})
    return sorted(remaining), drops


def balanced_selection(ranking, policy, count):
    ordered = ranking.sort_values(['Consensus Score', 'Importance Gain', 'AbsICIR', 'Feature'],
                                  ascending=[False, False, False, True])
    selected = []
    limits = policy.set_index('Category')
    for category in sorted(limits.index):
        candidates = ordered[ordered.Category == category].Feature.tolist()
        minimum = int(limits.loc[category, 'Min'])
        if len(candidates) < minimum:
            raise ValueError(f'Only {len(candidates)} qualifying features in {category}; policy needs {minimum}. '
                             'Inspect measured screening results; no names or scores will be fabricated.')
        selected.extend(candidates[:minimum])
    counts = ordered.set_index('Feature').loc[selected].Category.value_counts().to_dict()
    for row in ordered.itertuples(index=False):
        if len(selected) >= count:
            break
        if row.Feature not in selected and counts.get(row.Category, 0) < int(limits.loc[row.Category, 'Max']):
            selected.append(row.Feature)
            counts[row.Category] = counts.get(row.Category, 0)+1
    if len(selected) != count:
        raise ValueError(f'Only {len(selected)} qualifying features satisfy category limits; requested {count}')
    return ordered[ordered.Feature.isin(selected)].copy()


def prediction_metrics(actual, predicted, dates, minimum_stocks):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if not np.isfinite(predicted).all():
        raise ValueError('Nonfinite LightGBM predictions')
    residual = actual-predicted
    values = []
    frame = pd.DataFrame({'date': np.asarray(dates), 'actual': actual, 'prediction': predicted})
    for _, day in frame.groupby('date'):
        if len(day) >= minimum_stocks and day.actual.nunique() > 1 and day.prediction.nunique() > 1:
            rho = spearmanr(day.actual, day.prediction).statistic
            if np.isfinite(rho):
                values.append(float(rho))
    std = np.std(values, ddof=1) if len(values) > 1 else np.nan
    return {'RMSE': float(np.sqrt(np.mean(residual**2))), 'MAE': float(np.mean(np.abs(residual))),
            'Mean Daily RankIC': float(np.mean(values)) if values else np.nan,
            'Daily ICIR': float(np.mean(values))/std if np.isfinite(std) and std > 0 else np.nan,
            'Valid IC Days': len(values), 'Rows': len(actual)}


def markdown_table(frame):
    display = frame.copy().fillna('—')
    lines = ['| '+' | '.join(map(str, display.columns))+' |',
             '| '+' | '.join(['---']*len(display.columns))+' |']
    for row in display.itertuples(index=False, name=None):
        lines.append('| '+' | '.join(str(v).replace('|', '\\|') for v in row)+' |')
    return '\n'.join(lines)


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(args.data)
    manifest, policy = load_schema(args, raw)
    features = manifest.Feature.tolist()
    df = prepare_panel(raw, args, features)
    del raw
    write_json(args.output_dir/'experiment_settings.json', vars(args))
    manifest.to_csv(args.output_dir/'candidate_manifest_used.csv', index=False)
    rows = [feature_statistics(df, f, manifest.set_index('Feature').loc[f, 'Category'] == args.market_category, args)
            for f in features]
    evidence = pd.DataFrame(rows).merge(manifest, on='Feature', validate='one_to_one')
    evidence['P2 Eligible'] = (evidence.stationary_eligible & np.isfinite(evidence.ICIR)
        & (evidence.AbsICIR >= args.icir_cutoff)
        & (evidence['IC Observations'] >= args.minimum_ic_observations)
        & (evidence['IC Coverage'] >= args.minimum_ic_coverage))
    evidence['P2 Reason'] = evidence.apply(lambda row:
        ('Non-stationary flag from manifest' if not row.stationary_eligible else
         'Undefined ICIR' if not np.isfinite(row.ICIR) else
         'Insufficient IC observations/coverage' if row['IC Observations'] < args.minimum_ic_observations
            or row['IC Coverage'] < args.minimum_ic_coverage else
         'Below absolute ICIR cutoff' if row.AbsICIR < args.icir_cutoff else 'KEEP'), axis=1)
    evidence.to_csv(args.output_dir/'p2_feature_statistics.csv', index=False)
    eligible = evidence.loc[evidence['P2 Eligible'], 'Feature'].tolist()
    if not eligible:
        raise ValueError('No features passed P2; inspect computed statistics')
    lookup = evidence.set_index('Feature')
    matrix = df[eligible].corr(method='spearman', min_periods=args.minimum_ic_observations)
    matrix.to_csv(args.output_dir/'p3_c1_spearman_matrix.csv')
    phase1, p3_drops = prune_correlations(eligible, lookup, matrix, args.correlation_cutoff,
                                         args.icir_tie_tolerance, True)
    surviving, c1_drops = prune_correlations(phase1, lookup, matrix, args.correlation_cutoff,
                                            args.icir_tie_tolerance, False)
    pd.DataFrame(p3_drops, columns=['Feature', 'Kept Feature', 'Spearman Correlation', 'Reason']).to_csv(
        args.output_dir/'p3_within_category_drops.csv', index=False)
    pd.DataFrame(c1_drops, columns=['Feature', 'Kept Feature', 'Spearman Correlation', 'Reason']).to_csv(
        args.output_dir/'c1_cross_category_drops.csv', index=False)
    evidence['P3 Survived'] = evidence.Feature.isin(phase1)
    evidence['C1 Survived'] = evidence.Feature.isin(surviving)
    evidence.to_csv(args.output_dir/'phase1_c1_screening_summary.csv', index=False)
    if not surviving:
        raise ValueError('No candidates remain after correlation filtering')
    train_end, valid_start = pd.Timestamp(args.train_end), pd.Timestamp(args.validation_start)
    train = df[(df.date <= train_end) & (df._label_date <= train_end)]
    valid = df[df.date >= valid_start]
    if train.empty or valid.empty:
        raise ValueError('Empty chronological LightGBM training/validation period')
    params = {'objective': 'regression', 'metric': 'rmse', 'learning_rate': args.learning_rate,
              'num_leaves': args.num_leaves, 'max_depth': args.max_depth,
              'min_data_in_leaf': args.min_data_in_leaf, 'feature_fraction': args.feature_fraction,
              'bagging_fraction': args.bagging_fraction, 'bagging_freq': 1,
              'lambda_l1': args.lambda_l1, 'lambda_l2': args.lambda_l2, 'verbosity': -1,
              'seed': args.seed, 'feature_fraction_seed': args.seed, 'bagging_seed': args.seed,
              'num_threads': args.num_threads, 'force_col_wise': True}
    training = lgb.Dataset(train[surviving], train._target, feature_name=surviving)
    validation = lgb.Dataset(valid[surviving], valid._target, reference=training)
    model = lgb.train(params, training, num_boost_round=args.num_boost_round,
                      valid_sets=[validation], valid_names=['chronological_validation'],
                      callbacks=[lgb.log_evaluation(0)])
    model.save_model(str(args.output_dir/'c2_ranking_model.txt'))
    ranking = lookup.loc[surviving].reset_index()
    ranking['Importance Gain'] = model.feature_importance('gain')
    ranking['Importance Split'] = model.feature_importance('split')
    ranking = ranking.sort_values(['Importance Gain', 'Feature'], ascending=[False, True]).reset_index(drop=True)
    ranking['LightGBM Rank'] = np.arange(1, len(ranking)+1)
    gain_sum = ranking['Importance Gain'].sum()
    ranking['Gain Fraction'] = ranking['Importance Gain']/gain_sum if gain_sum > 0 else 0.0
    # Consensus is an explicit configurable implementation of C3's two rankings.
    ranking['ICIR Rank'] = ranking.AbsICIR.rank(method='min', ascending=False)
    ranking['Consensus Score'] = (args.icir_weight*ranking.AbsICIR.rank(pct=True)
        + (1-args.icir_weight)*ranking['Importance Gain'].rank(pct=True))
    ranking.to_csv(args.output_dir/'c2_lightgbm_feature_importance.csv', index=False)
    predictions = model.predict(valid[surviving])
    metrics = prediction_metrics(valid._target, predictions, valid.date, args.minimum_stocks)
    baseline = prediction_metrics(valid._target, np.repeat(train._target.mean(), len(valid)), valid.date, args.minimum_stocks)
    write_json(args.output_dir/'c2_validation_metrics.json', {
        'purpose': 'feature-ranking model; validation participated in feature screening, not an untouched test',
        'model': metrics, 'training_mean_baseline': baseline, 'parameters': params,
        'training_dates': [train.date.min(), train.date.max()],
        'validation_dates': [valid.date.min(), valid.date.max()]})
    pd.DataFrame({'date': valid.date, 'symbol': valid.symbol, 'actual_return': valid._target,
                  'predicted_return': predictions}).to_csv(args.output_dir/'c2_validation_predictions.csv', index=False)
    qualifying = ranking[(ranking['Importance Gain'] > 0)
                         & (ranking['Gain Fraction'] >= args.minimum_gain_fraction)]
    selected = balanced_selection(qualifying, policy, args.final_count)
    selected['Why Selected'] = selected.apply(lambda r:
        f"|ICIR|={r.AbsICIR:.6g}; gain rank={int(r['LightGBM Rank'])}; "
        f"consensus={r['Consensus Score']:.6g}; retained under {r.Category} coverage policy", axis=1)
    selected.insert(0, '#', np.arange(1, len(selected)+1))
    columns = ['#', 'Feature', 'Category', 'RankIC', 'ICIR', 'LightGBM Rank', 'Why Selected',
               'Importance Gain', 'Consensus Score', 'IC Method']
    selected[columns].to_csv(args.output_dir/'c3_final_feature_set.csv', index=False)
    counts = selected.Category.value_counts().rename_axis('Category').reset_index(name='Selected Features')
    report = ['# PRISM Block 2 Feature Selection', '',
        f"Computed statistics on {df.date.min().date()}–{df.date.max().date()}, using only labels dated "
        f"through {args.selection_end}. Counts: candidates={len(features)}, P2={len(eligible)}, "
        f"P3={len(phase1)}, C1={len(surviving)}, C3={len(selected)}.", '',
        '## Final selected features', '', markdown_table(selected[columns]), '',
        '## Category distribution', '', markdown_table(counts), '',
        '## Calculated validation metrics', '', markdown_table(pd.DataFrame([metrics])), '',
        f"P2 cutoff: |ICIR| >= {args.icir_cutoff}. P3/C1 remove Spearman redundancy above "
        f"{args.correlation_cutoff}, using absolute ICIR and supplied simplicity for close scores. "
        f"C3 consensus weight on ICIR: {args.icir_weight}; the remaining weight is on gain rank.", '',
        'Market features use time-series Spearman against the daily universe-average next-day return. '
        'Their ICIR uses overlapping rolling correlations, so it is not directly equivalent to daily '
        'cross-sectional ICIR. This is an explicit extension because the guide does not define a '
        'valid cross-sectional market IC procedure.', '',
        'P1 feature computation is upstream. This script consumes a complete feature panel and manifest. '
        'Stationarity flags are applied when supplied; no stationarity test or guarantee is invented. '
        'The guide gives approximate intermediate counts; actual counts are reported, never forced.', '',
        'No test-period returns are used to choose features. No independent test result is claimed. '
        'Using this fixed selection in earlier rolling Block 3 folds would introduce future information; '
        'repeat selection within each fold or begin downstream training after this selection period.']
    (args.output_dir/'c3_final_feature_set.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(f"Selected {len(selected)} calculated winners from {len(features)} candidates: {args.output_dir}")


if __name__ == '__main__':
    main()
