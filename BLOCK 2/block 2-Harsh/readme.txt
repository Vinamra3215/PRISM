PRISM BLOCK 2: DATA-DRIVEN SELECTION

Place prism_block2_selection.py in your Block 2 folder. All file paths are supplied
explicitly, so the folder name is not assumed. Use your precomputed 93-feature panel.

Run:
python prism_block2_selection.py --data YOUR_PANEL.parquet --manifest YOUR_MANIFEST.csv --category-policy prism_category_policy.csv --output-dir NEW_RESULTS_FOLDER

The manifest must list the 93 REAL dataset column names and their eight categories.
The supplied manifest template contains headers only because your actual 93 column
names were not supplied. Fill one row per candidate. Category names must match the
policy CSV. The policy uses the category labels and suggested ranges from the guide.
These are input metadata and configurable selection rules, not predetermined winners.
Simplicity is optional, numeric, smaller = simpler; use it only for close ICIR scores.
Optional Stationary values (Yes/No) apply your upstream stationarity results.
Without those values, stationarity is explicitly unverified; no stationarity test is invented.

Pipeline:
P1: Existing computed features are INPUTS; this script does not generate all 93 indicators.
P2: Compute real RankIC/ICIR on 2022-2023; screen by absolute ICIR and data coverage.
P3: Within-category Spearman correlation filtering above 0.7; retain higher absolute ICIR.
C1: Cross-category Spearman filtering above 0.7 with the same rule.
C2: LightGBM gain ranking, training on 2022 and validating on 2023.
C3: Combine ICIR and gain percentile ranks, enforce category coverage, select 19 by default.
Results: final CSV includes #, Feature, Category, RankIC, ICIR, LightGBM Rank and Why Selected.
Every metric, candidate count, selected name and reason is computed; no results are copied.

All dates, thresholds, selection counts, weights and model parameters are command-line
settings. --help lists them. The dates 2022-2023 are the guide's defaults, not fixed scores.
The guide conflicts on ICIR cutoff (0.1 in detailed section, 0.15 in summary).
Default is 0.15; use --icir-cutoff 0.1 if adopting the detailed section instead.
The guide does not specify a formula for combining ranks; default equal-weight
percentile consensus is explicit and configurable with --icir-weight.
The suggested intermediate counts (6-7/category, 30-35 total) are approximate.
This script does not retain statistically unsuitable candidates to force those counts.
If too few survivors satisfy category minima or the requested final size, it stops
with an explanation and preserves screening/ranking artifacts. Do not fabricate survivors.

MARKET FEATURES: important limitation in the guide
They have identical values across stocks on the same date, so daily cross-sectional
RankIC/ICIR is undefined. The script uses market time-series Spearman against the
universe-average next-day return. Market ICIR comes from rolling 20-date correlations
(--market-ic-window). This is a disclosed extension, not an exact rule supplied by
the guide; overlapping windows and different sampling mean it is not directly
comparable to stock daily cross-sectional ICIR. Confirm this convention with your team.

LABEL BOUNDARIES:
Prefer --label-date-column YOUR_ACTUAL_LABEL_DATE_COLUMN. Otherwise label maturity
is inferred as the next observed row per stock (horizon default one); this assumes
that this matches the supplied target's definition. Final missing inferred label
dates are excluded conservatively. Labels after selection-end are excluded. Training
labels after train-end are excluded. Inputs must be point-in-time and correctly
aligned; column names alone cannot prove that an upstream feature has no leakage.

ROLLING BLOCK 3:
The fixed 2022-2023 selection cannot be used as an independent selection for earlier
rolling folds. Repeat selection within each fold or begin downstream training after
the selection period. This script does not automatically integrate into Block 3.
The previous Block 3 loader has its own fixed market list; adapt it if this selector
chooses different market inputs.

Requires numpy, pandas, scipy, lightgbm and a Parquet reader such as pyarrow.
Syntax and synthetic screening/selection tests were performed here. Real training
was not run: your actual feature panel, mapping and LightGBM/SciPy runtime were not supplied.
