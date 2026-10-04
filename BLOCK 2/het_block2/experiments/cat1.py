import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "nifty50_stocks_ohlcv.parquet"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "p1_cat1"
)

SELECTION_START = "2022-01-01"
SELECTION_END = "2023-12-31"

CORRELATION_THRESHOLD = 0.70
TARGET_FEATURES = 7


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["symbol", "date"]).reset_index(drop=True)

print(f"Loaded data: {df.shape}")
print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
print(f"Stocks: {df['symbol'].nunique()}")


# ============================================================
# COMPUTE NEXT-DAY RETURN
# ============================================================

df["next_day_return"] = (
    df.groupby("symbol")["close"]
    .shift(-1)
    / df["close"]
    - 1
)


# ============================================================
# CATEGORY 1 FEATURES
# ============================================================

# 1. Log Return (1-day)
df["log_return_1d"] = (
    df.groupby("symbol")["close"]
    .transform(lambda x: np.log(x / x.shift(1)))
)


# 2. Return 5-day
df["return_5d"] = (
    df.groupby("symbol")["close"]
    .transform(lambda x: x / x.shift(5) - 1)
)


# 3. Return 10-day
df["return_10d"] = (
    df.groupby("symbol")["close"]
    .transform(lambda x: x / x.shift(10) - 1)
)


# 4. Return 20-day
df["return_20d"] = (
    df.groupby("symbol")["close"]
    .transform(lambda x: x / x.shift(20) - 1)
)


# 5. Return 60-day
df["return_60d"] = (
    df.groupby("symbol")["close"]
    .transform(lambda x: x / x.shift(60) - 1)
)


# 6. Return 120-day
df["return_120d"] = (
    df.groupby("symbol")["close"]
    .transform(lambda x: x / x.shift(120) - 1)
)


# 7. Intraday Range
df["intraday_range"] = (
    (df["high"] - df["low"]) / df["close"]
)


# 8. Open-Close Spread
df["open_close_spread"] = (
    (df["close"] - df["open"]) / df["open"]
)


# 9. Gap / Overnight Return
df["gap_overnight_return"] = (
    df["open"]
    / df.groupby("symbol")["close"].shift(1)
    - 1
)


# 10. Close-to-High Distance
df["close_to_high_distance"] = (
    (df["high"] - df["close"]) / df["close"]
)


# 11. 52-Week High Distance
rolling_high = (
    df.groupby("symbol")["high"]
    .transform(lambda x: x.rolling(252, min_periods=1).max())
)

df["52_week_high_distance"] = (
    df["close"] / rolling_high - 1
)


# 12. 52-Week Low Distance
rolling_low = (
    df.groupby("symbol")["low"]
    .transform(lambda x: x.rolling(252, min_periods=1).min())
)

df["52_week_low_distance"] = (
    df["close"] / rolling_low - 1
)


FEATURES = [
    "log_return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "return_120d",
    "intraday_range",
    "open_close_spread",
    "gap_overnight_return",
    "close_to_high_distance",
    "52_week_high_distance",
    "52_week_low_distance",
]


# ============================================================
# SAVE FULL FEATURE DATA
# ============================================================

feature_columns = [
    "date",
    "symbol",
    "next_day_return",
] + FEATURES

features_full = df[feature_columns].copy()

features_full.to_parquet(
    os.path.join(OUTPUT_DIR, "features_full.parquet"),
    index=False,
)

print("\nFull feature dataset saved.")


# ============================================================
# SELECTION DATASET
#
# IMPORTANT:
# Features were computed using the full dataset above.
# But RankIC / ICIR / correlation use ONLY 2022-2023.
# ============================================================

selection_df = df[
    (df["date"] >= SELECTION_START)
    & (df["date"] <= SELECTION_END)
].copy()

print(
    f"\nSelection period: "
    f"{selection_df['date'].min().date()} -> "
    f"{selection_df['date'].max().date()}"
)


# ============================================================
# DAILY RANKIC
#
# For each feature:
#
# IC_t = Spearman(feature_t, next_day_return_t)
#
# calculated cross-sectionally across stocks for each day.
# ============================================================

daily_ic_records = []

dates = sorted(selection_df["date"].dropna().unique())

for date in dates:

    day_data = selection_df[
        selection_df["date"] == date
    ]

    for feature in FEATURES:

        temp = day_data[
            [feature, "next_day_return"]
        ].dropna()

        # Need at least 2 stocks to calculate correlation
        if len(temp) < 2:
            ic = np.nan
        else:
            ic = temp[feature].corr(
                temp["next_day_return"],
                method="spearman",
            )

        daily_ic_records.append(
            {
                "date": date,
                "feature": feature,
                "daily_ic": ic,
                "n_stocks": len(temp),
            }
        )


daily_rankic = pd.DataFrame(daily_ic_records)

daily_rankic.to_csv(
    os.path.join(OUTPUT_DIR, "daily_rankic.csv"),
    index=False,
)

print("\nDaily RankIC calculated.")


# ============================================================
# RANKIC + ICIR
# ============================================================

metrics = []

for feature in FEATURES:

    feature_ic = daily_rankic.loc[
        daily_rankic["feature"] == feature,
        "daily_ic",
    ].dropna()

    if len(feature_ic) == 0:

        rankic = np.nan
        ic_std = np.nan
        icir = np.nan

    else:

        rankic = feature_ic.mean()

        ic_std = feature_ic.std(
            ddof=1
        )

        if ic_std == 0 or np.isnan(ic_std):
            icir = np.nan
        else:
            icir = rankic / ic_std

    metrics.append(
        {
            "feature": feature,
            "rankic": rankic,
            "ic_std": ic_std,
            "icir": icir,
            "n_daily_ic": len(feature_ic),
        }
    )


metrics_df = pd.DataFrame(metrics)

metrics_df["abs_rankic"] = metrics_df["rankic"].abs()
metrics_df["abs_icir"] = metrics_df["icir"].abs()

metrics_df = metrics_df.sort_values(
    "abs_icir",
    ascending=False,
).reset_index(drop=True)

metrics_df["icir_rank"] = (
    metrics_df.index + 1
)


metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "icir_metrics.csv"),
    index=False,
)

print("\nRankIC / ICIR metrics:")
print(
    metrics_df[
        [
            "feature",
            "rankic",
            "icir",
            "icir_rank",
        ]
    ].to_string(index=False)
)


# ============================================================
# REDUNDANCY CHECK
#
# Stack stock x day observations.
# Correlation is calculated between feature columns.
#
# IMPORTANT:
# Only 2022-2023 is used.
# ============================================================

correlation_data = selection_df[
    ["date", "symbol"] + FEATURES
].copy()

feature_matrix = correlation_data[FEATURES]

spearman_corr = feature_matrix.corr(
    method="spearman"
)

spearman_corr.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "spearman_correlation.csv",
    )
)


# ============================================================
# REDUNDANCY SELECTION
#
# We process features from strongest ICIR to weakest ICIR.
#
# If a feature is highly correlated with a feature already kept:
#
#     |correlation| > 0.70
#
# then it is considered redundant.
#
# HOWEVER:
# The winner is decided using HIGHER |RankIC|,
# as requested.
#
# We continue until TARGET_FEATURES non-redundant features
# are selected, or all features have been processed.
# ============================================================

metrics_lookup = (
    metrics_df
    .set_index("feature")
    .to_dict("index")
)

# Rank candidates by absolute ICIR first.
# This determines which strong features get considered first.
candidate_features = (
    metrics_df
    .dropna(subset=["rankic", "icir"])
    .sort_values(
        "abs_icir",
        ascending=False,
    )["feature"]
    .tolist()
)


selected_features = []
dropped_features = []

# Keep track of every redundancy decision
redundancy_records = []


def abs_rankic(feature):
    value = metrics_lookup[feature]["rankic"]

    if pd.isna(value):
        return -np.inf

    return abs(value)


for feature in candidate_features:

    if len(selected_features) == 0:
        selected_features.append(feature)
        continue

    redundant_with = None

    for selected in selected_features:

        corr_value = spearman_corr.loc[
            feature,
            selected,
        ]

        if pd.isna(corr_value):
            continue

        if abs(corr_value) > CORRELATION_THRESHOLD:

            redundant_with = selected

            redundancy_records.append(
                {
                    "feature_a": feature,
                    "feature_b": selected,
                    "correlation": corr_value,
                    "abs_correlation": abs(corr_value),
                    "rankic_a": metrics_lookup[feature]["rankic"],
                    "rankic_b": metrics_lookup[selected]["rankic"],
                }
            )

            # Compare ABSOLUTE RankIC.
            if abs_rankic(feature) > abs_rankic(selected):

                # New feature is stronger.
                selected_features.remove(selected)

                dropped_features.append(
                    {
                        "feature": selected,
                        "dropped_correlated_with": feature,
                        "correlation": corr_value,
                        "reason": "Redundant - lower absolute RankIC",
                    }
                )

                selected_features.append(feature)

            else:

                dropped_features.append(
                    {
                        "feature": feature,
                        "dropped_correlated_with": selected,
                        "correlation": corr_value,
                        "reason": "Redundant - lower absolute RankIC",
                    }
                )

            break

    if redundant_with is None:

        selected_features.append(feature)


# ============================================================
# FINAL TOP FEATURES
# ============================================================

# If redundancy filtering leaves more than the requested number,
# take the strongest non-redundant features by absolute ICIR.

selected_features = sorted(
    selected_features,
    key=lambda x: metrics_lookup[x]["abs_icir"],
    reverse=True,
)

selected_features = selected_features[
    :TARGET_FEATURES
]


# ============================================================
# BUILD FINAL DELIVERABLE
# ============================================================

final_rows = []

selected_set = set(selected_features)

dropped_lookup = {}

for item in dropped_features:

    feature = item["feature"]

    dropped_lookup[feature] = item


for feature in FEATURES:

    metric = metrics_lookup.get(feature)

    if metric is None:
        continue

    rankic = metric["rankic"]
    icir = metric["icir"]

    if feature in selected_set:

        verdict = "KEEP"

        dropped_with = "—"

    elif feature in dropped_lookup:

        drop_info = dropped_lookup[feature]

        verdict = "DISCARD (Redundant)"

        dropped_with = (
            f"Dropped — corr "
            f"{drop_info['correlation']:.4f} "
            f"with {drop_info['dropped_correlated_with']}"
        )

    else:

        verdict = "DISCARD (Not in Top Features)"

        dropped_with = "—"

    final_rows.append(
        {
            "feature": feature,
            "rankic": rankic,
            "icir": icir,
            "dropped_correlated_with": dropped_with,
            "verdict": verdict,
        }
    )


final_deliverable = pd.DataFrame(final_rows)

# Put selected features first, ordered by absolute ICIR.
final_deliverable["selected_order"] = (
    final_deliverable["feature"]
    .map(
        {
            feature: i
            for i, feature
            in enumerate(selected_features)
        }
    )
)

final_deliverable = final_deliverable.sort_values(
    "selected_order",
    na_position="last",
)

final_deliverable = final_deliverable.drop(
    columns=["selected_order"]
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

final_deliverable.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_deliverable.csv",
    ),
    index=False,
)

pd.DataFrame(
    redundancy_records
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "redundancy_pairs.csv",
    ),
    index=False,
)

pd.DataFrame(
    dropped_features
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dropped_correlated.csv",
    ),
    index=False,
)

pd.DataFrame(
    {
        "selected_feature": selected_features
    }
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "selected_features.csv",
    ),
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "Total Category 1 features",
            "Features with valid ICIR",
            "Target selected features",
            "Final selected features",
            "Selection start",
            "Selection end",
            "Correlation threshold",
        ],
        "value": [
            len(FEATURES),
            len(candidate_features),
            TARGET_FEATURES,
            len(selected_features),
            SELECTION_START,
            SELECTION_END,
            CORRELATION_THRESHOLD,
        ],
    }
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase1_summary.csv",
    ),
    index=False,
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print("\n")
print("=" * 80)
print("CATEGORY 1 FINAL FEATURE SELECTION")
print("=" * 80)

print(
    final_deliverable.to_string(
        index=False
    )
)

print("\nSelected features:")
for i, feature in enumerate(
    selected_features,
    start=1,
):
    print(
        f"{i}. {feature} "
        f"(RankIC={metrics_lookup[feature]['rankic']:.6f}, "
        f"ICIR={metrics_lookup[feature]['icir']:.6f})"
    )

print("\n")
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    f"All results saved to: {OUTPUT_DIR}"
)

print(
    "\nDone."
)