import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "p1_cat8"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

SELECTION_START = pd.Timestamp("2022-01-01")
SELECTION_END = pd.Timestamp("2023-12-31")

CORRELATION_THRESHOLD = 0.70
TARGET_FEATURES = 7


# ============================================================
# CATEGORY 8 FEATURE NAMES
# ============================================================

FEATURES = [
    "volume_weighted_momentum",
    "volatility_adjusted_return",
    "mean_reversion_signal",
    "trend_strength",
    "vix_adjusted_momentum",
    "gap_volume_interaction",
    "rsi_volume_divergence",
    "breakout_signal",
    "reversal_signal",
    "sector_momentum_alignment",
]


# ============================================================
# INPUT FILES
# ============================================================

CATEGORY_FILES = {
    "cat1": os.path.join(
        RESULTS_DIR,
        "p1_cat1",
        "features_full.parquet"
    ),
    "cat2": os.path.join(
        RESULTS_DIR,
        "p1_cat2",
        "features_full.parquet"
    ),
    "cat3": os.path.join(
        RESULTS_DIR,
        "p1_cat3",
        "features_full.parquet"
    ),
    "cat4": os.path.join(
        RESULTS_DIR,
        "p1_cat4",
        "features_full.parquet"
    ),
    "cat5": os.path.join(
        RESULTS_DIR,
        "p1_cat5",
        "features_full.parquet"
    ),
    "cat6": os.path.join(
        RESULTS_DIR,
        "p1_cat6",
        "features_full.parquet"
    ),
    "cat7": os.path.join(
        RESULTS_DIR,
        "p1_cat7",
        "features_full.parquet"
    ),
}


# ============================================================
# REQUIRED INPUT FEATURES
# ============================================================

REQUIRED_INPUTS = {
    "return_5d": "Category 1",
    "return_20d": "Category 1",
    "gap_overnight_return": "Category 1",

    "rolling_volatility_20d": "Category 2",

    "ma_ratio_20_60": "Category 3",
    "rsi_14d": "Category 3",

    "volume_ratio_20d": "Category 4",

    "lower_shadow_ratio": "Category 5",
    "bollinger_pct_b": "Category 5",

    "india_vix_level": "Category 6",

    "sector_relative_return": "Category 7",
}


# ============================================================
# HELPER: LOAD CATEGORY FILE
# ============================================================

def load_category(name, path):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nMissing {name} file:\n{path}\n"
            f"\nRun the corresponding category script first."
        )

    df = pd.read_parquet(path)

    required_keys = {
        "date",
        "symbol"
    }

    missing_keys = required_keys - set(df.columns)

    if missing_keys:
        raise ValueError(
            f"{name} is missing required columns: "
            f"{sorted(missing_keys)}"
        )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df["symbol"] = df["symbol"].astype(str)

    duplicate_count = df.duplicated(
        subset=["date", "symbol"]
    ).sum()

    if duplicate_count > 0:
        raise ValueError(
            f"{name} contains "
            f"{duplicate_count} duplicate "
            f"(date, symbol) rows."
        )

    return df


# ============================================================
# LOAD CATEGORY 1-7
# ============================================================

print()
print("=" * 70)
print("CATEGORY 8: DERIVED & INTERACTION FEATURES")
print("=" * 70)

print()
print("Loading Category 1-7 feature files...")
print()


cat1 = load_category(
    "Category 1",
    CATEGORY_FILES["cat1"]
)

cat2 = load_category(
    "Category 2",
    CATEGORY_FILES["cat2"]
)

cat3 = load_category(
    "Category 3",
    CATEGORY_FILES["cat3"]
)

cat4 = load_category(
    "Category 4",
    CATEGORY_FILES["cat4"]
)

cat5 = load_category(
    "Category 5",
    CATEGORY_FILES["cat5"]
)

cat6 = load_category(
    "Category 6",
    CATEGORY_FILES["cat6"]
)

cat7 = load_category(
    "Category 7",
    CATEGORY_FILES["cat7"]
)


print(f"Category 1: {cat1.shape}")
print(f"Category 2: {cat2.shape}")
print(f"Category 3: {cat3.shape}")
print(f"Category 4: {cat4.shape}")
print(f"Category 5: {cat5.shape}")
print(f"Category 6: {cat6.shape}")
print(f"Category 7: {cat7.shape}")


# ============================================================
# VERIFY REQUIRED INPUT COLUMNS
# ============================================================

print()
print("Checking required input features...")

loaded_categories = {
    "Category 1": cat1,
    "Category 2": cat2,
    "Category 3": cat3,
    "Category 4": cat4,
    "Category 5": cat5,
    "Category 6": cat6,
    "Category 7": cat7,
}

for column, category in REQUIRED_INPUTS.items():

    if column not in loaded_categories[category].columns:

        raise ValueError(
            f"\nRequired feature '{column}' "
            f"was not found in {category}."
        )

print("All required input features found.")


# ============================================================
# MERGE CATEGORY FEATURES
# ============================================================

KEYS = [
    "date",
    "symbol"
]

print()
print("Merging Category 1-7 features...")

# Category 1 contains the target next_day_return.
features = cat1[
    KEYS + [
        "next_day_return"
    ] + [
        col
        for col in cat1.columns
        if col not in KEYS
        and col != "next_day_return"
    ]
].copy()


other_categories = [
    ("Category 2", cat2),
    ("Category 3", cat3),
    ("Category 4", cat4),
    ("Category 5", cat5),
    ("Category 6", cat6),
    ("Category 7", cat7),
]


for category_name, df in other_categories:

    new_columns = [
        col
        for col in df.columns
        if col not in KEYS
        and col != "next_day_return"
        and col not in features.columns
    ]

    features = features.merge(
        df[
            KEYS + new_columns
        ],
        on=KEYS,
        how="left",
        validate="one_to_one"
    )


features = features.sort_values(
    ["symbol", "date"]
).reset_index(
    drop=True
)


print(
    f"Merged feature table: {features.shape}"
)


# ============================================================
# VERIFY TARGET
# ============================================================

if "next_day_return" not in features.columns:

    raise ValueError(
        "\nnext_day_return is missing after merging."
    )

print(
    "Target column next_day_return: FOUND"
)


# ============================================================
# COMPUTE CATEGORY 8 FEATURES
# ============================================================

print()
print("Computing Category 8 features...")
print()


# ------------------------------------------------------------
# 1. Volume-Weighted Momentum
# ------------------------------------------------------------
#
# 20-day return multiplied by volume ratio.
#
# ------------------------------------------------------------

features[
    "volume_weighted_momentum"
] = (
    features["return_20d"]
    * features["volume_ratio_20d"]
)


# ------------------------------------------------------------
# 2. Volatility-Adjusted Return
# ------------------------------------------------------------
#
# 20-day return divided by 20-day volatility.
#
# ------------------------------------------------------------

vol_20 = (
    features["rolling_volatility_20d"]
    .replace(0, np.nan)
)

features[
    "volatility_adjusted_return"
] = (
    features["return_20d"]
    / vol_20
)


# ------------------------------------------------------------
# 3. Mean Reversion Signal
# ------------------------------------------------------------
#
# Negative of 5-day return, scaled by volatility.
#
# ------------------------------------------------------------

features[
    "mean_reversion_signal"
] = (
    -features["return_5d"]
    / vol_20
)


# ------------------------------------------------------------
# 4. Trend Strength
# ------------------------------------------------------------
#
# Absolute distance of MA ratio from 1,
# scaled by volume ratio.
#
# ------------------------------------------------------------

features[
    "trend_strength"
] = (
    (
        features["ma_ratio_20_60"]
        - 1.0
    ).abs()
    * features["volume_ratio_20d"]
)


# ------------------------------------------------------------
# 5. VIX-Adjusted Momentum
# ------------------------------------------------------------
#
# 20-day return divided by India VIX.
#
# ------------------------------------------------------------

vix = (
    features["india_vix_level"]
    .replace(0, np.nan)
)

features[
    "vix_adjusted_momentum"
] = (
    features["return_20d"]
    / vix
)


# ------------------------------------------------------------
# 6. Gap-Volume Interaction
# ------------------------------------------------------------
#
# Overnight gap multiplied by volume ratio.
#
# ------------------------------------------------------------

features[
    "gap_volume_interaction"
] = (
    features["gap_overnight_return"]
    * features["volume_ratio_20d"]
)


# ------------------------------------------------------------
# 7. RSI-Volume Divergence
# ------------------------------------------------------------
#
# RSI direction minus volume direction.
#
# Direction is measured by the day-to-day change:
#
# positive change -> +1
# negative change -> -1
# unchanged       ->  0
#
# ------------------------------------------------------------

rsi_direction = (
    features
    .groupby("symbol")["rsi_14d"]
    .diff()
    .pipe(np.sign)
)

volume_direction = (
    features
    .groupby("symbol")["volume_ratio_20d"]
    .diff()
    .pipe(np.sign)
)

features[
    "rsi_volume_divergence"
] = (
    rsi_direction
    - volume_direction
)


# ------------------------------------------------------------
# 8. Breakout Signal
# ------------------------------------------------------------
#
# Close above Bollinger upper band
# AND volume ratio above 1.5.
#
# Bollinger %B > 1 means close is above
# the upper Bollinger band.
#
# ------------------------------------------------------------

features[
    "breakout_signal"
] = (
    (
        features["bollinger_pct_b"]
        > 1.0
    )
    &
    (
        features["volume_ratio_20d"]
        > 1.5
    )
).astype(float)


# ------------------------------------------------------------
# 9. Reversal Signal
# ------------------------------------------------------------
#
# RSI below 30
# AND lower shadow ratio above 0.5.
#
# ------------------------------------------------------------

features[
    "reversal_signal"
] = (
    (
        features["rsi_14d"]
        < 30
    )
    &
    (
        features["lower_shadow_ratio"]
        > 0.5
    )
).astype(float)


# ------------------------------------------------------------
# 10. Sector Momentum Alignment
# ------------------------------------------------------------
#
# Category 7 defines:
#
# sector_relative_return =
# stock 20-day return - sector 20-day return
#
# Therefore:
#
# sector 20-day return =
# stock 20-day return
# - sector_relative_return
#
# Then compare the direction of stock momentum
# with the direction of sector momentum.
#
# +1 = same direction
# -1 = opposite direction
#
# ------------------------------------------------------------

sector_return_20d = (
    features["return_20d"]
    - features["sector_relative_return"]
)

stock_direction = np.sign(
    features["return_20d"]
)

sector_direction = np.sign(
    sector_return_20d
)

valid_alignment = (
    stock_direction.notna()
    &
    sector_direction.notna()
    &
    (stock_direction != 0)
    &
    (sector_direction != 0)
)

features[
    "sector_momentum_alignment"
] = np.nan

features.loc[
    valid_alignment,
    "sector_momentum_alignment"
] = np.where(
    stock_direction.loc[
        valid_alignment
    ]
    ==
    sector_direction.loc[
        valid_alignment
    ],
    1.0,
    -1.0
)


# ============================================================
# VERIFY ALL 10 FEATURES
# ============================================================

missing_category8 = [
    feature
    for feature in FEATURES
    if feature not in features.columns
]

if missing_category8:

    raise ValueError(
        "\nCategory 8 computation failed.\n"
        "Missing features:\n"
        + "\n".join(
            f"  - {feature}"
            for feature in missing_category8
        )
    )

print(
    "All 10 Category 8 features computed successfully."
)


# ============================================================
# SAVE FULL FEATURES
# ============================================================

# Keep next_day_return in the output because it is useful
# for validation and downstream feature-selection work.
#
# It is NOT one of the 10 Category 8 features.

features_full = features[
    KEYS
    + ["next_day_return"]
    + FEATURES
].copy()


features_full.to_parquet(
    os.path.join(
        OUTPUT_DIR,
        "features_full.parquet"
    ),
    index=False
)


print()
print(
    "Saved:"
)
print(
    os.path.join(
        OUTPUT_DIR,
        "features_full.parquet"
    )
)


# ============================================================
# DATA QUALITY REPORT
# ============================================================

quality_rows = []

for feature in FEATURES:

    total_rows = len(features_full)

    missing_rows = (
        features_full[feature]
        .isna()
        .sum()
    )

    valid_rows = (
        total_rows
        - missing_rows
    )

    quality_rows.append({
        "feature": feature,
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "missing_rows": missing_rows,
        "missing_pct": (
            100.0
            * missing_rows
            / total_rows
        )
    })


quality_df = pd.DataFrame(
    quality_rows
)

quality_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "data_quality.csv"
    ),
    index=False
)


# ============================================================
# SELECTION PERIOD
# ============================================================

print()
print("=" * 70)
print("PHASE 1 SELECTION")
print("=" * 70)

selection = features[
    (
        features["date"]
        >= SELECTION_START
    )
    &
    (
        features["date"]
        <= SELECTION_END
    )
].copy()


print(
    f"Selection period: "
    f"{SELECTION_START.date()} "
    f"to "
    f"{SELECTION_END.date()}"
)

print(
    f"Selection rows: {len(selection)}"
)


# ============================================================
# DAILY CROSS-SECTIONAL RANKIC
# ============================================================

print()
print("Calculating daily cross-sectional RankIC...")


daily_ic_records = []


for date, day in selection.groupby(
    "date",
    sort=True
):

    for feature in FEATURES:

        temp = day[
            [
                feature,
                "next_day_return"
            ]
        ].dropna()

        # Need enough stocks to calculate
        # a meaningful cross-sectional correlation.
        if len(temp) < 3:
            continue

        # Spearman correlation is undefined
        # if either variable is constant.
        if temp[feature].nunique() < 2:
            continue

        if temp["next_day_return"].nunique() < 2:
            continue

        ic = temp[
            feature
        ].corr(
            temp["next_day_return"],
            method="spearman"
        )

        if pd.notna(ic):

            daily_ic_records.append({
                "date": date,
                "feature": feature,
                "IC": ic,
                "n_stocks": len(temp)
            })


daily_rankic = pd.DataFrame(
    daily_ic_records
)


if daily_rankic.empty:

    raise ValueError(
        "\nNo daily RankIC values were calculated."
    )


daily_rankic.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "daily_rankic.csv"
    ),
    index=False
)


# ============================================================
# RANKIC + ICIR
# ============================================================

print(
    "Calculating RankIC and ICIR..."
)


icir_rows = []


for feature in FEATURES:

    values = daily_rankic.loc[
        daily_rankic["feature"] == feature,
        "IC"
    ].dropna()

    n_daily_ic = len(values)

    if n_daily_ic == 0:

        rank_ic = np.nan
        icir = np.nan

    else:

        rank_ic = values.mean()

        if n_daily_ic < 2:

            icir = np.nan

        else:

            ic_std = values.std(
                ddof=1
            )

            if (
                pd.isna(ic_std)
                or ic_std == 0
            ):
                icir = np.nan
            else:
                icir = (
                    rank_ic
                    / ic_std
                )

    icir_rows.append({
        "feature": feature,
        "RankIC": rank_ic,
        "ICIR": icir,
        "n_daily_ic": n_daily_ic
    })


icir_metrics = pd.DataFrame(
    icir_rows
)

icir_metrics[
    "abs_ICIR"
] = icir_metrics[
    "ICIR"
].abs()


icir_metrics = icir_metrics.sort_values(
    "abs_ICIR",
    ascending=False,
    na_position="last"
).reset_index(
    drop=True
)


icir_metrics[
    "rank_by_abs_ICIR"
] = (
    np.arange(
        1,
        len(icir_metrics) + 1
    )
)


icir_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "icir_metrics.csv"
    ),
    index=False
)


# ============================================================
# WITHIN-CATEGORY SPEARMAN CORRELATION
# ============================================================

print(
    "Calculating within-category Spearman correlation..."
)


spearman_corr = selection[
    FEATURES
].corr(
    method="spearman"
)


spearman_corr.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "spearman_correlation.csv"
    )
)


# ============================================================
# REDUNDANCY REMOVAL
# ============================================================
#
# Features are considered redundant if:
#
# |Spearman correlation| > 0.70
#
# The feature with lower absolute ICIR is dropped.
#
# We process features from highest to lowest |ICIR|,
# which means a selected feature always has >= |ICIR|
# than a later feature that gets rejected because of it.
#
# No arbitrary ICIR cutoff is applied.
#
# ============================================================

print(
    "Removing correlated/redundant features..."
)


icir_lookup = (
    icir_metrics
    .set_index("feature")["abs_ICIR"]
    .to_dict()
)


ranked_features = (
    icir_metrics[
        "feature"
    ].tolist()
)


selected_features = []

dropped_records = []


for feature in ranked_features:

    feature_icir = icir_lookup.get(
        feature,
        np.nan
    )

    # Features without a usable ICIR cannot
    # be preferred over a feature with a valid ICIR.
    if pd.isna(feature_icir):

        dropped_records.append({
            "feature": feature,
            "correlated_with": "",
            "correlation": np.nan,
            "feature_abs_ICIR": np.nan,
            "correlated_feature_abs_ICIR": np.nan,
            "reason": "No valid ICIR"
        })

        continue

    is_redundant = False

    for existing in selected_features:

        correlation = spearman_corr.loc[
            feature,
            existing
        ]

        if (
            pd.notna(correlation)
            and abs(correlation)
            > CORRELATION_THRESHOLD
        ):

            existing_icir = icir_lookup[
                existing
            ]

            dropped_records.append({
                "feature": feature,
                "correlated_with": existing,
                "correlation": correlation,
                "feature_abs_ICIR": feature_icir,
                "correlated_feature_abs_ICIR": existing_icir,
                "reason": (
                    "Redundant; lower absolute ICIR"
                )
            })

            is_redundant = True

            break

    if not is_redundant:

        selected_features.append(
            feature
        )


# ============================================================
# TARGET 6-7 FEATURES
# ============================================================

selected_features = selected_features[
    :TARGET_FEATURES
]


selected_set = set(
    selected_features
)


# Features that survived correlation but fell
# outside the target top 7 are also recorded.

for feature in ranked_features:

    if feature in selected_set:
        continue

    already_recorded = any(
        row["feature"] == feature
        for row in dropped_records
    )

    if already_recorded:
        continue

    feature_icir = icir_lookup.get(
        feature,
        np.nan
    )

    dropped_records.append({
        "feature": feature,
        "correlated_with": "",
        "correlation": np.nan,
        "feature_abs_ICIR": feature_icir,
        "correlated_feature_abs_ICIR": np.nan,
        "reason": (
            "Outside target of "
            f"{TARGET_FEATURES} selected features"
        )
    })


# ============================================================
# SELECTED FEATURES TABLE
# ============================================================

selected_rows = []


for rank, feature in enumerate(
    selected_features,
    start=1
):

    row = icir_metrics[
        icir_metrics["feature"] == feature
    ].iloc[0]

    selected_rows.append({
        "selection_rank": rank,
        "feature": feature,
        "RankIC": row["RankIC"],
        "ICIR": row["ICIR"],
        "abs_ICIR": row["abs_ICIR"],
        "n_daily_ic": row["n_daily_ic"],
        "verdict": "KEEP"
    })


selected_df = pd.DataFrame(
    selected_rows
)


selected_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "selected_features.csv"
    ),
    index=False
)


# ============================================================
# DROPPED FEATURES
# ============================================================

dropped_df = pd.DataFrame(
    dropped_records
)


if not dropped_df.empty:

    dropped_df = dropped_df.sort_values(
        "feature_abs_ICIR",
        ascending=False,
        na_position="last"
    )


dropped_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dropped_correlated.csv"
    ),
    index=False
)


# ============================================================
# PHASE 1 SUMMARY
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "category",
        "total_features",
        "selection_start",
        "selection_end",
        "correlation_threshold",
        "target_features",
        "selected_features",
        "rankic_method",
        "selection_method"
    ],
    "value": [
        "Category 8 - Derived & Interaction",
        len(FEATURES),
        SELECTION_START.date(),
        SELECTION_END.date(),
        CORRELATION_THRESHOLD,
        TARGET_FEATURES,
        len(selected_features),
        (
            "Daily cross-sectional "
            "Spearman correlation between "
            "feature and next-day return"
        ),
        (
            "Rank features by absolute ICIR; "
            "remove |correlation| > 0.70 "
            "using lower absolute ICIR; "
            "retain top 7"
        )
    ]
})


summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "phase1_summary.csv"
    ),
    index=False
)


# ============================================================
# FINAL DELIVERABLE
# ============================================================

final_deliverable = selected_df[
    [
        "selection_rank",
        "feature",
        "RankIC",
        "ICIR",
        "abs_ICIR",
        "n_daily_ic",
        "verdict"
    ]
].copy()


final_deliverable.insert(
    2,
    "category",
    "Derived & Interaction"
)


final_deliverable.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_deliverable.csv"
    ),
    index=False
)


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print()
print("=" * 70)
print("CATEGORY 8 COMPLETE")
print("=" * 70)

print()
print("Selected features:")

if final_deliverable.empty:

    print(
        "ERROR: No features were selected."
    )

else:

    print(
        final_deliverable.to_string(
            index=False
        )
    )


print()
print(
    f"Selected {len(selected_features)} "
    f"of {len(FEATURES)} features."
)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print("Files created:")

for filename in sorted(
    os.listdir(OUTPUT_DIR)
):

    print(
        f"  {filename}"
    )

print()
print("=" * 70)