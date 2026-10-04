import os
import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "nifty50_stocks_ohlcv.parquet"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "p1_cat2"
)

SELECTION_START = "2022-01-01"
SELECTION_END = "2023-12-31"

CORRELATION_THRESHOLD = 0.70
TARGET_FEATURES = 7


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_parquet(
    INPUT_FILE
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = (
    df
    .sort_values(
        ["symbol", "date"]
    )
    .reset_index(drop=True)
)

print(
    f"Loaded data: {df.shape}"
)

print(
    f"Date range: "
    f"{df['date'].min().date()} -> "
    f"{df['date'].max().date()}"
)

print(
    f"Stocks: "
    f"{df['symbol'].nunique()}"
)


# ============================================================
# BASIC RETURNS
# ============================================================

df["daily_return"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        x.pct_change()
    )
)

df["log_return"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        np.log(x / x.shift(1))
    )
)

df["next_day_return"] = (
    df.groupby("symbol")["close"]
    .shift(-1)
    / df["close"]
    - 1
)


# ============================================================
# CATEGORY 2 FEATURES
# ============================================================

# ------------------------------------------------------------
# 1. Rolling Volatility 5-day
# ------------------------------------------------------------

df["rolling_volatility_5d"] = (
    df.groupby("symbol")["log_return"]
    .transform(
        lambda x:
        x.rolling(
            window=5,
            min_periods=5
        ).std()
    )
)


# ------------------------------------------------------------
# 2. Rolling Volatility 10-day
# ------------------------------------------------------------

df["rolling_volatility_10d"] = (
    df.groupby("symbol")["log_return"]
    .transform(
        lambda x:
        x.rolling(
            window=10,
            min_periods=10
        ).std()
    )
)


# ------------------------------------------------------------
# 3. Rolling Volatility 20-day
# ------------------------------------------------------------

df["rolling_volatility_20d"] = (
    df.groupby("symbol")["log_return"]
    .transform(
        lambda x:
        x.rolling(
            window=20,
            min_periods=20
        ).std()
    )
)


# ------------------------------------------------------------
# 4. Rolling Volatility 60-day
# ------------------------------------------------------------

df["rolling_volatility_60d"] = (
    df.groupby("symbol")["log_return"]
    .transform(
        lambda x:
        x.rolling(
            window=60,
            min_periods=60
        ).std()
    )
)


# ------------------------------------------------------------
# 5. Volatility Ratio 5d / 20d
# ------------------------------------------------------------

df["volatility_ratio_5d_20d"] = (
    df["rolling_volatility_5d"]
    / df["rolling_volatility_20d"]
)


# ------------------------------------------------------------
# 6. Volatility Ratio 20d / 60d
# ------------------------------------------------------------

df["volatility_ratio_20d_60d"] = (
    df["rolling_volatility_20d"]
    / df["rolling_volatility_60d"]
)


# ============================================================
# TRUE RANGE
# ============================================================

previous_close = (
    df.groupby("symbol")["close"]
    .shift(1)
)

true_range_components = pd.concat(
    [
        df["high"] - df["low"],
        (df["high"] - previous_close).abs(),
        (df["low"] - previous_close).abs(),
    ],
    axis=1
)

true_range = (
    true_range_components
    .max(axis=1)
)

df["true_range"] = true_range


# ------------------------------------------------------------
# 7. ATR 14-day
# ------------------------------------------------------------

df["atr_14d"] = (
    df.groupby("symbol")["true_range"]
    .transform(
        lambda x:
        x.rolling(
            window=14,
            min_periods=14
        ).mean()
    )
)


# ------------------------------------------------------------
# 8. Normalized ATR
# ------------------------------------------------------------

df["normalized_atr"] = (
    df["atr_14d"]
    / df["close"]
)


# ============================================================
# GARMAN-KLASS VOLATILITY
# ============================================================

log_hl = np.log(
    df["high"] / df["low"]
)

log_co = np.log(
    df["close"] / df["open"]
)

gk_daily = (
    0.5 * (log_hl ** 2)
    - (
        2 * np.log(2) - 1
    ) * (log_co ** 2)
)

gk_daily = (
    gk_daily
    .where(gk_daily >= 0)
)

df["garman_klass_volatility"] = (
    gk_daily
    .groupby(df["symbol"])
    .transform(
        lambda x:
        np.sqrt(
            x.rolling(
                window=20,
                min_periods=20
            ).mean()
        )
    )
)


# ============================================================
# PARKINSON VOLATILITY
# ============================================================

parkinson_daily = (
    np.log(
        df["high"] / df["low"]
    ) ** 2
) / (
    4 * np.log(2)
)

df["parkinson_volatility"] = (
    parkinson_daily
    .groupby(df["symbol"])
    .transform(
        lambda x:
        np.sqrt(
            x.rolling(
                window=20,
                min_periods=20
            ).mean()
        )
    )
)


# ------------------------------------------------------------
# 11. Rolling Skewness 20-day
# ------------------------------------------------------------

df["rolling_skewness_20d"] = (
    df.groupby("symbol")["log_return"]
    .transform(
        lambda x:
        x.rolling(
            window=20,
            min_periods=20
        ).skew()
    )
)


# ------------------------------------------------------------
# 12. Rolling Kurtosis 20-day
# ------------------------------------------------------------

df["rolling_kurtosis_20d"] = (
    df.groupby("symbol")["log_return"]
    .transform(
        lambda x:
        x.rolling(
            window=20,
            min_periods=20
        ).kurt()
    )
)


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "rolling_volatility_5d",
    "rolling_volatility_10d",
    "rolling_volatility_20d",
    "rolling_volatility_60d",
    "volatility_ratio_5d_20d",
    "volatility_ratio_20d_60d",
    "atr_14d",
    "normalized_atr",
    "garman_klass_volatility",
    "parkinson_volatility",
    "rolling_skewness_20d",
    "rolling_kurtosis_20d",
]


# ============================================================
# SAVE FULL FEATURE DATA
# ============================================================

feature_columns = [
    "date",
    "symbol",
    "next_day_return",
] + FEATURES

features_full = (
    df[feature_columns]
    .copy()
)

features_full.to_parquet(
    os.path.join(
        OUTPUT_DIR,
        "features_full.parquet"
    ),
    index=False
)

print(
    "\nFull Category 2 feature dataset saved."
)


# ============================================================
# SELECTION DATA
#
# ONLY 2022-2023
# ============================================================

selection_df = df[
    (df["date"] >= SELECTION_START)
    &
    (df["date"] <= SELECTION_END)
].copy()

print(
    f"\nSelection period: "
    f"{selection_df['date'].min().date()} -> "
    f"{selection_df['date'].max().date()}"
)


# ============================================================
# DAILY RANKIC
# ============================================================

daily_ic_records = []

dates = sorted(
    selection_df["date"]
    .dropna()
    .unique()
)

for date in dates:

    day_data = selection_df[
        selection_df["date"] == date
    ]

    for feature in FEATURES:

        temp = day_data[
            [
                feature,
                "next_day_return"
            ]
        ].dropna()

        if len(temp) < 2:

            ic = np.nan

        else:

            ic = temp[feature].corr(
                temp["next_day_return"],
                method="spearman"
            )

        daily_ic_records.append(
            {
                "date": date,
                "feature": feature,
                "daily_ic": ic,
                "n_stocks": len(temp),
            }
        )


daily_rankic = pd.DataFrame(
    daily_ic_records
)

daily_rankic.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "daily_rankic.csv"
    ),
    index=False
)

print(
    "\nDaily RankIC calculated."
)


# ============================================================
# RANKIC + ICIR
# ============================================================

metrics = []

for feature in FEATURES:

    feature_ic = (
        daily_rankic.loc[
            daily_rankic["feature"] == feature,
            "daily_ic"
        ]
        .dropna()
    )

    if len(feature_ic) == 0:

        rankic = np.nan
        ic_std = np.nan
        icir = np.nan

    else:

        rankic = feature_ic.mean()

        ic_std = feature_ic.std(
            ddof=1
        )

        if (
            ic_std == 0
            or np.isnan(ic_std)
        ):

            icir = np.nan

        else:

            icir = (
                rankic
                / ic_std
            )

    metrics.append(
        {
            "feature": feature,
            "rankic": rankic,
            "ic_std": ic_std,
            "icir": icir,
            "n_daily_ic": len(
                feature_ic
            ),
        }
    )


metrics_df = pd.DataFrame(
    metrics
)

metrics_df["abs_rankic"] = (
    metrics_df["rankic"].abs()
)

metrics_df["abs_icir"] = (
    metrics_df["icir"].abs()
)

metrics_df = (
    metrics_df
    .sort_values(
        "abs_icir",
        ascending=False
    )
    .reset_index(drop=True)
)

metrics_df["icir_rank"] = (
    metrics_df.index + 1
)

metrics_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "icir_metrics.csv"
    ),
    index=False
)


# ============================================================
# PRINT METRICS
# ============================================================

print(
    "\nRankIC / ICIR:"
)

print(
    metrics_df[
        [
            "feature",
            "rankic",
            "icir",
            "icir_rank"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# FEATURE CORRELATION
#
# Stock x day rows
# Features as columns
#
# ONLY 2022-2023
# ============================================================

correlation_data = selection_df[
    ["date", "symbol"] + FEATURES
].copy()

feature_matrix = (
    correlation_data[FEATURES]
)

spearman_corr = (
    feature_matrix.corr(
        method="spearman"
    )
)

spearman_corr.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "spearman_correlation.csv"
    )
)


# ============================================================
# REDUNDANCY SELECTION
#
# No ICIR cutoff.
#
# Features are considered in descending |ICIR|.
#
# If |correlation| > 0.70:
#
#     compare |RankIC|
#
# Keep the feature with higher |RankIC|.
# ============================================================

metrics_lookup = (
    metrics_df
    .set_index("feature")
    .to_dict("index")
)

candidate_features = (
    metrics_df
    .dropna(
        subset=[
            "rankic",
            "icir"
        ]
    )
    .sort_values(
        "abs_icir",
        ascending=False
    )["feature"]
    .tolist()
)


selected_features = []

dropped_features = []

redundancy_records = []


def abs_rankic(feature):

    value = (
        metrics_lookup[
            feature
        ]["rankic"]
    )

    if pd.isna(value):

        return -np.inf

    return abs(value)


for feature in candidate_features:

    if len(selected_features) == 0:

        selected_features.append(
            feature
        )

        continue

    redundant_with = None

    for selected in selected_features:

        corr_value = (
            spearman_corr.loc[
                feature,
                selected
            ]
        )

        if pd.isna(corr_value):

            continue

        if (
            abs(corr_value)
            > CORRELATION_THRESHOLD
        ):

            redundant_with = selected

            redundancy_records.append(
                {
                    "feature_a": feature,
                    "feature_b": selected,
                    "correlation": corr_value,
                    "abs_correlation": abs(
                        corr_value
                    ),
                    "rankic_a":
                        metrics_lookup[
                            feature
                        ]["rankic"],
                    "rankic_b":
                        metrics_lookup[
                            selected
                        ]["rankic"],
                }
            )

            # Compare ABSOLUTE RankIC
            if (
                abs_rankic(feature)
                > abs_rankic(selected)
            ):

                selected_features.remove(
                    selected
                )

                dropped_features.append(
                    {
                        "feature":
                            selected,
                        "dropped_correlated_with":
                            feature,
                        "correlation":
                            corr_value,
                        "reason":
                            "Redundant - lower absolute RankIC",
                    }
                )

                selected_features.append(
                    feature
                )

            else:

                dropped_features.append(
                    {
                        "feature":
                            feature,
                        "dropped_correlated_with":
                            selected,
                        "correlation":
                            corr_value,
                        "reason":
                            "Redundant - lower absolute RankIC",
                    }
                )

            break

    if redundant_with is None:

        selected_features.append(
            feature
        )


# ============================================================
# TOP NON-REDUNDANT FEATURES
# ============================================================

selected_features = sorted(
    selected_features,
    key=lambda x:
        metrics_lookup[x]["abs_icir"],
    reverse=True
)

selected_features = (
    selected_features[
        :TARGET_FEATURES
    ]
)


# ============================================================
# FINAL DELIVERABLE
# ============================================================

dropped_lookup = {}

for item in dropped_features:

    dropped_lookup[
        item["feature"]
    ] = item


final_rows = []

selected_set = set(
    selected_features
)

for feature in FEATURES:

    metric = (
        metrics_lookup.get(
            feature
        )
    )

    if metric is None:
        continue

    rankic = metric["rankic"]
    icir = metric["icir"]

    if feature in selected_set:

        verdict = "KEEP"

        dropped_with = "—"

    elif feature in dropped_lookup:

        drop_info = (
            dropped_lookup[
                feature
            ]
        )

        verdict = (
            "DISCARD (Redundant)"
        )

        dropped_with = (
            f"Dropped — corr "
            f"{drop_info['correlation']:.4f} "
            f"with "
            f"{drop_info['dropped_correlated_with']}"
        )

    else:

        verdict = (
            "DISCARD (Not in Top Features)"
        )

        dropped_with = "—"

    final_rows.append(
        {
            "feature": feature,
            "rankic": rankic,
            "icir": icir,
            "dropped_correlated_with":
                dropped_with,
            "verdict": verdict,
        }
    )


final_deliverable = (
    pd.DataFrame(final_rows)
)


# Put selected features first,
# ordered by |ICIR|.

final_deliverable[
    "selected_order"
] = (
    final_deliverable[
        "feature"
    ].map(
        {
            feature: i
            for i, feature
            in enumerate(
                selected_features
            )
        }
    )
)

final_deliverable = (
    final_deliverable
    .sort_values(
        "selected_order",
        na_position="last"
    )
    .drop(
        columns=["selected_order"]
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

final_deliverable.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_deliverable.csv"
    ),
    index=False
)

pd.DataFrame(
    redundancy_records
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "redundancy_pairs.csv"
    ),
    index=False
)

pd.DataFrame(
    dropped_features
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dropped_correlated.csv"
    ),
    index=False
)

pd.DataFrame(
    {
        "selected_feature":
            selected_features
    }
).to_csv(
    os.path.join(
        OUTPUT_DIR,
        "selected_features.csv"
    ),
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = pd.DataFrame(
    {
        "metric": [
            "Total Category 2 features",
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
        "phase1_summary.csv"
    ),
    index=False
)


# ============================================================
# PRINT FINAL RESULT
# ============================================================

print("\n")
print("=" * 80)
print(
    "CATEGORY 2 FINAL FEATURE SELECTION"
)
print("=" * 80)

print(
    final_deliverable.to_string(
        index=False
    )
)

print("\nSelected features:")

for i, feature in enumerate(
    selected_features,
    start=1
):

    print(
        f"{i}. {feature} "
        f"(RankIC="
        f"{metrics_lookup[feature]['rankic']:.6f}, "
        f"ICIR="
        f"{metrics_lookup[feature]['icir']:.6f})"
    )


print("\n")
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(
    f"All results saved to: "
    f"{OUTPUT_DIR}"
)

print(
    "\nDone."
)