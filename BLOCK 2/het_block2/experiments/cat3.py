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
    "p1_cat3"
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
# BASIC SERIES
# ============================================================

grouped_close = df.groupby("symbol")["close"]

df["daily_return"] = (
    grouped_close
    .transform(
        lambda x: x.pct_change()
    )
)

df["log_return"] = (
    grouped_close
    .transform(
        lambda x:
        np.log(x / x.shift(1))
    )
)

df["next_day_return"] = (
    grouped_close
    .shift(-1)
    / df["close"]
    - 1
)


# ============================================================
# 1. RSI 14-DAY
# ============================================================

def calculate_rsi(series, period):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = (
        gain
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    avg_loss = (
        loss
        .ewm(
            alpha=1 / period,
            adjust=False,
            min_periods=period
        )
        .mean()
    )

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    rsi = (
        100
        - (
            100
            / (1 + rs)
        )
    )

    # If average loss is zero, RSI is 100
    rsi = rsi.where(
        avg_loss != 0,
        100
    )

    return rsi


df["rsi_14d"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        calculate_rsi(x, 14)
    )
)


# ============================================================
# 2. RSI 7-DAY
# ============================================================

df["rsi_7d"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        calculate_rsi(x, 7)
    )
)


# ============================================================
# 3. MACD LINE
# ============================================================

df["ema_12"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        x.ewm(
            span=12,
            adjust=False
        ).mean()
    )
)

df["ema_26"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        x.ewm(
            span=26,
            adjust=False
        ).mean()
    )
)

df["macd_line"] = (
    df["ema_12"]
    - df["ema_26"]
)


# ============================================================
# 4. MACD SIGNAL LINE
# ============================================================

df["macd_signal"] = (
    df.groupby("symbol")["macd_line"]
    .transform(
        lambda x:
        x.ewm(
            span=9,
            adjust=False
        ).mean()
    )
)


# ============================================================
# 5. MACD HISTOGRAM
# ============================================================

df["macd_histogram"] = (
    df["macd_line"]
    - df["macd_signal"]
)


# ============================================================
# 6. ROC 12-DAY
# ============================================================

df["roc_12d"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        x / x.shift(12) - 1
    )
)


# ============================================================
# MOVING AVERAGES
# ============================================================

df["sma_5"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        x.rolling(
            5,
            min_periods=5
        ).mean()
    )
)

df["sma_20"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        x.rolling(
            20,
            min_periods=20
        ).mean()
    )
)

df["sma_50"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        x.rolling(
            50,
            min_periods=50
        ).mean()
    )
)

df["sma_60"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        x.rolling(
            60,
            min_periods=60
        ).mean()
    )
)

df["sma_200"] = (
    df.groupby("symbol")["close"]
    .transform(
        lambda x:
        x.rolling(
            200,
            min_periods=200
        ).mean()
    )
)


# ============================================================
# 7. MA RATIO 5 / 20
# ============================================================

df["ma_ratio_5_20"] = (
    df["sma_5"]
    / df["sma_20"]
)


# ============================================================
# 8. MA RATIO 20 / 60
# ============================================================

df["ma_ratio_20_60"] = (
    df["sma_20"]
    / df["sma_60"]
)


# ============================================================
# 9. MA RATIO 50 / 200
# ============================================================

df["ma_ratio_50_200"] = (
    df["sma_50"]
    / df["sma_200"]
)


# ============================================================
# 10. PRICE VS SMA 20
# ============================================================

df["price_vs_sma_20"] = (
    df["close"]
    / df["sma_20"]
    - 1
)


# ============================================================
# 11. PRICE VS SMA 50
# ============================================================

df["price_vs_sma_50"] = (
    df["close"]
    / df["sma_50"]
    - 1
)


# ============================================================
# 12. STOCHASTIC %K
# ============================================================

rolling_low_14 = (
    df.groupby("symbol")["low"]
    .transform(
        lambda x:
        x.rolling(
            14,
            min_periods=14
        ).min()
    )
)

rolling_high_14 = (
    df.groupby("symbol")["high"]
    .transform(
        lambda x:
        x.rolling(
            14,
            min_periods=14
        ).max()
    )
)

stochastic_range = (
    rolling_high_14
    - rolling_low_14
)

df["stochastic_k"] = (
    (
        df["close"]
        - rolling_low_14
    )
    / stochastic_range
    * 100
)


# ============================================================
# 13. WILLIAMS %R
# ============================================================

df["williams_r"] = (
    (
        rolling_high_14
        - df["close"]
    )
    / stochastic_range
    * -100
)


# ============================================================
# 14. CCI 20-DAY
# ============================================================

df["typical_price"] = (
    df["high"]
    + df["low"]
    + df["close"]
) / 3

df["tp_sma_20"] = (
    df.groupby("symbol")["typical_price"]
    .transform(
        lambda x:
        x.rolling(
            20,
            min_periods=20
        ).mean()
    )
)


def mean_absolute_deviation(series):

    mean_value = series.mean()

    return np.mean(
        np.abs(
            series - mean_value
        )
    )


df["tp_mad_20"] = (
    df.groupby("symbol")["typical_price"]
    .transform(
        lambda x:
        x.rolling(
            20,
            min_periods=20
        ).apply(
            mean_absolute_deviation,
            raw=False
        )
    )
)

df["cci_20d"] = (
    (
        df["typical_price"]
        - df["tp_sma_20"]
    )
    / (
        0.015
        * df["tp_mad_20"]
    )
)


# ============================================================
# 15. MOMENTUM SCORE
#
# Average z-score of returns at:
# 5d, 10d, 20d, 60d, 120d
#
# These horizons are an implementation choice because the
# project document does not specify the exact horizons.
# ============================================================

momentum_horizons = [
    5,
    10,
    20,
    60,
    120
]

momentum_zscore_columns = []

for horizon in momentum_horizons:

    return_column = (
        f"_momentum_return_{horizon}"
    )

    zscore_column = (
        f"_momentum_z_{horizon}"
    )

    df[return_column] = (
        df.groupby("symbol")["close"]
        .transform(
            lambda x, h=horizon:
            x / x.shift(h) - 1
        )
    )

    rolling_mean = (
        df.groupby("symbol")[return_column]
        .transform(
            lambda x:
            x.rolling(
                60,
                min_periods=60
            ).mean()
        )
    )

    rolling_std = (
        df.groupby("symbol")[return_column]
        .transform(
            lambda x:
            x.rolling(
                60,
                min_periods=60
            ).std()
        )
    )

    df[zscore_column] = (
        (
            df[return_column]
            - rolling_mean
        )
        / rolling_std
    )

    momentum_zscore_columns.append(
        zscore_column
    )


df["momentum_score"] = (
    df[
        momentum_zscore_columns
    ].mean(axis=1)
)


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "rsi_14d",
    "rsi_7d",
    "macd_line",
    "macd_signal",
    "macd_histogram",
    "roc_12d",
    "ma_ratio_5_20",
    "ma_ratio_20_60",
    "ma_ratio_50_200",
    "price_vs_sma_20",
    "price_vs_sma_50",
    "stochastic_k",
    "williams_r",
    "cci_20d",
    "momentum_score",
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
    "\nFull Category 3 feature dataset saved."
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
# Rows = stock x day
# Columns = features
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


# Selected features first,
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
            "Total Category 3 features",
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
    "CATEGORY 3 FINAL FEATURE SELECTION"
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