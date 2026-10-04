import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# PATHS
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
    "p1_cat4"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

SELECTION_START = "2022-01-01"
SELECTION_END = "2023-12-31"

CORRELATION_THRESHOLD = 0.70
TARGET_FEATURES = 7


# ============================================================
# CATEGORY 4 FEATURES
# ============================================================

FEATURES = [
    "volume_ratio_5d",
    "volume_ratio_20d",
    "vwap_deviation",
    "obv",
    "obv_roc",
    "vpt",
    "log_turnover",
    "volume_momentum",
    "ad_line",
    "mfi_14d",
    "force_index_13d",
    "volume_cv_20d",
]


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_features(group):
    group = group.sort_values("date").copy()

    close = group["close"]
    high = group["high"]
    low = group["low"]
    volume = group["volume"]

    # --------------------------------------------------------
    # 1. Volume Ratio 5-day
    # Today's volume / 5-day average volume
    # --------------------------------------------------------
    volume_ma_5 = volume.rolling(5).mean()
    group["volume_ratio_5d"] = volume / volume_ma_5

    # --------------------------------------------------------
    # 2. Volume Ratio 20-day
    # Today's volume / 20-day average volume
    # --------------------------------------------------------
    volume_ma_20 = volume.rolling(20).mean()
    group["volume_ratio_20d"] = volume / volume_ma_20

    # --------------------------------------------------------
    # 3. VWAP Deviation
    # Cumulative VWAP = cumulative(price * volume) /
    #                   cumulative(volume)
    # --------------------------------------------------------
    typical_price = (high + low + close) / 3

    cumulative_pv = (typical_price * volume).cumsum()
    cumulative_volume = volume.cumsum()

    vwap = cumulative_pv / cumulative_volume

    group["vwap_deviation"] = close / vwap - 1

    # --------------------------------------------------------
    # 4. On-Balance Volume (OBV)
    # --------------------------------------------------------
    price_change = close.diff()

    direction = np.sign(price_change)

    group["obv"] = (direction * volume).fillna(0).cumsum()

    # --------------------------------------------------------
    # 5. OBV Rate of Change
    # 20-day ROC of OBV
    # --------------------------------------------------------
    group["obv_roc"] = group["obv"].pct_change(20)

    # --------------------------------------------------------
    # 6. Volume-Price Trend (VPT)
    # VPT_t = VPT_(t-1) + Volume_t * Return_t
    # --------------------------------------------------------
    returns = close.pct_change()

    vpt_change = volume * returns

    group["vpt"] = vpt_change.fillna(0).cumsum()

    # --------------------------------------------------------
    # 7. Log Turnover
    # log(volume * price)
    # --------------------------------------------------------
    turnover = volume * close

    group["log_turnover"] = np.log1p(turnover)

    # --------------------------------------------------------
    # 8. Volume Momentum
    # 5-day average volume / 20-day average volume
    # --------------------------------------------------------
    group["volume_momentum"] = volume_ma_5 / volume_ma_20

    # --------------------------------------------------------
    # 9. Accumulation / Distribution Line
    # Money Flow Multiplier:
    #
    # ((Close - Low) - (High - Close)) / (High - Low)
    #
    # AD = cumulative(MFM * Volume)
    # --------------------------------------------------------
    high_low_range = high - low

    money_flow_multiplier = (
        ((close - low) - (high - close))
        / high_low_range.replace(0, np.nan)
    )

    money_flow_volume = money_flow_multiplier * volume

    group["ad_line"] = money_flow_volume.fillna(0).cumsum()

    # --------------------------------------------------------
    # 10. Money Flow Index (14-day)
    # --------------------------------------------------------
    typical_price = (high + low + close) / 3

    raw_money_flow = typical_price * volume

    tp_change = typical_price.diff()

    positive_flow = raw_money_flow.where(tp_change > 0, 0)
    negative_flow = raw_money_flow.where(tp_change < 0, 0)

    positive_flow_sum = positive_flow.rolling(14).sum()
    negative_flow_sum = negative_flow.abs().rolling(14).sum()

    money_ratio = (
        positive_flow_sum /
        negative_flow_sum.replace(0, np.nan)
    )

    group["mfi_14d"] = 100 - (100 / (1 + money_ratio))

    # --------------------------------------------------------
    # 11. Force Index (13-day)
    #
    # Force Index = price change * volume
    # Then smooth over 13 days
    # --------------------------------------------------------
    force_index_raw = close.diff() * volume

    group["force_index_13d"] = force_index_raw.ewm(
        span=13,
        adjust=False,
        min_periods=13
    ).mean()

    # --------------------------------------------------------
    # 12. Volume Coefficient of Variation
    #
    # CV = rolling standard deviation / rolling mean
    # --------------------------------------------------------
    rolling_volume_mean = volume.rolling(20).mean()
    rolling_volume_std = volume.rolling(20).std()

    group["volume_cv_20d"] = (
        rolling_volume_std /
        rolling_volume_mean.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Next-day return
    # --------------------------------------------------------
    group["next_day_return"] = close.shift(-1) / close - 1

    return group


# ============================================================
# DAILY RANKIC
# ============================================================

def calculate_daily_rankic(df, feature):
    selection_df = df[
        (df["date"] >= SELECTION_START) &
        (df["date"] <= SELECTION_END)
    ].copy()

    daily_results = []

    for date, day_data in selection_df.groupby("date"):

        valid = day_data[[feature, "next_day_return"]].dropna()

        if len(valid) < 3:
            continue

        ic, _ = spearmanr(
            valid[feature],
            valid["next_day_return"]
        )

        if np.isfinite(ic):
            daily_results.append(
                {
                    "date": date,
                    "feature": feature,
                    "rank_ic": ic,
                    "n_stocks": len(valid),
                }
            )

    return pd.DataFrame(daily_results)


# ============================================================
# ICIR METRICS
# ============================================================

def calculate_icir(daily_rankic):
    rows = []

    for feature, group in daily_rankic.groupby("feature"):

        ic_values = group["rank_ic"].dropna()

        if len(ic_values) == 0:
            continue

        mean_ic = ic_values.mean()
        std_ic = ic_values.std()

        if std_ic == 0 or not np.isfinite(std_ic):
            icir = np.nan
        else:
            icir = mean_ic / std_ic

        rows.append(
            {
                "feature": feature,
                "rank_ic": mean_ic,
                "ic_std": std_ic,
                "icir": icir,
                "abs_icir": abs(icir) if np.isfinite(icir) else np.nan,
                "n_days": len(ic_values),
            }
        )

    return pd.DataFrame(rows).sort_values(
        "abs_icir",
        ascending=False
    )


# ============================================================
# REDUNDANCY REMOVAL
# ============================================================

def select_non_redundant_features(df, icir_metrics):
    feature_data = df[
        (df["date"] >= SELECTION_START) &
        (df["date"] <= SELECTION_END)
    ][FEATURES].copy()

    correlation_matrix = feature_data.corr(
        method="spearman"
    )

    correlation_matrix.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "spearman_correlation.csv"
        )
    )

    ranked_features = (
        icir_metrics
        .sort_values("abs_icir", ascending=False)
        ["feature"]
        .tolist()
    )

    selected = []
    dropped = []

    for feature in ranked_features:

        if len(selected) == 0:
            selected.append(feature)
            continue

        redundant_with = None

        for selected_feature in selected:

            corr = correlation_matrix.loc[
                feature,
                selected_feature
            ]

            if pd.notna(corr) and abs(corr) > CORRELATION_THRESHOLD:
                redundant_with = selected_feature
                break

        if redundant_with is None:
            selected.append(feature)

        else:
            feature_icir = icir_metrics.loc[
                icir_metrics["feature"] == feature,
                "abs_icir"
            ].iloc[0]

            selected_icir = icir_metrics.loc[
                icir_metrics["feature"] == redundant_with,
                "abs_icir"
            ].iloc[0]

            # Higher |ICIR| survives
            if feature_icir > selected_icir:

                selected.remove(redundant_with)
                selected.append(feature)

                dropped.append(
                    {
                        "dropped_feature": redundant_with,
                        "kept_feature": feature,
                        "correlation": correlation_matrix.loc[
                            feature,
                            redundant_with
                        ],
                        "dropped_abs_icir": selected_icir,
                        "kept_abs_icir": feature_icir,
                    }
                )

            else:

                dropped.append(
                    {
                        "dropped_feature": feature,
                        "kept_feature": redundant_with,
                        "correlation": correlation_matrix.loc[
                            feature,
                            redundant_with
                        ],
                        "dropped_abs_icir": feature_icir,
                        "kept_abs_icir": selected_icir,
                    }
                )

    dropped_df = pd.DataFrame(dropped)

    dropped_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "dropped_correlated.csv"
        ),
        index=False
    )

    # Keep strongest 7 according to |ICIR|
    selected_metrics = icir_metrics[
        icir_metrics["feature"].isin(selected)
    ].sort_values(
        "abs_icir",
        ascending=False
    )

    selected_final = selected_metrics.head(
        TARGET_FEATURES
    )["feature"].tolist()

    return selected_final, correlation_matrix, dropped_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CATEGORY 4: VOLUME FEATURES")
    print("=" * 70)

    print("\nLoading data...")

    df = pd.read_parquet(INPUT_FILE)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True)

    print(f"Rows: {len(df):,}")
    print(f"Stocks: {df['symbol'].nunique()}")
    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Calculate features
    # --------------------------------------------------------

    print("\nCalculating Category 4 features...")

    df = (
        df.groupby(
            "symbol",
            group_keys=False
        )
        .apply(calculate_features)
        .reset_index(drop=True)
    )

    feature_output = df[
        [
            "date",
            "symbol",
            "next_day_return",
        ] + FEATURES
    ]

    feature_output.to_parquet(
        os.path.join(
            OUTPUT_DIR,
            "features_full.parquet"
        ),
        index=False
    )

    print("Feature calculation complete.")

    # --------------------------------------------------------
    # Daily RankIC
    # --------------------------------------------------------

    print("\nCalculating daily RankIC...")

    all_rankic = []

    for feature in FEATURES:

        print(f"  {feature}")

        feature_rankic = calculate_daily_rankic(
            df,
            feature
        )

        all_rankic.append(feature_rankic)

    daily_rankic = pd.concat(
        all_rankic,
        ignore_index=True
    )

    daily_rankic.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "daily_rankic.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # ICIR
    # --------------------------------------------------------

    print("\nCalculating ICIR...")

    icir_metrics = calculate_icir(
        daily_rankic
    )

    icir_metrics.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "icir_metrics.csv"
        ),
        index=False
    )

    print("\nICIR ranking:")
    print(
        icir_metrics[
            [
                "feature",
                "rank_ic",
                "icir",
                "abs_icir",
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Redundancy removal
    # --------------------------------------------------------

    print("\nRemoving redundant features...")

    selected_features, correlation_matrix, dropped = (
        select_non_redundant_features(
            df,
            icir_metrics
        )
    )

    # --------------------------------------------------------
    # Save selected features
    # --------------------------------------------------------

    selected_df = icir_metrics[
        icir_metrics["feature"].isin(
            selected_features
        )
    ].copy()

    selected_df = selected_df.sort_values(
        "abs_icir",
        ascending=False
    )

    selected_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "selected_features.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Phase 1 summary
    # --------------------------------------------------------

    summary = pd.DataFrame(
        {
            "category": ["Category 4"],
            "total_features": [len(FEATURES)],
            "selected_features": [len(selected_features)],
            "correlation_threshold": [
                CORRELATION_THRESHOLD
            ],
            "selection_start": [SELECTION_START],
            "selection_end": [SELECTION_END],
            "selected_feature_names": [
                ", ".join(selected_features)
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

    # --------------------------------------------------------
    # Final deliverable
    # --------------------------------------------------------

    final_deliverable = selected_df[
        [
            "feature",
            "rank_ic",
            "ic_std",
            "icir",
            "abs_icir",
            "n_days",
        ]
    ].copy()

    final_deliverable.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "final_deliverable.csv"
        ),
        index=False
    )

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CATEGORY 4 FINAL SELECTION")
    print("=" * 70)

    for i, feature in enumerate(
        selected_features,
        start=1
    ):
        print(f"{i}. {feature}")

    print(
        f"\nSelected {len(selected_features)} "
        f"out of {len(FEATURES)} features."
    )

    print(
        f"\nResults saved to:\n"
        f"{OUTPUT_DIR}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()