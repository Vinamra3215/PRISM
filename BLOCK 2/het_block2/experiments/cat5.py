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
    "p1_cat5"
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
# CATEGORY 5 FEATURES
# ============================================================

FEATURES = [
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "body_ratio",
    "candle_direction",
    "consecutive_up_days",
    "consecutive_down_days",
    "bollinger_pct_b",
    "bollinger_band_width",
    "keltner_channel_position",
    "donchian_channel_position",
]


# ============================================================
# FEATURE CALCULATION
# ============================================================

def calculate_features(group):

    group = group.sort_values("date").copy()

    open_price = group["open"]
    high = group["high"]
    low = group["low"]
    close = group["close"]

    # --------------------------------------------------------
    # Basic candle quantities
    # --------------------------------------------------------

    candle_range = high - low
    body = (close - open_price).abs()

    upper_shadow = (
        high - pd.concat(
            [open_price, close],
            axis=1
        ).max(axis=1)
    )

    lower_shadow = (
        pd.concat(
            [open_price, close],
            axis=1
        ).min(axis=1) - low
    )

    safe_range = candle_range.replace(0, np.nan)

    # --------------------------------------------------------
    # 1. Upper Shadow Ratio
    # --------------------------------------------------------

    group["upper_shadow_ratio"] = (
        upper_shadow / safe_range
    )

    # --------------------------------------------------------
    # 2. Lower Shadow Ratio
    # --------------------------------------------------------

    group["lower_shadow_ratio"] = (
        lower_shadow / safe_range
    )

    # --------------------------------------------------------
    # 3. Body Ratio
    # --------------------------------------------------------

    group["body_ratio"] = (
        body / safe_range
    )

    # --------------------------------------------------------
    # 4. Candle Direction
    #
    # +1 bullish
    # -1 bearish
    #  0 unchanged
    # --------------------------------------------------------

    group["candle_direction"] = np.sign(
        close - open_price
    )

    # --------------------------------------------------------
    # 5. Consecutive Up Days
    # --------------------------------------------------------

    positive = close.diff() > 0

    up_count = []
    count = 0

    for value in positive:

        if value:
            count += 1
        else:
            count = 0

        up_count.append(count)

    group["consecutive_up_days"] = up_count

    # --------------------------------------------------------
    # 6. Consecutive Down Days
    # --------------------------------------------------------

    negative = close.diff() < 0

    down_count = []
    count = 0

    for value in negative:

        if value:
            count += 1
        else:
            count = 0

        down_count.append(count)

    group["consecutive_down_days"] = down_count

    # --------------------------------------------------------
    # 7. Bollinger Band %B
    #
    # 20-day SMA
    # 20-day standard deviation
    #
    # %B = (Close - Lower Band) /
    #       (Upper Band - Lower Band)
    # --------------------------------------------------------

    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()

    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20

    band_range = (
        upper_band - lower_band
    ).replace(0, np.nan)

    group["bollinger_pct_b"] = (
        (close - lower_band) /
        band_range
    )

    # --------------------------------------------------------
    # 8. Bollinger Band Width
    #
    # Width = (Upper - Lower) / Middle
    # --------------------------------------------------------

    group["bollinger_band_width"] = (
        (upper_band - lower_band) /
        sma_20.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # ATR for Keltner Channel
    # --------------------------------------------------------

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr_20 = true_range.rolling(20).mean()

    # --------------------------------------------------------
    # Keltner Channel
    #
    # Middle = EMA 20
    # Upper = Middle + 2 * ATR
    # Lower = Middle - 2 * ATR
    #
    # Position = (Close - Lower) /
    #            (Upper - Lower)
    # --------------------------------------------------------

    ema_20 = close.ewm(
        span=20,
        adjust=False,
        min_periods=20
    ).mean()

    keltner_upper = ema_20 + 2 * atr_20
    keltner_lower = ema_20 - 2 * atr_20

    keltner_range = (
        keltner_upper - keltner_lower
    ).replace(0, np.nan)

    group["keltner_channel_position"] = (
        (close - keltner_lower) /
        keltner_range
    )

    # --------------------------------------------------------
    # 10. Donchian Channel Position
    #
    # 20-day high/low range
    # --------------------------------------------------------

    donchian_high = high.rolling(20).max()
    donchian_low = low.rolling(20).min()

    donchian_range = (
        donchian_high - donchian_low
    ).replace(0, np.nan)

    group["donchian_channel_position"] = (
        (close - donchian_low) /
        donchian_range
    )

    # --------------------------------------------------------
    # Next-day return
    # --------------------------------------------------------

    group["next_day_return"] = (
        close.shift(-1) / close - 1
    )

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

        valid = day_data[
            [feature, "next_day_return"]
        ].dropna()

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
# ICIR
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
                "abs_icir": (
                    abs(icir)
                    if np.isfinite(icir)
                    else np.nan
                ),
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

def select_non_redundant_features(
    df,
    icir_metrics
):

    selection_df = df[
        (df["date"] >= SELECTION_START) &
        (df["date"] <= SELECTION_END)
    ].copy()

    feature_data = selection_df[FEATURES]

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
        .sort_values(
            "abs_icir",
            ascending=False
        )["feature"]
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

            if (
                pd.notna(corr)
                and abs(corr) > CORRELATION_THRESHOLD
            ):
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

            if feature_icir > selected_icir:

                selected.remove(
                    redundant_with
                )

                selected.append(
                    feature
                )

                dropped.append(
                    {
                        "dropped_feature":
                            redundant_with,
                        "kept_feature":
                            feature,
                        "correlation":
                            correlation_matrix.loc[
                                feature,
                                redundant_with
                            ],
                        "dropped_abs_icir":
                            selected_icir,
                        "kept_abs_icir":
                            feature_icir,
                    }
                )

            else:

                dropped.append(
                    {
                        "dropped_feature":
                            feature,
                        "kept_feature":
                            redundant_with,
                        "correlation":
                            correlation_matrix.loc[
                                feature,
                                redundant_with
                            ],
                        "dropped_abs_icir":
                            feature_icir,
                        "kept_abs_icir":
                            selected_icir,
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

    # Keep strongest 7 non-redundant features
    selected_metrics = icir_metrics[
        icir_metrics["feature"].isin(
            selected
        )
    ].sort_values(
        "abs_icir",
        ascending=False
    )

    selected_final = selected_metrics.head(
        TARGET_FEATURES
    )["feature"].tolist()

    return (
        selected_final,
        correlation_matrix,
        dropped_df
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CATEGORY 5: CANDLESTICK & PRICE STRUCTURE FEATURES")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading data...")

    df = pd.read_parquet(
        INPUT_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = df.sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True)

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Stocks: {df['symbol'].nunique()}"
    )

    print(
        f"Date range: "
        f"{df['date'].min().date()} → "
        f"{df['date'].max().date()}"
    )

    # --------------------------------------------------------
    # Calculate features
    # --------------------------------------------------------

    print(
        "\nCalculating Category 5 features..."
    )

    df = (
        df.groupby(
            "symbol",
            group_keys=False
        )
        .apply(
            calculate_features
        )
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

    print(
        "Feature calculation complete."
    )

    # --------------------------------------------------------
    # Daily RankIC
    # --------------------------------------------------------

    print(
        "\nCalculating daily RankIC..."
    )

    all_rankic = []

    for feature in FEATURES:

        print(
            f"  {feature}"
        )

        feature_rankic = (
            calculate_daily_rankic(
                df,
                feature
            )
        )

        all_rankic.append(
            feature_rankic
        )

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

    print(
        "\nCalculating ICIR..."
    )

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

    print(
        "\nICIR ranking:"
    )

    print(
        icir_metrics[
            [
                "feature",
                "rank_ic",
                "icir",
                "abs_icir",
            ]
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Redundancy removal
    # --------------------------------------------------------

    print(
        "\nRemoving redundant features..."
    )

    selected_features, _, dropped = (
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
            "category": [
                "Category 5"
            ],
            "total_features": [
                len(FEATURES)
            ],
            "selected_features": [
                len(selected_features)
            ],
            "correlation_threshold": [
                CORRELATION_THRESHOLD
            ],
            "selection_start": [
                SELECTION_START
            ],
            "selection_end": [
                SELECTION_END
            ],
            "selected_feature_names": [
                ", ".join(
                    selected_features
                )
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
    # Final output
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "CATEGORY 5 FINAL SELECTION"
    )

    print(
        "=" * 70
    )

    for i, feature in enumerate(
        selected_features,
        start=1
    ):
        print(
            f"{i}. {feature}"
        )

    print(
        f"\nSelected "
        f"{len(selected_features)} "
        f"out of "
        f"{len(FEATURES)} "
        f"features."
    )

    print(
        f"\nResults saved to:"
        f"\n{OUTPUT_DIR}"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()