import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "p1_cat7")

os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FILE = os.path.join(
    DATA_DIR,
    "nifty50_stocks_ohlcv.parquet"
)


# ============================================================
# CONFIG
# ============================================================

SELECTION_START = "2022-01-01"
SELECTION_END = "2023-12-31"

CORR_THRESHOLD = 0.70
TARGET_FEATURES = 7

VOLATILITY_LOOKBACK = 20
RETURN_LOOKBACK = 20
BETA_LOOKBACK = 60
IDIOVOL_LOOKBACK = 60
RELATIVE_STRENGTH_LOOKBACK = 20


# ============================================================
# SECTOR INDEX FILES
# ============================================================

SECTOR_FILES = {
    "bank": os.path.join(DATA_DIR, "nifty_bank.parquet"),
    "it": os.path.join(DATA_DIR, "nifty_it.parquet"),
    "pharma": os.path.join(DATA_DIR, "nifty_pharma.parquet"),
    "auto": os.path.join(DATA_DIR, "nifty_auto.parquet"),
    "metal": os.path.join(DATA_DIR, "nifty_metal.parquet"),
    "energy": os.path.join(DATA_DIR, "nifty_energy.parquet"),
}


# ============================================================
# SECTOR MAP
# ============================================================
#
# The project specification requires Sector-Relative Return
# but does not provide a stock -> sector mapping.
#
# This mapping is therefore an implementation assumption.
#
# Stocks not assigned to one of the available sector indices
# will receive NaN for Sector-Relative Return.
#
# Do not silently substitute another sector index.
#

SECTOR_MAP = {

    # Banking
    "AXISBANK": "bank",
    "HDFCBANK": "bank",
    "ICICIBANK": "bank",
    "INDUSINDBK": "bank",
    "KOTAKBANK": "bank",
    "SBIN": "bank",

    # Information Technology
    "HCLTECH": "it",
    "INFY": "it",
    "TCS": "it",
    "TECHM": "it",
    "WIPRO": "it",

    # Pharma / Healthcare proxy
    "CIPLA": "pharma",
    "DRREDDY": "pharma",
    "SUNPHARMA": "pharma",

    # Auto
    "BAJAJ-AUTO": "auto",
    "EICHERMOT": "auto",
    "HEROMOTOCO": "auto",
    "M&M": "auto",
    "MARUTI": "auto",

    # Metals
    "HINDALCO": "metal",
    "JSWSTEEL": "metal",
    "TATASTEEL": "metal",

    # Energy
    "COALINDIA": "energy",
    "NTPC": "energy",
    "ONGC": "energy",
    "POWERGRID": "energy",
    "RELIANCE": "energy",
}


# ============================================================
# HELPERS
# ============================================================

def standardize_stock_data(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["symbol", "date"]
    ).reset_index(drop=True)

    return df


def load_index(path, name):
    if not os.path.exists(path):
        print(
            f"WARNING: Missing sector index file for {name}: "
            f"{path}"
        )
        return None

    df = pd.read_parquet(path)

    if df.empty:
        print(
            f"WARNING: Sector index file for {name} is empty: "
            f"{path}"
        )
        return None

    df = df.copy()

    # Find date column
    date_col = None

    for col in [
        "date",
        "Date",
        "datetime",
        "Datetime",
        "timestamp"
    ]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(
            f"No date column found in {path}"
        )

    df["date"] = pd.to_datetime(df[date_col])

    # Find close
    close_col = None

    for col in [
        "adj_close",
        "Adj Close",
        "adjusted_close",
        "close",
        "Close"
    ]:
        if col in df.columns:
            close_col = col
            break

    if close_col is None:
        raise ValueError(
            f"No close/adjusted close column found in {path}"
        )

    df = (
        df[["date", close_col]]
        .rename(columns={close_col: "index_close"})
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    df[f"{name}_return"] = (
        df["index_close"].pct_change()
    )

    df[f"{name}_return_20d"] = (
        df["index_close"].pct_change(RETURN_LOOKBACK)
    )

    return df[
        [
            "date",
            "index_close",
            f"{name}_return",
            f"{name}_return_20d"
        ]
    ]


def cross_sectional_zscore(series):
    mean = series.mean()
    std = series.std(ddof=0)

    if pd.isna(std) or std == 0:
        return np.nan

    return (series - mean) / std


def rank_percentile(series):
    return series.rank(
        method="average",
        pct=True
    )


def rolling_beta(stock_returns, market_returns):
    covariance = (
        stock_returns
        .rolling(BETA_LOOKBACK)
        .cov(market_returns)
    )

    market_variance = (
        market_returns
        .rolling(BETA_LOOKBACK)
        .var()
    )

    return covariance / market_variance


def rolling_idiosyncratic_vol(stock_returns, market_returns):
    beta = rolling_beta(
        stock_returns,
        market_returns
    )

    residual = (
        stock_returns
        - beta * market_returns
    )

    return residual.rolling(
        IDIOVOL_LOOKBACK
    ).std()


# ============================================================
# LOAD STOCK DATA
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY 7: CROSS-SECTIONAL RELATIVE FEATURES")
print("=" * 70)

print("\nLoading stock data...")

stocks = pd.read_parquet(INPUT_FILE)

stocks = standardize_stock_data(stocks)

print(f"Shape: {stocks.shape}")
print(f"Stocks: {stocks['symbol'].nunique()}")
print(
    f"Date range: "
    f"{stocks['date'].min().date()} -> "
    f"{stocks['date'].max().date()}"
)


# ============================================================
# BASIC STOCK FEATURES NEEDED AS INPUTS
# ============================================================

print("\nComputing base stock returns and volatility...")

stocks["daily_return"] = (
    stocks
    .groupby("symbol")["close"]
    .pct_change()
)

stocks["return_20d"] = (
    stocks
    .groupby("symbol")["close"]
    .pct_change(RETURN_LOOKBACK)
)

stocks["volume"] = stocks["volume"].astype(float)

stocks["volatility_20d"] = (
    stocks
    .groupby("symbol")["daily_return"]
    .transform(
        lambda x: x.rolling(
            VOLATILITY_LOOKBACK
        ).std()
    )
)


# ============================================================
# 1. CROSS-SECTIONAL RETURN Z-SCORE
# ============================================================

print("1/10 Cross-sectional return z-score...")

stocks["cross_sectional_return_zscore"] = (
    stocks
    .groupby("date")["daily_return"]
    .transform(cross_sectional_zscore)
)


# ============================================================
# 2. CROSS-SECTIONAL VOLUME Z-SCORE
# ============================================================

print("2/10 Cross-sectional volume z-score...")

stocks["cross_sectional_volume_zscore"] = (
    stocks
    .groupby("date")["volume"]
    .transform(cross_sectional_zscore)
)


# ============================================================
# 3. CROSS-SECTIONAL VOLATILITY Z-SCORE
# ============================================================

print("3/10 Cross-sectional volatility z-score...")

stocks["cross_sectional_volatility_zscore"] = (
    stocks
    .groupby("date")["volatility_20d"]
    .transform(cross_sectional_zscore)
)


# ============================================================
# 4. RETURN RANK PERCENTILE (20-DAY)
# ============================================================

print("4/10 Return rank percentile...")

stocks["return_rank_percentile_20d"] = (
    stocks
    .groupby("date")["return_20d"]
    .transform(rank_percentile)
)


# ============================================================
# 5. VOLUME RANK PERCENTILE
# ============================================================

print("5/10 Volume rank percentile...")

stocks["volume_rank_percentile"] = (
    stocks
    .groupby("date")["volume"]
    .transform(rank_percentile)
)


# ============================================================
# LOAD MARKET INDEX
# ============================================================

print("\nLoading NIFTY 50 index...")

nifty_path_candidates = [
    os.path.join(DATA_DIR, "nifty50.parquet"),
    os.path.join(DATA_DIR, "nifty_50.parquet"),
    os.path.join(DATA_DIR, "nifty50_index.parquet"),
    os.path.join(DATA_DIR, "nifty_index.parquet"),
]

nifty_path = None

for path in nifty_path_candidates:
    if os.path.exists(path):
        nifty_path = path
        break

if nifty_path is None:
    raise FileNotFoundError(
        "\nNIFTY 50 index data is required for Category 7.\n"
        "Expected one of:\n"
        + "\n".join(
            f"  {x}"
            for x in nifty_path_candidates
        )
    )

nifty = load_index(
    nifty_path,
    "nifty50"
)

nifty = nifty.rename(
    columns={
        "index_close": "nifty50_close",
        "nifty50_return": "nifty50_return",
        "nifty50_return_20d": "nifty50_return_20d"
    }
)


# ============================================================
# MERGE NIFTY DATA
# ============================================================

stocks = stocks.merge(
    nifty[
        [
            "date",
            "nifty50_close",
            "nifty50_return",
            "nifty50_return_20d"
        ]
    ],
    on="date",
    how="left"
)


# ============================================================
# 6. SECTOR-RELATIVE RETURN
# ============================================================

print("\n6/10 Sector-relative return...")

sector_data = {}

for sector, path in SECTOR_FILES.items():

    sector_df = load_index(
        path,
        sector
    )

    if sector_df is not None:
        sector_data[sector] = sector_df

        stocks = stocks.merge(
            sector_df[
                [
                    "date",
                    f"{sector}_return_20d"
                ]
            ],
            on="date",
            how="left"
        )


stocks["sector_relative_return"] = np.nan

for symbol, sector in SECTOR_MAP.items():

    if sector not in sector_data:
        continue

    sector_column = (
        f"{sector}_return_20d"
    )

    mask = stocks["symbol"] == symbol

    stocks.loc[mask, "sector_relative_return"] = (
        stocks.loc[mask, "return_20d"]
        - stocks.loc[mask, sector_column]
    )


# ============================================================
# 7. MARKET-RELATIVE RETURN
# ============================================================

print("7/10 Market-relative return...")

stocks["market_relative_return"] = (
    stocks["return_20d"]
    - stocks["nifty50_return_20d"]
)


# ============================================================
# 8. BETA (60-DAY)
# ============================================================

print("8/10 Beta (60-day)...")

stocks["beta_60d"] = np.nan

for symbol, group in stocks.groupby("symbol"):

    idx = group.index

    stock_ret = group["daily_return"]
    market_ret = group["nifty50_return"]

    beta = rolling_beta(
        stock_ret,
        market_ret
    )

    stocks.loc[idx, "beta_60d"] = (
        beta.values
    )


# ============================================================
# 9. IDIOSYNCRATIC VOLATILITY
# ============================================================

print("9/10 Idiosyncratic volatility...")

stocks["idiosyncratic_volatility"] = np.nan

for symbol, group in stocks.groupby("symbol"):

    idx = group.index

    stock_ret = group["daily_return"]
    market_ret = group["nifty50_return"]

    idio_vol = rolling_idiosyncratic_vol(
        stock_ret,
        market_ret
    )

    stocks.loc[idx, "idiosyncratic_volatility"] = (
        idio_vol.values
    )


# ============================================================
# 10. RELATIVE STRENGTH VS NIFTY
# ============================================================
#
# Definition used here:
#
# Relative strength ratio = stock price / NIFTY price
# Relative strength feature = 20-day rate of change
# of that ratio.
#
# The project specifies the concept but does not specify
# the lookback horizon, so 20 days is an implementation
# assumption.
# ============================================================

print("10/10 Relative strength vs NIFTY...")

stocks["stock_nifty_ratio"] = (
    stocks["close"]
    / stocks["nifty50_close"]
)

stocks["relative_strength_vs_nifty"] = (
    stocks
    .groupby("symbol")["stock_nifty_ratio"]
    .pct_change(RELATIVE_STRENGTH_LOOKBACK)
)


# ============================================================
# FINAL FEATURE LIST
# ============================================================

FEATURES = [
    "cross_sectional_return_zscore",
    "cross_sectional_volume_zscore",
    "cross_sectional_volatility_zscore",
    "return_rank_percentile_20d",
    "volume_rank_percentile",
    "sector_relative_return",
    "market_relative_return",
    "beta_60d",
    "idiosyncratic_volatility",
    "relative_strength_vs_nifty",
]


# ============================================================
# NEXT-DAY STOCK RETURN
# ============================================================

print("\nComputing next-day returns...")

stocks["next_day_return"] = (
    stocks
    .groupby("symbol")["close"]
    .shift(-1)
    / stocks["close"]
    - 1
)


# ============================================================
# SAVE FULL FEATURES
# ============================================================

features_full = stocks[
    ["date", "symbol"] + FEATURES
].copy()

features_full.to_parquet(
    os.path.join(
        OUTPUT_DIR,
        "features_full.parquet"
    ),
    index=False
)


# ============================================================
# SELECTION PERIOD
# ============================================================

selection = stocks[
    (stocks["date"] >= SELECTION_START)
    & (stocks["date"] <= SELECTION_END)
].copy()


# ============================================================
# DAILY CROSS-SECTIONAL RANKIC
# ============================================================

print("\nCalculating daily cross-sectional RankIC...")

daily_ic_records = []


for date, day in selection.groupby("date"):

    for feature in FEATURES:

        temp = day[
            [feature, "next_day_return"]
        ].dropna()

        # Need enough stocks to make a meaningful
        # cross-sectional correlation.
        if len(temp) < 3:
            continue

        # Spearman correlation is undefined if either
        # feature or target is constant.
        if temp[feature].nunique() < 2:
            continue

        if temp["next_day_return"].nunique() < 2:
            continue

        ic = temp[feature].corr(
            temp["next_day_return"],
            method="spearman"
        )

        daily_ic_records.append({
            "date": date,
            "feature": feature,
            "IC": ic,
            "n_stocks": len(temp)
        })


daily_rankic = pd.DataFrame(
    daily_ic_records
)


# ============================================================
# ICIR METRICS
# ============================================================

print("Calculating RankIC and ICIR...")

icir_rows = []

for feature in FEATURES:

    values = daily_rankic.loc[
        daily_rankic["feature"] == feature,
        "IC"
    ].dropna()

    if len(values) == 0:
        rank_ic = np.nan
        icir = np.nan

    else:
        rank_ic = values.mean()

        std_ic = values.std(ddof=1)

        if std_ic == 0 or pd.isna(std_ic):
            icir = np.nan
        else:
            icir = rank_ic / std_ic

    icir_rows.append({
        "feature": feature,
        "RankIC": rank_ic,
        "ICIR": icir,
        "n_daily_ic": len(values)
    })


icir_metrics = pd.DataFrame(icir_rows)

icir_metrics = (
    icir_metrics
    .assign(
        abs_ICIR=lambda x: x["ICIR"].abs()
    )
    .sort_values(
        "abs_ICIR",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE RANKIC OUTPUTS
# ============================================================

daily_rankic.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "daily_rankic.csv"
    ),
    index=False
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

print("\nCalculating within-category Spearman correlation...")

corr_data = selection[FEATURES]

spearman_corr = corr_data.corr(
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

print("Removing highly correlated features...")

ranking = icir_metrics.copy()

selected = []
dropped = []


for feature in ranking["feature"]:

    keep = True

    for existing in selected:

        corr = spearman_corr.loc[
            feature,
            existing
        ]

        if (
            pd.notna(corr)
            and abs(corr) > CORR_THRESHOLD
        ):

            keep = False

            dropped.append({
                "feature": feature,
                "correlated_with": existing,
                "correlation": corr,
                "feature_abs_ICIR": abs(
                    ranking.loc[
                        ranking["feature"] == feature,
                        "ICIR"
                    ].iloc[0]
                ),
                "correlated_feature_abs_ICIR": abs(
                    ranking.loc[
                        ranking["feature"] == existing,
                        "ICIR"
                    ].iloc[0]
                ),
                "reason": (
                    "Dropped because correlation exceeded "
                    "threshold and correlated feature had "
                    "higher absolute ICIR"
                )
            })

            break

    if keep:
        selected.append(feature)


# ============================================================
# TARGET 6-7 FEATURES
# ============================================================

selected = selected[:TARGET_FEATURES]


# ============================================================
# SELECTED FEATURES
# ============================================================

selected_df = icir_metrics[
    icir_metrics["feature"].isin(selected)
].copy()

selected_df["selection_rank"] = (
    selected_df["feature"].map(
        {
            feature: i + 1
            for i, feature in enumerate(selected)
        }
    )
)

selected_df["verdict"] = "KEEP"

selected_df = selected_df.sort_values(
    "selection_rank"
)


# ============================================================
# DROPPED FEATURES
# ============================================================

dropped_df = pd.DataFrame(dropped)

dropped_names = set(
    dropped_df["feature"]
    if not dropped_df.empty
    else []
)

low_rank_features = set(
    FEATURES
) - set(selected) - dropped_names


# Add non-correlated features that were outside target
# because we only keep the top 7.
for feature in low_rank_features:

    row = icir_metrics[
        icir_metrics["feature"] == feature
    ]

    if len(row) == 0:
        continue

    dropped.append({
        "feature": feature,
        "correlated_with": "",
        "correlation": np.nan,
        "feature_abs_ICIR": abs(
            row["ICIR"].iloc[0]
        ),
        "correlated_feature_abs_ICIR": np.nan,
        "reason": (
            "Not selected because target is "
            "top 7 non-redundant features"
        )
    })


dropped_df = pd.DataFrame(dropped)

dropped_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dropped_correlated.csv"
    ),
    index=False
)

selected_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "selected_features.csv"
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
        "cross_sectional_unit",
        "rankic_method",
        "beta_lookback",
        "idiosyncratic_volatility_lookback",
        "relative_strength_lookback"
    ],
    "value": [
        "Category 7 - Cross-Sectional",
        len(FEATURES),
        SELECTION_START,
        SELECTION_END,
        CORR_THRESHOLD,
        TARGET_FEATURES,
        len(selected),
        "All available stocks on the same trading day",
        "Daily Spearman correlation with next-day stock return",
        BETA_LOOKBACK,
        IDIOVOL_LOOKBACK,
        RELATIVE_STRENGTH_LOOKBACK
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
        "n_daily_ic",
        "verdict"
    ]
].copy()

final_deliverable.insert(
    2,
    "category",
    "Cross-Sectional"
)

final_deliverable.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_deliverable.csv"
    ),
    index=False
)


# ============================================================
# DATA QUALITY REPORT
# ============================================================

quality_rows = []

for feature in FEATURES:

    missing = features_full[feature].isna().sum()
    total = len(features_full)

    quality_rows.append({
        "feature": feature,
        "total_rows": total,
        "missing_rows": missing,
        "missing_pct": 100 * missing / total
    })


quality_df = pd.DataFrame(quality_rows)

quality_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "data_quality.csv"
    ),
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY 7 COMPLETE")
print("=" * 70)

print("\nSelected features:")

print(
    final_deliverable.to_string(
        index=False
    )
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFiles created:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    print(f"  {filename}")

print("\n" + "=" * 70)