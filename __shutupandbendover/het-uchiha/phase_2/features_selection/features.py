import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

STOCK_FILE = BASE_DIR / "nifty50_stocks_ohlcv.parquet"
NIFTY_FILE = BASE_DIR / "nifty_ohlcv.parquet"

FEATURE_OUTPUT = BASE_DIR / "category7_features.parquet"
ICIR_OUTPUT = BASE_DIR / "category7_icir.csv"
CORR_OUTPUT = BASE_DIR / "category7_correlation.csv"
SELECTED_OUTPUT = BASE_DIR / "category7_selected.csv"

SELECTION_START = "2022-01-01"
SELECTION_END = "2023-12-31"

ICIR_THRESHOLD = 0.15
CORR_THRESHOLD = 0.70


# ============================================================
# SECTOR MAPPING
# ============================================================
# Sector return = equal-weighted average return of stocks
# belonging to that sector.

SECTOR_MAP = {
    "MAXHEALTH": "Healthcare",
    "ADANIENT": "Industrials",
    "ADANIPORTS": "Industrials",
    "APOLLOHOSP": "Healthcare",
    "ASIANPAINT": "Consumer",
    "AXISBANK": "Financials",
    "BAJAJ-AUTO": "Automobile",
    "BAJFINANCE": "Financials",
    "BAJAJFINSV": "Financials",
    "BEL": "Industrials",
    "BHARTIARTL": "Telecom",
    "CIPLA": "Healthcare",
    "COALINDIA": "Energy",
    "DRREDDY": "Healthcare",
    "EICHERMOT": "Automobile",
    "ETERNAL": "Consumer",
    "GRASIM": "Industrials",
    "HCLTECH": "IT",
    "HDFCBANK": "Financials",
    "HDFCLIFE": "Financials",
    "HEROMOTOCO": "Automobile",
    "HINDALCO": "Materials",
    "HINDUNILVR": "Consumer",
    "ICICIBANK": "Financials",
    "INDUSINDBK": "Financials",
    "INFY": "IT",
    "ITC": "Consumer",
    "JIOFIN": "Financials",
    "JSWSTEEL": "Materials",
    "KOTAKBANK": "Financials",
    "LT": "Industrials",
    "M&M": "Automobile",
    "MARUTI": "Automobile",
    "NESTLEIND": "Consumer",
    "NTPC": "Utilities",
    "ONGC": "Energy",
    "POWERGRID": "Utilities",
    "RELIANCE": "Energy",
    "SBILIFE": "Financials",
    "SBIN": "Financials",
    "SHRIRAMFIN": "Financials",
    "SUNPHARMA": "Healthcare",
    "TATACONSUM": "Consumer",
    "TATAMOTORS": "Automobile",
    "TATASTEEL": "Materials",
    "TCS": "IT",
    "TECHM": "IT",
    "TITAN": "Consumer",
    "TRENT": "Consumer",
    "ULTRACEMCO": "Materials",
    "WIPRO": "IT",
}


# ============================================================
# HELPERS
# ============================================================

def zscore_cross_section(series):
    mean = series.mean()
    std = series.std(ddof=0)

    if std == 0 or pd.isna(std):
        return pd.Series(np.nan, index=series.index)

    return (series - mean) / std


def rank_percentile(series):
    return series.rank(
        pct=True,
        method="average"
    )


def rolling_beta(stock_returns, market_returns, window=60):
    covariance = (
        stock_returns
        .rolling(window)
        .cov(market_returns)
    )

    market_variance = (
        market_returns
        .rolling(window)
        .var()
    )

    return covariance / market_variance


# ============================================================
# START
# ============================================================

print("=" * 70)
print("CATEGORY 7: CROSS-SECTIONAL RELATIVE FEATURES")
print("=" * 70)


# ============================================================
# LOAD STOCK DATA
# ============================================================

print("\nLoading stock data...")

stocks = pd.read_parquet(STOCK_FILE)

stocks["date"] = pd.to_datetime(
    stocks["date"]
)

stocks = (
    stocks
    .sort_values(
        ["symbol", "date"]
    )
    .reset_index(drop=True)
)

required_stock_cols = {
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

missing = (
    required_stock_cols
    - set(stocks.columns)
)

if missing:
    raise ValueError(
        f"Missing required stock columns: "
        f"{sorted(missing)}"
    )

print(
    f"Stocks: {stocks['symbol'].nunique()}"
)

print(
    f"Rows:   {len(stocks):,}"
)

print(
    f"Dates:  "
    f"{stocks['date'].min().date()} -> "
    f"{stocks['date'].max().date()}"
)


# ============================================================
# LOAD NIFTY DATA
# ============================================================

print("\nLoading NIFTY data...")

nifty = pd.read_parquet(
    NIFTY_FILE
)

# ------------------------------------------------------------
# IMPORTANT:
# In nifty_ohlcv.parquet, Date is the INDEX.
# Columns are:
# open, high, low, close, volume
# ------------------------------------------------------------

nifty = nifty.reset_index()

# After reset_index(), the index is normally called "Date".
# Handle both Date and date just in case.
if "Date" in nifty.columns:
    nifty = nifty.rename(
        columns={"Date": "date"}
    )

elif "date" not in nifty.columns:
    raise ValueError(
        "Could not find NIFTY date information. "
        f"Columns found: {nifty.columns.tolist()}"
    )

nifty["date"] = pd.to_datetime(
    nifty["date"]
)

nifty = (
    nifty
    .sort_values("date")
    .reset_index(drop=True)
)

if "close" not in nifty.columns:
    raise ValueError(
        "NIFTY file does not contain "
        f"'close'. Columns found: "
        f"{nifty.columns.tolist()}"
    )

nifty = (
    nifty[
        [
            "date",
            "close",
        ]
    ]
    .drop_duplicates("date")
    .copy()
)

nifty = nifty.rename(
    columns={
        "close": "nifty_close"
    }
)

nifty["nifty_return_1d"] = (
    nifty["nifty_close"]
    .pct_change()
)

print(
    f"NIFTY rows: {len(nifty):,}"
)

print(
    f"NIFTY dates: "
    f"{nifty['date'].min().date()} -> "
    f"{nifty['date'].max().date()}"
)


# ============================================================
# BASIC STOCK RETURNS
# ============================================================

print("\nCalculating stock returns...")

stocks["return_1d"] = (
    stocks
    .groupby("symbol")["close"]
    .pct_change()
)

stocks["return_20d"] = (
    stocks
    .groupby("symbol")["close"]
    .pct_change(20)
)


# ============================================================
# FEATURE 1
# CROSS-SECTIONAL RETURN Z-SCORE
# ============================================================

print(
    "\nFeature 1: "
    "Cross-Sectional Return Z-Score"
)

stocks["cs_return_z"] = (
    stocks
    .groupby("date")["return_1d"]
    .transform(
        zscore_cross_section
    )
)


# ============================================================
# FEATURE 2
# CROSS-SECTIONAL VOLUME Z-SCORE
# ============================================================

print(
    "Feature 2: "
    "Cross-Sectional Volume Z-Score"
)

stocks["cs_volume_z"] = (
    stocks
    .groupby("date")["volume"]
    .transform(
        zscore_cross_section
    )
)


# ============================================================
# FEATURE 3
# CROSS-SECTIONAL VOLATILITY Z-SCORE
# ============================================================

print(
    "Feature 3: "
    "Cross-Sectional Volatility Z-Score"
)

stocks["volatility_20d"] = (
    stocks
    .groupby("symbol")["return_1d"]
    .transform(
        lambda x:
            x.rolling(
                20,
                min_periods=20
            ).std()
    )
)

stocks["cs_volatility_z"] = (
    stocks
    .groupby("date")["volatility_20d"]
    .transform(
        zscore_cross_section
    )
)


# ============================================================
# FEATURE 4
# RETURN RANK PERCENTILE (20-DAY)
# ============================================================

print(
    "Feature 4: "
    "Return Rank Percentile (20-day)"
)

stocks["return_rank_pct_20d"] = (
    stocks
    .groupby("date")["return_20d"]
    .transform(
        rank_percentile
    )
)


# ============================================================
# FEATURE 5
# VOLUME RANK PERCENTILE
# ============================================================

print(
    "Feature 5: "
    "Volume Rank Percentile"
)

stocks["volume_rank_pct"] = (
    stocks
    .groupby("date")["volume"]
    .transform(
        rank_percentile
    )
)


# ============================================================
# SECTOR MAPPING
# ============================================================

print("\nMapping sectors...")

stocks["sector"] = (
    stocks["symbol"]
    .map(SECTOR_MAP)
)

missing_sector_symbols = sorted(
    stocks.loc[
        stocks["sector"].isna(),
        "symbol"
    ]
    .dropna()
    .unique()
)

if missing_sector_symbols:
    raise ValueError(
        "Missing sector mapping for:\n"
        + "\n".join(
            missing_sector_symbols
        )
    )


# ============================================================
# SECTOR DAILY RETURNS
# ============================================================

print(
    "Calculating sector returns..."
)

sector_daily = (
    stocks
    .dropna(
        subset=["return_1d"]
    )
    .groupby(
        [
            "date",
            "sector",
        ]
    )["return_1d"]
    .mean()
    .reset_index(
        name="sector_return_1d"
    )
)

sector_daily = (
    sector_daily
    .sort_values(
        [
            "sector",
            "date",
        ]
    )
    .reset_index(drop=True)
)

# 20-day compounded sector return
sector_daily["sector_return_20d"] = (
    sector_daily
    .groupby("sector")[
        "sector_return_1d"
    ]
    .transform(
        lambda x:
            (
                (1 + x)
                .rolling(
                    20,
                    min_periods=20
                )
                .apply(
                    np.prod,
                    raw=True
                )
            )
            - 1
    )
)


# ============================================================
# FEATURE 6
# SECTOR-RELATIVE RETURN
# ============================================================

print(
    "Feature 6: "
    "Sector-Relative Return"
)

stocks = stocks.merge(
    sector_daily[
        [
            "date",
            "sector",
            "sector_return_20d",
        ]
    ],
    on=[
        "date",
        "sector",
    ],
    how="left",
)

stocks["sector_relative_return"] = (
    stocks["return_20d"]
    - stocks["sector_return_20d"]
)


# ============================================================
# MERGE NIFTY
# ============================================================

print(
    "\nMerging NIFTY data..."
)

stocks = stocks.merge(
    nifty[
        [
            "date",
            "nifty_close",
            "nifty_return_1d",
        ]
    ],
    on="date",
    how="left",
)


# ============================================================
# NIFTY 20-DAY RETURN
# ============================================================

nifty["nifty_return_20d"] = (
    nifty["nifty_close"]
    .pct_change(20)
)


# ============================================================
# FEATURE 7
# MARKET-RELATIVE RETURN
# ============================================================

print(
    "Feature 7: "
    "Market-Relative Return"
)

stocks = stocks.merge(
    nifty[
        [
            "date",
            "nifty_return_20d",
        ]
    ],
    on="date",
    how="left",
)

stocks["market_relative_return"] = (
    stocks["return_20d"]
    - stocks["nifty_return_20d"]
)


# ============================================================
# FEATURE 8
# BETA (60-DAY)
# ============================================================

print(
    "Feature 8: Beta (60-day)"
)

stocks = (
    stocks
    .sort_values(
        [
            "symbol",
            "date",
        ]
    )
    .reset_index(drop=True)
)

stocks["beta_60d"] = np.nan

for symbol, idx in stocks.groupby(
    "symbol"
).groups.items():

    stock_returns = (
        stocks
        .loc[idx, "return_1d"]
        .reset_index(drop=True)
    )

    market_returns = (
        stocks
        .loc[
            idx,
            "nifty_return_1d"
        ]
        .reset_index(drop=True)
    )

    beta = rolling_beta(
        stock_returns,
        market_returns,
        window=60
    )

    stocks.loc[
        idx,
        "beta_60d"
    ] = beta.to_numpy()


# ============================================================
# FEATURE 9
# IDIOSYNCRATIC VOLATILITY
# ============================================================

print(
    "Feature 9: "
    "Idiosyncratic Volatility"
)

stocks["residual_return"] = (
    stocks["return_1d"]
    - (
        stocks["beta_60d"]
        * stocks["nifty_return_1d"]
    )
)

stocks[
    "idiosyncratic_volatility"
] = (
    stocks
    .groupby("symbol")[
        "residual_return"
    ]
    .transform(
        lambda x:
            x.rolling(
                60,
                min_periods=60
            ).std()
    )
)


# ============================================================
# FEATURE 10
# RELATIVE STRENGTH VS NIFTY
# ============================================================

print(
    "Feature 10: "
    "Relative Strength vs NIFTY"
)

stocks["stock_nifty_ratio"] = (
    stocks["close"]
    / stocks["nifty_close"]
)

stocks[
    "relative_strength_vs_nifty"
] = (
    stocks
    .groupby("symbol")[
        "stock_nifty_ratio"
    ]
    .pct_change(20)
)


# ============================================================
# NEXT-DAY RETURN
# ============================================================

print(
    "\nCalculating next-day returns..."
)

stocks["next_day_return"] = (
    stocks
    .groupby("symbol")["close"]
    .shift(-1)
    / stocks["close"]
    - 1
)


# ============================================================
# FEATURE LIST
# ============================================================

FEATURES = [
    "cs_return_z",
    "cs_volume_z",
    "cs_volatility_z",
    "return_rank_pct_20d",
    "volume_rank_pct",
    "sector_relative_return",
    "market_relative_return",
    "beta_60d",
    "idiosyncratic_volatility",
    "relative_strength_vs_nifty",
]


# ============================================================
# SAVE FULL FEATURE DATA
# ============================================================

print(
    "\nSaving complete feature dataset..."
)

output_columns = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "sector",
    *FEATURES,
    "next_day_return",
]

stocks[
    output_columns
].to_parquet(
    FEATURE_OUTPUT,
    index=False
)

print(
    f"Saved: {FEATURE_OUTPUT}"
)


# ============================================================
# PHASE 2
# RANKIC / ICIR
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PHASE 2: RANKIC / ICIR"
)

print(
    "=" * 70
)

selection = stocks[
    (
        stocks["date"]
        >= SELECTION_START
    )
    &
    (
        stocks["date"]
        <= SELECTION_END
    )
].copy()


def calculate_daily_rank_ic(
    group,
    feature
):

    temp = group[
        [
            feature,
            "next_day_return",
        ]
    ].dropna()

    if len(temp) < 3:
        return np.nan

    if (
        temp[feature]
        .nunique()
        < 2
    ):
        return np.nan

    if (
        temp["next_day_return"]
        .nunique()
        < 2
    ):
        return np.nan

    return spearmanr(
        temp[feature],
        temp["next_day_return"]
    ).statistic


icir_results = []

daily_ic_all = {}

for feature in FEATURES:

    print(
        f"Calculating ICIR: "
        f"{feature}"
    )

    daily_ic = (
        selection
        .groupby("date")
        .apply(
            lambda g:
                calculate_daily_rank_ic(
                    g,
                    feature
                ),
            include_groups=False
        )
        .dropna()
    )

    daily_ic_all[
        feature
    ] = daily_ic

    if len(daily_ic) == 0:

        rank_ic = np.nan
        ic_std = np.nan
        icir = np.nan

    else:

        rank_ic = (
            daily_ic.mean()
        )

        ic_std = (
            daily_ic.std(
                ddof=1
            )
        )

        if (
            ic_std == 0
            or pd.isna(ic_std)
        ):
            icir = np.nan

        else:
            icir = (
                rank_ic
                / ic_std
            )

    icir_results.append(
        {
            "feature": feature,
            "rank_ic": rank_ic,
            "ic_std": ic_std,
            "icir": icir,
            "n_days": len(
                daily_ic
            ),
        }
    )


icir_df = pd.DataFrame(
    icir_results
)

icir_df["abs_icir"] = (
    icir_df["icir"].abs()
)

icir_df = (
    icir_df
    .sort_values(
        "abs_icir",
        ascending=False
    )
    .reset_index(drop=True)
)

icir_df.to_csv(
    ICIR_OUTPUT,
    index=False
)

print(
    f"\nSaved: {ICIR_OUTPUT}"
)

print(
    "\nICIR RESULTS"
)

print(
    "-" * 70
)

print(
    icir_df[
        [
            "feature",
            "rank_ic",
            "icir",
            "n_days",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# PHASE 3
# WITHIN-CATEGORY SPEARMAN CORRELATION
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "PHASE 3: "
    "WITHIN-CATEGORY CORRELATION"
)

print(
    "=" * 70
)

corr_data = selection[
    FEATURES
].copy()

correlation = (
    corr_data.corr(
        method="spearman"
    )
)

correlation.to_csv(
    CORR_OUTPUT
)

print(
    f"Saved: {CORR_OUTPUT}"
)


# ============================================================
# ICIR FILTER
# ============================================================

print(
    "\nFiltering features by ICIR..."
)

icir_pass = set(
    icir_df.loc[
        icir_df["abs_icir"]
        >= ICIR_THRESHOLD,
        "feature",
    ]
)

print(
    f"Features passing "
    f"|ICIR| >= "
    f"{ICIR_THRESHOLD}: "
    f"{len(icir_pass)}"
)

for feature in FEATURES:

    row = icir_df[
        icir_df["feature"]
        == feature
    ].iloc[0]

    status = (
        "KEEP"
        if feature in icir_pass
        else "DROP"
    )

    print(
        f"{status:4s}  "
        f"{feature:35s} "
        f"ICIR={row['icir']:.6f}"
    )


# ============================================================
# CORRELATION FILTER
# ============================================================

print(
    "\nApplying correlation filter..."
)

selected_features = list(
    icir_df[
        icir_df["feature"].isin(
            icir_pass
        )
    ]
    .sort_values(
        "abs_icir",
        ascending=False
    )["feature"]
)

dropped_redundant = []

for i in range(
    len(selected_features)
):

    feature_a = (
        selected_features[i]
    )

    if (
        feature_a
        in dropped_redundant
    ):
        continue

    for j in range(
        i + 1,
        len(selected_features)
    ):

        feature_b = (
            selected_features[j]
        )

        if (
            feature_b
            in dropped_redundant
        ):
            continue

        corr_value = (
            correlation.loc[
                feature_a,
                feature_b,
            ]
        )

        if pd.isna(
            corr_value
        ):
            continue

        if (
            abs(corr_value)
            > CORR_THRESHOLD
        ):

            dropped_redundant.append(
                feature_b
            )

            print(
                f"DROP {feature_b} "
                f"(corr with "
                f"{feature_a} = "
                f"{corr_value:.4f})"
            )


final_features = [
    f
    for f in selected_features
    if f not in dropped_redundant
]


# ============================================================
# SAVE SELECTED FEATURES
# ============================================================

selected_df = icir_df[
    icir_df["feature"].isin(
        final_features
    )
].copy()

selected_df["selected"] = True

selected_df = (
    selected_df
    .sort_values(
        "abs_icir",
        ascending=False
    )
)

selected_df.to_csv(
    SELECTED_OUTPUT,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "CATEGORY 7 COMPLETE"
)

print(
    "=" * 70
)

print(
    "\nSelected features:"
)

for i, feature in enumerate(
    final_features,
    start=1
):

    row = icir_df[
        icir_df["feature"]
        == feature
    ].iloc[0]

    print(
        f"{i:2d}. "
        f"{feature:35s} "
        f"RankIC="
        f"{row['rank_ic']:.6f} "
        f"ICIR="
        f"{row['icir']:.6f}"
    )


print(
    f"\nTotal selected: "
    f"{len(final_features)}"
)

print(
    "\nOutput files:"
)

print(
    f"1. {FEATURE_OUTPUT}"
)

print(
    f"2. {ICIR_OUTPUT}"
)

print(
    f"3. {CORR_OUTPUT}"
)

print(
    f"4. {SELECTED_OUTPUT}"
)

print(
    "\nDone."
)

gc.collect()