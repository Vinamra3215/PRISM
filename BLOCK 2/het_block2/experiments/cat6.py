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
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "p1_cat6")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

SELECTION_START = "2022-01-01"
SELECTION_END = "2023-12-31"

CORR_THRESHOLD = 0.70
TARGET_FEATURES = 7


# ============================================================
# FILES
# ============================================================

STOCK_FILE = os.path.join(
    DATA_DIR,
    "nifty50_stocks_ohlcv.parquet"
)

MARKET_FILES = {
    "nifty50": [
        "nifty50.parquet",
        "nifty_50.parquet",
        "nifty50_index.parquet",
        "nifty_index.parquet",
    ],

    "vix": [
        "india_vix.parquet",
        "nifty_vix.parquet",
        "vix.parquet",
    ],

    "bank": [
        "nifty_bank.parquet",
        "niftybank.parquet",
    ],

    "it": [
        "nifty_it.parquet",
    ],

    "pharma": [
        "nifty_pharma.parquet",
    ],

    "auto": [
        "nifty_auto.parquet",
    ],

    "metal": [
        "nifty_metal.parquet",
    ],

    "energy": [
        "nifty_energy.parquet",
    ],

    "usdinr": [
        "usdinr.parquet",
        "usd_inr.parquet",
        "usd_inr_x.parquet",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def find_existing_file(candidates):
    for filename in candidates:
        path = os.path.join(DATA_DIR, filename)

        if os.path.exists(path):
            return path

    return None


def load_market_file(name):
    candidates = MARKET_FILES[name]
    path = find_existing_file(candidates)

    if path is None:
        raise FileNotFoundError(
            f"\nMissing market data for '{name}'.\n"
            f"Expected one of:\n"
            + "\n".join(f"  {x}" for x in candidates)
            + "\n"
        )

    df = pd.read_parquet(path)

    if df.empty:
        raise ValueError(
            f"\nMarket file '{path}' exists but contains no data."
        )

    print(f"Loaded {name:8s}: {os.path.basename(path):30s} {df.shape}")

    return df


def standardize_market_dataframe(df):
    df = df.copy()

    # --------------------------------------------------------
    # Date
    # --------------------------------------------------------

    date_col = None

    for col in ["date", "Date", "datetime", "Datetime", "timestamp"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        raise ValueError(
            "Could not find a date column in market dataset."
        )

    df["date"] = pd.to_datetime(df[date_col])

    # --------------------------------------------------------
    # Column names
    # --------------------------------------------------------

    rename_map = {}

    for col in df.columns:
        lower = str(col).lower()

        if lower in ["adj_close", "adj close", "adjusted_close"]:
            rename_map[col] = "adj_close"

        elif lower == "close":
            rename_map[col] = "close"

        elif lower == "open":
            rename_map[col] = "open"

        elif lower == "high":
            rename_map[col] = "high"

        elif lower == "low":
            rename_map[col] = "low"

        elif lower == "volume":
            rename_map[col] = "volume"

    df = df.rename(columns=rename_map)

    # Prefer adjusted close when available.
    if "adj_close" in df.columns:
        df["price"] = df["adj_close"]
    elif "close" in df.columns:
        df["price"] = df["close"]
    else:
        raise ValueError(
            "Market dataset does not contain close or adjusted close."
        )

    df = (
        df[["date", "price"]]
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


def daily_return(series, periods=1):
    return series.pct_change(periods)


def spearman_ic(feature, target):
    temp = pd.DataFrame({
        "feature": feature,
        "target": target
    }).dropna()

    if len(temp) < 3:
        return np.nan

    if temp["feature"].nunique() < 2:
        return np.nan

    if temp["target"].nunique() < 2:
        return np.nan

    return temp["feature"].corr(
        temp["target"],
        method="spearman"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY 6: MARKET-LEVEL FEATURES")
print("=" * 70)

print("\nLoading market datasets...\n")

nifty50 = standardize_market_dataframe(
    load_market_file("nifty50")
)

vix = standardize_market_dataframe(
    load_market_file("vix")
)

bank = standardize_market_dataframe(
    load_market_file("bank")
)

it = standardize_market_dataframe(
    load_market_file("it")
)

pharma = standardize_market_dataframe(
    load_market_file("pharma")
)

auto = standardize_market_dataframe(
    load_market_file("auto")
)

metal = standardize_market_dataframe(
    load_market_file("metal")
)

energy = standardize_market_dataframe(
    load_market_file("energy")
)

usdinr = standardize_market_dataframe(
    load_market_file("usdinr")
)


# ============================================================
# BUILD DAILY MARKET TABLE
# ============================================================

print("\nBuilding market-level feature table...")

market = pd.DataFrame({
    "date": nifty50["date"]
})

market = market.sort_values("date").reset_index(drop=True)


def merge_price_data(base, df, column_name):
    temp = df[["date", "price"]].copy()
    temp = temp.rename(columns={"price": column_name})

    return base.merge(
        temp,
        on="date",
        how="left"
    )


market = merge_price_data(
    market,
    nifty50,
    "nifty50_price"
)

market = merge_price_data(
    market,
    vix,
    "vix_level"
)

market = merge_price_data(
    market,
    bank,
    "bank_price"
)

market = merge_price_data(
    market,
    it,
    "it_price"
)

market = merge_price_data(
    market,
    pharma,
    "pharma_price"
)

market = merge_price_data(
    market,
    auto,
    "auto_price"
)

market = merge_price_data(
    market,
    metal,
    "metal_price"
)

market = merge_price_data(
    market,
    energy,
    "energy_price"
)

market = merge_price_data(
    market,
    usdinr,
    "usdinr_price"
)


market = market.sort_values("date").reset_index(drop=True)


# ============================================================
# CATEGORY 6 FEATURES
# ============================================================

# ------------------------------------------------------------
# 1. NIFTY 50 Daily Return
# ------------------------------------------------------------

market["nifty50_daily_return"] = (
    market["nifty50_price"].pct_change()
)


# ------------------------------------------------------------
# 2. NIFTY 50 5-day Return
# ------------------------------------------------------------

market["nifty50_return_5d"] = (
    market["nifty50_price"].pct_change(5)
)


# ------------------------------------------------------------
# 3. NIFTY 50 Rolling Vol 20-day
# ------------------------------------------------------------

nifty_daily_return = market["nifty50_price"].pct_change()

market["nifty50_rolling_vol_20d"] = (
    nifty_daily_return
    .rolling(20)
    .std()
)


# ------------------------------------------------------------
# 4. India VIX Level
# ------------------------------------------------------------

market["india_vix_level"] = market["vix_level"]


# ------------------------------------------------------------
# 5. India VIX Change
# ------------------------------------------------------------

market["india_vix_change"] = (
    market["vix_level"].diff()
)


# ------------------------------------------------------------
# 6. NIFTY Bank Return
# ------------------------------------------------------------

market["nifty_bank_return"] = (
    market["bank_price"].pct_change()
)


# ------------------------------------------------------------
# 7. NIFTY IT Return
# ------------------------------------------------------------

market["nifty_it_return"] = (
    market["it_price"].pct_change()
)


# ------------------------------------------------------------
# 8. NIFTY Pharma Return
# ------------------------------------------------------------

market["nifty_pharma_return"] = (
    market["pharma_price"].pct_change()
)


# ------------------------------------------------------------
# 9. NIFTY Auto Return
# ------------------------------------------------------------

market["nifty_auto_return"] = (
    market["auto_price"].pct_change()
)


# ------------------------------------------------------------
# 10. NIFTY Metal Return
# ------------------------------------------------------------

market["nifty_metal_return"] = (
    market["metal_price"].pct_change()
)


# ------------------------------------------------------------
# 11. NIFTY Energy Return
# ------------------------------------------------------------

market["nifty_energy_return"] = (
    market["energy_price"].pct_change()
)


# ------------------------------------------------------------
# 12. USD/INR Daily Return
# ------------------------------------------------------------

market["usdinr_daily_return"] = (
    market["usdinr_price"].pct_change()
)


FEATURES = [
    "nifty50_daily_return",
    "nifty50_return_5d",
    "nifty50_rolling_vol_20d",
    "india_vix_level",
    "india_vix_change",
    "nifty_bank_return",
    "nifty_it_return",
    "nifty_pharma_return",
    "nifty_auto_return",
    "nifty_metal_return",
    "nifty_energy_return",
    "usdinr_daily_return",
]


# ============================================================
# TARGET FOR MARKET-LEVEL SELECTION
# ============================================================

# Since a market-level feature has exactly one value per day,
# cross-sectional stock RankIC is not mathematically defined.
#
# Therefore, use the next-day NIFTY 50 return as the time-series
# target for evaluating predictive association.

market["next_day_nifty50_return"] = (
    market["nifty50_price"].shift(-1)
    / market["nifty50_price"]
    - 1
)


# ============================================================
# SAVE FULL DAILY MARKET FEATURES
# ============================================================

market_daily = market[
    ["date"] + FEATURES
].copy()

market_daily.to_parquet(
    os.path.join(
        OUTPUT_DIR,
        "market_daily_features.parquet"
    ),
    index=False
)


# ============================================================
# SELECTION PERIOD
# ============================================================

selection = market[
    (market["date"] >= SELECTION_START)
    & (market["date"] <= SELECTION_END)
].copy()


# ============================================================
# TIME-SERIES IC / ICIR
# ============================================================

print("\nCalculating market-feature IC metrics...")

ic_results = []

for feature in FEATURES:

    daily_ic = spearman_ic(
        selection[feature],
        selection["next_day_nifty50_return"]
    )

    ic_results.append({
        "feature": feature,
        "RankIC": daily_ic,
        "ICIR": np.nan,
        "observations": selection[
            [feature, "next_day_nifty50_return"]
        ].dropna().shape[0]
    })


ic_metrics = pd.DataFrame(ic_results)


# ------------------------------------------------------------
# For a time-series market feature there is one IC over the
# complete selection period, rather than one cross-sectional
# IC per day.
#
# Therefore ICIR in the conventional daily-cross-sectional
# sense is not defined here.
# ------------------------------------------------------------

ic_metrics["ICIR"] = np.nan

ic_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "icir_metrics.csv"
    ),
    index=False
)


# ============================================================
# FEATURE CORRELATION
# ============================================================

print("Calculating market-feature Spearman correlation...")

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
# REDUNDANCY SELECTION
# ============================================================

# Rank by absolute time-series RankIC.

ranking = (
    ic_metrics
    .assign(abs_RankIC=lambda x: x["RankIC"].abs())
    .sort_values(
        "abs_RankIC",
        ascending=False
    )
    .reset_index(drop=True)
)


selected = []
dropped = []

for feature in ranking["feature"]:

    keep = True

    for existing in selected:

        corr = spearman_corr.loc[
            feature,
            existing
        ]

        if pd.notna(corr) and abs(corr) > CORR_THRESHOLD:

            keep = False

            dropped.append({
                "feature": feature,
                "correlated_with": existing,
                "correlation": corr,
                "reason": "Higher-ranked correlated market feature"
            })

            break

    if keep:
        selected.append(feature)


# ============================================================
# LIMIT TO TARGET
# ============================================================

selected = selected[:TARGET_FEATURES]


# ============================================================
# OUTPUTS
# ============================================================

selected_df = (
    ic_metrics[
        ic_metrics["feature"].isin(selected)
    ]
    .copy()
)

selected_df["selection_rank"] = (
    selected_df["feature"]
    .map({
        feature: i + 1
        for i, feature in enumerate(selected)
    })
)

selected_df = selected_df.sort_values(
    "selection_rank"
)

selected_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "selected_features.csv"
    ),
    index=False
)


dropped_df = pd.DataFrame(dropped)

dropped_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dropped_correlated.csv"
    ),
    index=False
)


# ============================================================
# BROADCAST MARKET FEATURES TO ALL 50 STOCKS
# ============================================================

print("\nBroadcasting market features to stock-day rows...")

if not os.path.exists(STOCK_FILE):
    raise FileNotFoundError(
        f"Stock dataset not found:\n{STOCK_FILE}"
    )

stocks = pd.read_parquet(STOCK_FILE)

stocks["date"] = pd.to_datetime(stocks["date"])


market_for_broadcast = market[
    ["date"] + FEATURES
].copy()


features_full = stocks[
    ["date", "symbol"]
].merge(
    market_for_broadcast,
    on="date",
    how="left"
)


features_full.to_parquet(
    os.path.join(
        OUTPUT_DIR,
        "features_full.parquet"
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
        "target_selected",
        "selected_count",
        "selection_method",
        "rankic_definition",
        "icir_definition"
    ],
    "value": [
        "Category 6 - Market-Level",
        len(FEATURES),
        SELECTION_START,
        SELECTION_END,
        CORR_THRESHOLD,
        TARGET_FEATURES,
        len(selected),
        "Time-series Spearman association with next-day NIFTY 50 return",
        "Spearman correlation over the Jan 2022-Dec 2023 daily observations",
        "Not defined in the conventional cross-sectional daily RankIC framework"
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

final_deliverable = selected_df.copy()

final_deliverable.insert(
    0,
    "category",
    "Market-Level"
)

final_deliverable = final_deliverable[
    [
        "selection_rank",
        "feature",
        "category",
        "RankIC",
        "ICIR",
        "observations"
    ]
]

final_deliverable.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "final_deliverable.csv"
    ),
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("CATEGORY 6 COMPLETE")
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