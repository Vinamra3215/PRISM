import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# CATEGORY 5 — PHASE 1 — P2
# RankIC and ICIR Calculation
# ============================================================

BASE_DIR = Path("/home/soq/__shutupandbendover/yatin/final features")

P1_FILE = (
    BASE_DIR
    / "category_5"
    / "phase1_final"
    / "category5_all_companies_features.parquet"
)

RAW_DIR = BASE_DIR / "NIFTY50_OHLCV"

OUTPUT_DIR = (
    BASE_DIR
    / "category_5"
    / "p2_rankic_icir"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# P2 SELECTION PERIOD
# ============================================================

START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2023-12-31")


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
    "bollinger_band_percent_b",
    "bollinger_band_width",
    "keltner_channel_position",
    "donchian_channel_position",
]


# ============================================================
# SPECIAL RAW FILENAMES
# ============================================================

SPECIAL_COMPANY_MAPPING = {
    "INDIGO_2022-01-01_to_2026-09-04": "INDIGO",
    "INDIGO_2022-01-01_to-2026-09-04_OHLCV": "INDIGO",
    "JIOFIN_2022_2026": "JIOFIN",
    "JIOFIN_2022_2026_OHLCV": "JIOFIN",
}


# ============================================================
# GET CANONICAL COMPANY NAME
# ============================================================

def get_canonical_company(file_path):

    stem = file_path.stem

    # Special filenames first
    if stem in SPECIAL_COMPANY_MAPPING:
        return SPECIAL_COMPANY_MAPPING[stem]

    # Normal files:
    # ADANIENT_OHLCV.parquet
    # stem = ADANIENT_OHLCV
    # remove complete "_OHLCV" suffix
    if stem.endswith("_OHLCV"):
        return stem[:-len("_OHLCV")]

    return stem


# ============================================================
# LOAD P1 FEATURES
# ============================================================

print("=" * 80)
print("CATEGORY 5 — P2 RANKIC / ICIR")
print("=" * 80)

print("\nLoading P1 feature file...")

p1 = pd.read_parquet(P1_FILE)

p1["date"] = pd.to_datetime(
    p1["date"]
)

p1 = p1[
    (p1["date"] >= START_DATE)
    & (p1["date"] <= END_DATE)
].copy()

required_columns = [
    "company",
    "date",
] + FEATURES

missing_columns = [
    col
    for col in required_columns
    if col not in p1.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required P1 columns: {missing_columns}"
    )

p1_companies = sorted(
    p1["company"]
    .dropna()
    .unique()
)

print(
    f"P1 rows in P2 period: {len(p1)}"
)

print(
    f"P1 companies: {len(p1_companies)}"
)

print(
    f"P2 date range: "
    f"{p1['date'].min().date()} "
    f"to "
    f"{p1['date'].max().date()}"
)

if len(p1_companies) != 50:
    raise ValueError(
        f"Expected 50 P1 companies, "
        f"found {len(p1_companies)}"
    )


# ============================================================
# LOAD RAW FILES
# ============================================================

raw_files = sorted(
    RAW_DIR.glob("*.parquet")
)

print("\n" + "-" * 80)
print("RAW COMPANY MAPPING")
print("-" * 80)

if len(raw_files) != 50:
    raise ValueError(
        f"Expected exactly 50 active raw parquet files, "
        f"found {len(raw_files)}"
    )

raw_company_map = {}

for file_path in raw_files:

    canonical_name = get_canonical_company(
        file_path
    )

    raw_company_map[
        file_path.name
    ] = canonical_name

    print(
        f"{file_path.name}"
    )

    print(
        f"    -> {canonical_name}"
    )


# ============================================================
# VERIFY COMPANY MAPPING
# ============================================================

raw_canonical_companies = sorted(
    set(raw_company_map.values())
)

p1_set = set(
    p1_companies
)

raw_set = set(
    raw_canonical_companies
)

missing_from_raw = sorted(
    p1_set - raw_set
)

extra_in_raw = sorted(
    raw_set - p1_set
)

matched = sorted(
    p1_set & raw_set
)

print("\n" + "-" * 80)
print("COMPANY MAPPING VALIDATION")
print("-" * 80)

print(
    f"P1 companies:              {len(p1_set)}"
)

print(
    f"Raw canonical companies:   {len(raw_set)}"
)

print(
    f"Matched companies:         {len(matched)}"
)

print(
    f"Missing companies:         {len(missing_from_raw)}"
)

print(
    f"Extra companies:           {len(extra_in_raw)}"
)

if missing_from_raw:

    print(
        "\n❌ Missing from raw:"
    )

    for company in missing_from_raw:
        print(
            "   ",
            company
        )


if extra_in_raw:

    print(
        "\n❌ Extra raw companies:"
    )

    for company in extra_in_raw:
        print(
            "   ",
            company
        )


if missing_from_raw or extra_in_raw:

    raise ValueError(
        "Company mapping validation FAILED. "
        "P2 will not continue."
    )


print(
    "\n✅ ALL 50 COMPANIES MATCH CORRECTLY."
)


# ============================================================
# LOAD CLOSE PRICES
# ============================================================

close_parts = []

print("\n" + "-" * 80)
print("LOADING RAW CLOSE DATA")
print("-" * 80)

for i, file_path in enumerate(
    raw_files,
    start=1
):

    company = get_canonical_company(
        file_path
    )

    df = pd.read_parquet(
        file_path
    )

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    # ----------------------------------------
    # Find Date column
    # ----------------------------------------

    if "Date" in df.columns:
        date_col = "Date"

    elif "date" in df.columns:
        date_col = "date"

    else:
        raise ValueError(
            f"No Date column found in "
            f"{file_path.name}"
        )

    # ----------------------------------------
    # Find Close column
    # ----------------------------------------

    if "Close" not in df.columns:

        raise ValueError(
            f"No Close column found in "
            f"{file_path.name}"
        )

    df[date_col] = pd.to_datetime(
        df[date_col]
    )

    df = df[
        [
            date_col,
            "Close",
        ]
    ].copy()

    df = df.rename(
        columns={
            date_col: "date",
            "Close": "close",
        }
    )

    df = df.sort_values(
        "date"
    )

    # ----------------------------------------
    # Duplicate date check
    # ----------------------------------------

    if df["date"].duplicated().any():

        raise ValueError(
            f"Duplicate dates found in "
            f"raw file: {file_path.name}"
        )

    # ----------------------------------------
    # Next trading day return
    # ----------------------------------------

    df["next_day_return"] = (
        df["close"].shift(-1)
        / df["close"]
        - 1.0
    )

    df["company"] = company

    df = df[
        [
            "company",
            "date",
            "close",
            "next_day_return",
        ]
    ]

    # Keep P2 period plus enough future data
    # to calculate Dec 2023 next-day returns.
    df = df[
        (df["date"] >= START_DATE)
        & (
            df["date"]
            <= pd.Timestamp("2024-01-05")
        )
    ].copy()

    close_parts.append(
        df
    )

    print(
        f"[{i:02d}/50] "
        f"{company:<15} "
        f"rows={len(df)}"
    )


close_df = pd.concat(
    close_parts,
    ignore_index=True
)


# ============================================================
# VERIFY RAW COMPANY COVERAGE
# ============================================================

close_companies = sorted(
    close_df["company"].unique()
)

if set(close_companies) != p1_set:

    raise ValueError(
        "After loading raw data, "
        "company sets do not match P1."
    )

print(
    "\n✅ Raw close data contains all 50 companies."
)


# ============================================================
# MERGE FEATURES + NEXT-DAY RETURNS
# ============================================================

print("\n" + "-" * 80)
print(
    "MERGING FEATURES WITH NEXT-DAY RETURNS"
)
print("-" * 80)

merged = p1.merge(
    close_df[
        [
            "company",
            "date",
            "next_day_return",
        ]
    ],
    on=[
        "company",
        "date",
    ],
    how="left",
    validate="one_to_one",
)

print(
    f"Merged rows: {len(merged)}"
)

if len(merged) != len(p1):

    raise ValueError(
        "Merge changed the number of P1 rows."
    )


# ============================================================
# VERIFY RETURN COVERAGE
# ============================================================

coverage = (
    merged
    .groupby("company")["next_day_return"]
    .apply(
        lambda x: x.notna().sum()
    )
    .sort_index()
)

print(
    "\nNext-day return coverage by company:"
)

print(
    coverage.to_string()
)

missing_return_companies = sorted(
    coverage[
        coverage == 0
    ].index.tolist()
)

if missing_return_companies:

    raise ValueError(
        "These companies have ZERO "
        "next-day returns after merge: "
        f"{missing_return_companies}"
    )


print(
    f"\nCompanies with next-day return data: "
    f"{len(coverage[coverage > 0])}/50"
)


# ============================================================
# REMOVE ROWS WITHOUT TARGET
# ============================================================

before_target_filter = len(
    merged
)

merged = merged[
    merged["next_day_return"].notna()
].copy()

after_target_filter = len(
    merged
)

print(
    f"\nRows before target filter: "
    f"{before_target_filter}"
)

print(
    f"Rows with valid next-day return: "
    f"{after_target_filter}"
)

print(
    f"Rows removed because target unavailable: "
    f"{before_target_filter - after_target_filter}"
)


# ============================================================
# CALCULATE DAILY RANKIC
# ============================================================

print("\n" + "-" * 80)
print(
    "CALCULATING DAILY SPEARMAN RANKIC"
)
print("-" * 80)

daily_results = []

unique_dates = sorted(
    merged["date"]
    .dropna()
    .unique()
)

print(
    f"Trading dates available: "
    f"{len(unique_dates)}"
)

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning
)

warnings.filterwarnings(
    "ignore",
    message="ConstantInputWarning*"
)


for feature in FEATURES:

    print(
        f"\nFeature: {feature}"
    )

    feature_results = []

    for date in unique_dates:

        day = merged[
            merged["date"] == date
        ][
            [
                "company",
                feature,
                "next_day_return",
            ]
        ].copy()

        day = day.dropna(
            subset=[
                feature,
                "next_day_return",
            ]
        )

        n_stocks = len(day)

        # Need at least 3 stocks
        if n_stocks < 3:

            ic = np.nan

        else:

            x = day[
                feature
            ].to_numpy(
                dtype=float
            )

            y = day[
                "next_day_return"
            ].to_numpy(
                dtype=float
            )

            # Constant values produce undefined
            # Spearman correlation.
            if (
                np.all(x == x[0])
                or np.all(y == y[0])
            ):

                ic = np.nan

            else:

                result = spearmanr(
                    x,
                    y,
                    nan_policy="omit",
                )

                ic = result.statistic

                if not np.isfinite(ic):
                    ic = np.nan

        feature_results.append(
            {
                "date": date,
                "feature": feature,
                "daily_rankic": ic,
                "n_stocks": n_stocks,
            }
        )

    daily_results.extend(
        feature_results
    )

    valid_count = sum(
        pd.notna(
            x["daily_rankic"]
        )
        for x in feature_results
    )

    print(
        f"  Valid daily ICs: "
        f"{valid_count}/"
        f"{len(feature_results)}"
    )


daily_rankic = pd.DataFrame(
    daily_results
)

daily_rankic["date"] = pd.to_datetime(
    daily_rankic["date"]
)


# ============================================================
# SAVE DAILY RANKIC
# ============================================================

daily_output = (
    OUTPUT_DIR
    / "category5_p2_daily_rankic.csv"
)

daily_rankic.to_csv(
    daily_output,
    index=False
)

print(
    f"\nSaved daily RankIC:"
)

print(
    daily_output
)


# ============================================================
# CALCULATE SUMMARY
# ============================================================

summary_rows = []

for feature in FEATURES:

    values = (
        daily_rankic.loc[
            daily_rankic["feature"] == feature,
            "daily_rankic",
        ]
        .dropna()
        .astype(float)
    )

    if len(values) == 0:

        rankic = np.nan
        std_ic = np.nan
        icir = np.nan

    else:

        rankic = values.mean()

        std_ic = values.std(
            ddof=1
        )

        if (
            pd.isna(std_ic)
            or std_ic == 0
        ):

            icir = np.nan

        else:

            icir = (
                rankic
                / std_ic
            )

    abs_icir = (
        abs(icir)
        if pd.notna(icir)
        else np.nan
    )

    # ----------------------------------------
    # P2 thresholds
    # ----------------------------------------

    if pd.isna(icir):

        verdict = "DISCARD"

    elif abs_icir > 0.4:

        verdict = "STRONG KEEP"

    elif abs_icir >= 0.2:

        verdict = "DECENT KEEP"

    elif abs_icir >= 0.1:

        verdict = "WEAK KEEP IF NEEDED"

    else:

        verdict = "DISCARD"

    summary_rows.append(
        {
            "feature": feature,
            "RankIC": rankic,
            "ICIR": icir,
            "abs_ICIR": abs_icir,
            "valid_daily_IC_count": len(values),
            "verdict": verdict,
        }
    )


summary = pd.DataFrame(
    summary_rows
)

summary = summary.sort_values(
    by="abs_ICIR",
    ascending=False,
    na_position="last",
).reset_index(
    drop=True
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_output = (
    OUTPUT_DIR
    / "category5_p2_summary.csv"
)

summary.to_csv(
    summary_output,
    index=False
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "category":
        "Category 5 — Candlestick & Price Structure",

    "phase":
        "Phase 1",

    "step":
        "P2 — RankIC and ICIR",

    "feature_selection_period": {
        "start": "2022-01-01",
        "end": "2023-12-31",
    },

    "number_of_companies": 50,

    "number_of_features":
        len(FEATURES),

    "features":
        FEATURES,

    "rankic_method":
        "Daily cross-sectional Spearman correlation "
        "between feature values and next-trading-day "
        "actual returns.",

    "rankic_definition":
        "Mean of daily RankIC values.",

    "icir_definition":
        "Mean daily RankIC divided by sample standard "
        "deviation of daily RankIC.",

    "thresholds": {

        "above_0.4":
            "STRONG KEEP",

        "0.2_to_0.4":
            "DECENT KEEP",

        "0.1_to_0.2":
            "WEAK KEEP IF NEEDED",

        "below_0.1":
            "DISCARD",
    },

    "company_mapping":
        raw_company_map,

    "special_company_mapping":
        SPECIAL_COMPANY_MAPPING,

    "uses_model_training":
        False,

    "uses_lightgbm":
        False,

    "uses_phase_2":
        False,

    "uses_2024_or_later_for_selection":
        False,

    "continuous_learning":
        False,
}


metadata_output = (
    OUTPUT_DIR
    / "category5_p2_metadata.json"
)

with open(
    metadata_output,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        default=str,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("CATEGORY 5 — P2 SUMMARY")
print("=" * 80)

print(
    summary[
        [
            "feature",
            "RankIC",
            "ICIR",
            "abs_ICIR",
            "verdict",
        ]
    ].to_string(
        index=False
    )
)

print("\n" + "-" * 80)

print(
    "Daily RankIC file:"
)

print(
    daily_output
)

print(
    "\nP2 summary file:"
)

print(
    summary_output
)

print(
    "\nMetadata file:"
)

print(
    metadata_output
)

print("\n" + "=" * 80)
print("P2 COMPLETED SUCCESSFULLY")
print("=" * 80)
print("Company mapping: 50/50 PASS")
print("Feature selection period: Jan 2022 – Dec 2023")
print("Model training: NO")
print("LightGBM: NO")
print("Phase 2: NO")
print("Continuous learning: NO")
print("=" * 80)
