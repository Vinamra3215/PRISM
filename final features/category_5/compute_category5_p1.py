import pandas as pd
import numpy as np
from pathlib import Path
import json
import traceback


# ============================================================
# PRISM - CATEGORY 5
# PHASE 1 - STEP P1
# Candlestick & Price Structure Features
#
# IMPORTANT:
# - Compute all 10 Category 5 features
# - All 50 companies
# - Full available data: Jan 2022 -> Present
# - NO continuous learning
# - NO model training
# - NO RankIC/ICIR here
# - NO feature selection here
# - Raw OHLCV files are NEVER modified
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/home/soq/__shutupandbendover/yatin/final features")

DATA_DIR = BASE_DIR / "NIFTY50_OHLCV"

CATEGORY_DIR = BASE_DIR / "category_5"

FEATURE_DIR = CATEGORY_DIR / "feature_computation"

VALIDATION_DIR = CATEGORY_DIR / "validation"

METADATA_DIR = CATEGORY_DIR / "metadata"

FINAL_DIR = CATEGORY_DIR / "phase1_final"


# Create directories
FEATURE_DIR.mkdir(parents=True, exist_ok=True)
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
METADATA_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CATEGORY 5 FEATURES
# ============================================================

FEATURE_COLUMNS = [
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
# PARAMETERS
# ============================================================

# Bollinger Bands
BB_WINDOW = 20
BB_STD_MULTIPLIER = 2.0

# Keltner Channel
# The PRISM document specifies ATR-based Keltner Channel
# but does not specify the exact window/multiplier.
# We use a standard, explicit configuration:
KC_EMA_WINDOW = 20
KC_ATR_WINDOW = 20
KC_ATR_MULTIPLIER = 2.0

# Donchian Channel
DONCHIAN_WINDOW = 20


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_divide(numerator, denominator):
    """
    Element-wise division.

    If denominator is zero, return NaN rather than
    generating infinity.
    """
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")

    result = numerator / denominator.replace(0, np.nan)

    result = result.replace([np.inf, -np.inf], np.nan)

    return result


def extract_company_name(file_path):
    """
    Convert filename into a clean company name.
    """

    name = file_path.stem

    # Examples:
    # ADANIENT_OHLCV
    # JIOFIN_2022_2026
    # INDIGO_2022-01-01_to_2026-09-04

    if name.endswith("_OHLCV"):
        return name[:-6]

    if name.startswith("JIOFIN"):
        return "JIOFIN"

    if name.startswith("INDIGO"):
        return "INDIGO"

    return name


def normalize_columns(df):
    """
    Normalize column names.
    """

    df = df.copy()

    df.columns = [
        str(col).strip().lower().replace("_", " ")
        for col in df.columns
    ]

    # Convert possible naming variations
    rename_map = {
        "adj close": "adj close",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "date": "date",
    }

    df = df.rename(columns=rename_map)

    return df


def load_ohlcv(file_path):
    """
    Load and validate basic OHLCV structure.
    """

    df = pd.read_parquet(file_path)

    df = normalize_columns(df)

    required = [
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    missing_columns = [
        col for col in required
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Keep required raw columns only for computation
    df = df[required].copy()

    # Convert types
    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Sort chronologically
    df = (
        df
        .sort_values("date")
        .drop_duplicates(subset=["date"], keep="first")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# TRUE RANGE
# ============================================================

def calculate_true_range(df):
    """
    True Range:

    TR = max(
        High - Low,
        abs(High - Previous Close),
        abs(Low - Previous Close)
    )
    """

    previous_close = df["close"].shift(1)

    range_1 = df["high"] - df["low"]

    range_2 = (
        df["high"] - previous_close
    ).abs()

    range_3 = (
        df["low"] - previous_close
    ).abs()

    true_range = pd.concat(
        [range_1, range_2, range_3],
        axis=1
    ).max(axis=1)

    return true_range


# ============================================================
# CONSECUTIVE STREAK FUNCTION
# ============================================================

def calculate_positive_streak(condition):
    """
    Count consecutive True values.

    Example:

    True  -> 1
    True  -> 2
    True  -> 3
    False -> 0
    True  -> 1
    """

    condition = condition.fillna(False)

    streak = np.zeros(len(condition), dtype=np.int64)

    current = 0

    for i, value in enumerate(condition.to_numpy()):

        if value:
            current += 1
        else:
            current = 0

        streak[i] = current

    return pd.Series(
        streak,
        index=condition.index,
        dtype="int64"
    )


# ============================================================
# CATEGORY 5 FEATURE COMPUTATION
# ============================================================

def compute_category5_features(df):
    """
    Compute all 10 Category 5 features.
    """

    result = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # BASIC CANDLE VALUES
    # --------------------------------------------------------

    candle_range = df["high"] - df["low"]

    candle_body = (
        df["close"] - df["open"]
    ).abs()

    upper_wick = (
        df["high"]
        - df[["open", "close"]].max(axis=1)
    )

    lower_wick = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    )

    # --------------------------------------------------------
    # 1. UPPER SHADOW RATIO
    #
    # Upper shadow / full candle range
    # --------------------------------------------------------

    result["upper_shadow_ratio"] = safe_divide(
        upper_wick,
        candle_range
    )

    # --------------------------------------------------------
    # 2. LOWER SHADOW RATIO
    #
    # Lower shadow / full candle range
    # --------------------------------------------------------

    result["lower_shadow_ratio"] = safe_divide(
        lower_wick,
        candle_range
    )

    # --------------------------------------------------------
    # 3. BODY RATIO
    #
    # Absolute candle body / full candle range
    # --------------------------------------------------------

    result["body_ratio"] = safe_divide(
        candle_body,
        candle_range
    )

    # --------------------------------------------------------
    # 4. CANDLE DIRECTION
    #
    # +1 = bullish
    # -1 = bearish
    #  0 = open == close
    # --------------------------------------------------------

    result["candle_direction"] = np.select(
        [
            df["close"] > df["open"],
            df["close"] < df["open"],
        ],
        [
            1,
            -1,
        ],
        default=0
    )

    # --------------------------------------------------------
    # DAILY CLOSE RETURN
    #
    # Used specifically for consecutive up/down days
    # because the PRISM document defines these as
    # positive/negative RETURNS.
    # --------------------------------------------------------

    daily_return = (
        df["close"]
        .pct_change()
    )

    # --------------------------------------------------------
    # 5. CONSECUTIVE UP DAYS
    #
    # Number of consecutive days with positive return
    # --------------------------------------------------------

    result["consecutive_up_days"] = (
        calculate_positive_streak(
            daily_return > 0
        )
    )

    # --------------------------------------------------------
    # 6. CONSECUTIVE DOWN DAYS
    #
    # Number of consecutive days with negative return
    # --------------------------------------------------------

    result["consecutive_down_days"] = (
        calculate_positive_streak(
            daily_return < 0
        )
    )

    # --------------------------------------------------------
    # 7. BOLLINGER BAND %B
    #
    # Middle = 20-day SMA
    # Upper  = Middle + 2 * rolling std
    # Lower  = Middle - 2 * rolling std
    #
    # %B = (Close - Lower) / (Upper - Lower)
    # --------------------------------------------------------

    bb_middle = (
        df["close"]
        .rolling(
            window=BB_WINDOW,
            min_periods=BB_WINDOW
        )
        .mean()
    )

    bb_std = (
        df["close"]
        .rolling(
            window=BB_WINDOW,
            min_periods=BB_WINDOW
        )
        .std()
    )

    bb_upper = (
        bb_middle
        + BB_STD_MULTIPLIER * bb_std
    )

    bb_lower = (
        bb_middle
        - BB_STD_MULTIPLIER * bb_std
    )

    result["bollinger_band_percent_b"] = safe_divide(
        df["close"] - bb_lower,
        bb_upper - bb_lower
    )

    # --------------------------------------------------------
    # 8. BOLLINGER BAND WIDTH
    #
    # Width = (Upper - Lower) / Middle
    # --------------------------------------------------------

    result["bollinger_band_width"] = safe_divide(
        bb_upper - bb_lower,
        bb_middle
    )

    # --------------------------------------------------------
    # TRUE RANGE + ATR
    # --------------------------------------------------------

    true_range = calculate_true_range(df)

    atr = (
        true_range
        .rolling(
            window=KC_ATR_WINDOW,
            min_periods=KC_ATR_WINDOW
        )
        .mean()
    )

    # --------------------------------------------------------
    # KELTNER CHANNEL
    #
    # Middle = EMA(20)
    # Upper  = EMA(20) + 2*ATR(20)
    # Lower  = EMA(20) - 2*ATR(20)
    #
    # Position = (Close - Lower) / (Upper - Lower)
    #
    # The PRISM document specifies ATR-based Keltner
    # positioning but does not specify parameters.
    # These parameters are recorded in metadata.
    # --------------------------------------------------------

    kc_middle = (
        df["close"]
        .ewm(
            span=KC_EMA_WINDOW,
            adjust=False,
            min_periods=KC_EMA_WINDOW
        )
        .mean()
    )

    kc_upper = (
        kc_middle
        + KC_ATR_MULTIPLIER * atr
    )

    kc_lower = (
        kc_middle
        - KC_ATR_MULTIPLIER * atr
    )

    result["keltner_channel_position"] = safe_divide(
        df["close"] - kc_lower,
        kc_upper - kc_lower
    )

    # --------------------------------------------------------
    # 10. DONCHIAN CHANNEL POSITION
    #
    # 20-day high-low range
    #
    # Position = (Close - Rolling Low)
    #            / (Rolling High - Rolling Low)
    # --------------------------------------------------------

    donchian_high = (
        df["high"]
        .rolling(
            window=DONCHIAN_WINDOW,
            min_periods=DONCHIAN_WINDOW
        )
        .max()
    )

    donchian_low = (
        df["low"]
        .rolling(
            window=DONCHIAN_WINDOW,
            min_periods=DONCHIAN_WINDOW
        )
        .min()
    )

    result["donchian_channel_position"] = safe_divide(
        df["close"] - donchian_low,
        donchian_high - donchian_low
    )

    # --------------------------------------------------------
    # FINAL CLEANUP
    # --------------------------------------------------------

    result = result.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return result


# ============================================================
# PROCESS ONE COMPANY
# ============================================================

def process_company(file_path):

    company = extract_company_name(file_path)

    print("\n" + "-" * 90)
    print(f"Processing: {company}")
    print(f"File     : {file_path.name}")

    df = load_ohlcv(file_path)

    feature_df = compute_category5_features(df)

    output = pd.concat(
        [
            df[["date"]],
            feature_df
        ],
        axis=1
    )

    output["company"] = company

    # Put company first
    output = output[
        ["company", "date"] + FEATURE_COLUMNS
    ]

    output_file = (
        FEATURE_DIR
        / f"{company}_category5_features.parquet"
    )

    output.to_parquet(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # Validation information
    # --------------------------------------------------------

    feature_nan_counts = {
        feature: int(output[feature].isna().sum())
        for feature in FEATURE_COLUMNS
    }

    feature_inf_counts = {}

    for feature in FEATURE_COLUMNS:
        feature_inf_counts[feature] = int(
            np.isinf(
                output[feature].to_numpy(
                    dtype=float,
                    na_value=np.nan
                )
            ).sum()
        )

    row = {
        "company": company,
        "source_file": file_path.name,
        "output_file": output_file.name,
        "rows": len(output),
        "start_date": output["date"].min(),
        "end_date": output["date"].max(),
        "feature_nan_total": int(
            output[FEATURE_COLUMNS].isna().sum().sum()
        ),
        "feature_inf_total": int(
            sum(feature_inf_counts.values())
        ),
    }

    for feature in FEATURE_COLUMNS:
        row[f"{feature}_nan"] = feature_nan_counts[feature]

    print(
        f"Rows      : {len(output)}"
    )

    print(
        f"Date range: "
        f"{output['date'].min().date()} -> "
        f"{output['date'].max().date()}"
    )

    print(
        f"Output    : {output_file}"
    )

    print(
        f"Total NaN : "
        f"{row['feature_nan_total']}"
    )

    return output, row


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("PRISM - CATEGORY 5 - PHASE 1 - P1")
    print("CANDLESTICK & PRICE STRUCTURE FEATURES")
    print("=" * 100)

    print()
    print(f"Raw data directory:")
    print(f"  {DATA_DIR}")

    print()
    print(f"Feature output directory:")
    print(f"  {FEATURE_DIR}")

    print()
    print("Feature computation period:")
    print("  Full available data: Jan 2022 -> Present")

    print()
    print("RankIC / ICIR:")
    print("  NOT computed in P1")

    print()
    print("Continuous learning:")
    print("  NOT USED")

    # --------------------------------------------------------
    # Find raw parquet files
    # --------------------------------------------------------

    files = sorted(
        DATA_DIR.glob("*.parquet")
    )

    print()
    print(f"Parquet files found: {len(files)}")

    if len(files) != 50:
        raise RuntimeError(
            f"Expected exactly 50 Parquet files, "
            f"but found {len(files)}."
        )

    # --------------------------------------------------------
    # Process all companies
    # --------------------------------------------------------

    all_feature_frames = []

    validation_rows = []

    successful = 0

    failed = []

    for file_path in files:

        try:

            feature_df, validation_row = (
                process_company(file_path)
            )

            all_feature_frames.append(
                feature_df
            )

            validation_rows.append(
                validation_row
            )

            successful += 1

        except Exception as e:

            print()
            print("ERROR")
            print(f"Company file: {file_path.name}")
            print(f"Reason: {e}")

            traceback.print_exc()

            failed.append(
                {
                    "file": file_path.name,
                    "error": str(e),
                }
            )

    # --------------------------------------------------------
    # Stop if any company failed
    # --------------------------------------------------------

    if failed:

        error_file = (
            VALIDATION_DIR
            / "p1_failed_companies.json"
        )

        with open(
            error_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                failed,
                f,
                indent=4
            )

        raise RuntimeError(
            f"{len(failed)} companies failed. "
            f"Check {error_file}"
        )

    # --------------------------------------------------------
    # Combine all companies
    # --------------------------------------------------------

    combined = pd.concat(
        all_feature_frames,
        ignore_index=True
    )

    combined = (
        combined
        .sort_values(
            ["date", "company"]
        )
        .reset_index(drop=True)
    )

    combined_file = (
        FINAL_DIR
        / "category5_all_companies_features.parquet"
    )

    combined.to_parquet(
        combined_file,
        index=False
    )

    # --------------------------------------------------------
    # Validation dataframe
    # --------------------------------------------------------

    validation_df = pd.DataFrame(
        validation_rows
    )

    validation_df = (
        validation_df
        .sort_values("company")
        .reset_index(drop=True)
    )

    validation_file = (
        VALIDATION_DIR
        / "p1_company_validation.csv"
    )

    validation_df.to_csv(
        validation_file,
        index=False
    )

    # --------------------------------------------------------
    # Global feature NaN report
    # --------------------------------------------------------

    global_nan_report = []

    for feature in FEATURE_COLUMNS:

        nan_count = int(
            combined[feature].isna().sum()
        )

        total_count = len(combined)

        valid_count = (
            total_count - nan_count
        )

        global_nan_report.append(
            {
                "feature": feature,
                "total_rows": total_count,
                "valid_values": valid_count,
                "nan_values": nan_count,
                "nan_percentage": (
                    100.0 * nan_count / total_count
                    if total_count > 0
                    else np.nan
                ),
            }
        )

    global_nan_df = pd.DataFrame(
        global_nan_report
    )

    global_nan_file = (
        VALIDATION_DIR
        / "p1_feature_nan_report.csv"
    )

    global_nan_df.to_csv(
        global_nan_file,
        index=False
    )

    # --------------------------------------------------------
    # Company summary
    # --------------------------------------------------------

    company_summary = (
        combined
        .groupby("company")
        .agg(
            rows=("date", "count"),
            start_date=("date", "min"),
            end_date=("date", "max"),
        )
        .reset_index()
        .sort_values("company")
    )

    company_summary_file = (
        VALIDATION_DIR
        / "p1_company_summary.csv"
    )

    company_summary.to_csv(
        company_summary_file,
        index=False
    )

    # --------------------------------------------------------
    # Feature definitions
    # --------------------------------------------------------

    feature_definitions = pd.DataFrame(
        [
            {
                "feature": "upper_shadow_ratio",
                "definition": (
                    "(High - max(Open, Close)) / "
                    "(High - Low)"
                ),
                "category": "Candlestick & Price Structure",
            },
            {
                "feature": "lower_shadow_ratio",
                "definition": (
                    "(min(Open, Close) - Low) / "
                    "(High - Low)"
                ),
                "category": "Candlestick & Price Structure",
            },
            {
                "feature": "body_ratio",
                "definition": (
                    "abs(Close - Open) / "
                    "(High - Low)"
                ),
                "category": "Candlestick & Price Structure",
            },
            {
                "feature": "candle_direction",
                "definition": (
                    "+1 bullish, -1 bearish, "
                    "0 when Open equals Close"
                ),
                "category": "Candlestick & Price Structure",
            },
            {
                "feature": "consecutive_up_days",
                "definition": (
                    "Consecutive days with positive "
                    "close-to-close return"
                ),
                "category": "Candlestick & Price Structure",
            },
            {
                "feature": "consecutive_down_days",
                "definition": (
                    "Consecutive days with negative "
                    "close-to-close return"
                ),
                "category": "Candlestick & Price Structure",
            },
            {
                "feature": "bollinger_band_percent_b",
                "definition": (
                    "(Close - Lower Band) / "
                    "(Upper Band - Lower Band); "
                    "20-day SMA and 2 standard deviations"
                ),
                "category": "Candlestick & Price Structure",
            },
            {
                "feature": "bollinger_band_width",
                "definition": (
                    "(Upper Band - Lower Band) / "
                    "Middle Band; 20-day SMA and "
                    "2 standard deviations"
                ),
                "category": "Candlestick & Price Structure",
            },
            {
                "feature": "keltner_channel_position",
                "definition": (
                    "(Close - Lower KC) / "
                    "(Upper KC - Lower KC); "
                    "EMA20 +/- 2*ATR20"
                ),
                "category": "Candlestick & Price Structure",
            },
            {
                "feature": "donchian_channel_position",
                "definition": (
                    "(Close - 20-day Low) / "
                    "(20-day High - 20-day Low)"
                ),
                "category": "Candlestick & Price Structure",
            },
        ]
    )

    definitions_file = (
        METADATA_DIR
        / "category5_feature_definitions.csv"
    )

    feature_definitions.to_csv(
        definitions_file,
        index=False
    )

    # --------------------------------------------------------
    # Parameter metadata
    # --------------------------------------------------------

    metadata = {
        "project": "PRISM",
        "block": 2,
        "category": 5,
        "category_name": (
            "Candlestick & Price Structure Features"
        ),
        "phase": 1,
        "step": "P1",
        "continuous_learning": False,
        "model_training": False,
        "feature_selection": False,
        "rankic_calculation": False,
        "icir_calculation": False,
        "data_period": (
            "Full available data: January 2022 -> Present"
        ),
        "raw_data_directory": str(DATA_DIR),
        "number_of_companies": len(files),
        "number_of_features": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "bollinger": {
            "window": BB_WINDOW,
            "std_multiplier": BB_STD_MULTIPLIER,
        },
        "keltner": {
            "ema_window": KC_EMA_WINDOW,
            "atr_window": KC_ATR_WINDOW,
            "atr_multiplier": KC_ATR_MULTIPLIER,
        },
        "donchian": {
            "window": DONCHIAN_WINDOW,
        },
        "notes": [
            (
                "Expected rolling-window NaNs are retained."
            ),
            (
                "No artificial values are created for "
                "missing historical periods."
            ),
            (
                "JIOFIN uses only its actual available "
                "history."
            ),
            (
                "Raw OHLCV files are not modified."
            ),
            (
                "2024-present data is computed in P1 but "
                "must not be used for P2/P3 feature selection."
            ),
        ],
    }

    metadata_file = (
        METADATA_DIR
        / "category5_p1_metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("P1 COMPLETE")
    print("=" * 100)

    print()
    print(f"Companies processed : {successful}/{len(files)}")
    print(f"Features per company: {len(FEATURE_COLUMNS)}")
    print(f"Total combined rows : {len(combined)}")

    print()
    print(
        f"Combined feature file:\n"
        f"  {combined_file}"
    )

    print()
    print(
        f"Company validation:\n"
        f"  {validation_file}"
    )

    print()
    print(
        f"Feature NaN report:\n"
        f"  {global_nan_file}"
    )

    print()
    print(
        f"Company summary:\n"
        f"  {company_summary_file}"
    )

    print()
    print(
        f"Feature definitions:\n"
        f"  {definitions_file}"
    )

    print()
    print(
        f"Metadata:\n"
        f"  {metadata_file}"
    )

    print()
    print("IMPORTANT:")
    print(
        "P2 RankIC/ICIR must use ONLY January 2022 "
        "through December 2023."
    )

    print(
        "P3 correlation must also use ONLY January 2022 "
        "through December 2023."
    )

    print(
        "2024-present data remains untouched for feature "
        "selection and is reserved for final evaluation."
    )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
