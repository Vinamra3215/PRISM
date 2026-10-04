import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


# =============================================================================
# CATEGORY 5 — FINAL PHASE 1 DELIVERABLE
# =============================================================================

BASE_DIR = Path(
    "/home/soq/__shutupandbendover/yatin/final features/category_5"
)

P1_FILE = (
    BASE_DIR
    / "phase1_final"
    / "category5_all_companies_features.parquet"
)

P2_FILE = (
    BASE_DIR
    / "p2_rankic_icir"
    / "category5_p2_summary.csv"
)

P3_SURVIVOR_FILE = (
    BASE_DIR
    / "p3_correlation"
    / "category5_p3_survivors.csv"
)

P3_DROPPED_FILE = (
    BASE_DIR
    / "p3_correlation"
    / "category5_p3_dropped_features.csv"
)

OUTPUT_DIR = BASE_DIR / "phase1_final"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_FILE = (
    OUTPUT_DIR
    / "category5_phase1_final_feature_selection.csv"
)

STATIONARITY_FILE = (
    OUTPUT_DIR
    / "category5_stationarity_by_company.csv"
)

METADATA_FILE = (
    OUTPUT_DIR
    / "category5_phase1_final_metadata.json"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

SELECTION_START = pd.Timestamp("2022-01-01")
SELECTION_END = pd.Timestamp("2023-12-31")

ADF_PVALUE_THRESHOLD = 0.05

EXPECTED_FEATURES = [
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


# =============================================================================
# ADF STATIONARITY TEST
# =============================================================================

def stationarity_test(series):
    """
    Augmented Dickey-Fuller test.

    H0:
        Series is non-stationary / contains a unit root.

    Decision:
        p-value < 0.05 -> stationary
        p-value >= 0.05 -> non-stationary
    """

    series = (
        pd.Series(series)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(series) < 20:
        return {
            "stationary": None,
            "p_value": np.nan,
            "status": "INSUFFICIENT_DATA",
            "observations": len(series),
        }

    # Constant series are stationary for our feature-screening purpose.
    if series.nunique() <= 1:
        return {
            "stationary": True,
            "p_value": 0.0,
            "status": "CONSTANT_SERIES",
            "observations": len(series),
        }

    try:

        result = adfuller(
            series,
            autolag="AIC"
        )

        p_value = float(result[1])

        return {
            "stationary": p_value < ADF_PVALUE_THRESHOLD,
            "p_value": p_value,
            "status": "TESTED",
            "observations": len(series),
        }

    except Exception as exc:

        return {
            "stationary": None,
            "p_value": np.nan,
            "status": f"ERROR: {str(exc)}",
            "observations": len(series),
        }


# =============================================================================
# FEATURE-LEVEL STATIONARITY LABEL
# =============================================================================

def stationarity_label(results):

    valid = results[
        results["stationary"].notna()
    ]

    if len(valid) == 0:
        return "Unknown"

    stationary_count = int(
        valid["stationary"].sum()
    )

    total_count = len(valid)

    if stationary_count == total_count:
        return "Yes"

    if stationary_count == 0:
        return "No"

    return "Mixed"


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("CATEGORY 5 — FINAL PHASE 1 FEATURE SELECTION")
print("=" * 80)


# =============================================================================
# LOAD P1
# =============================================================================

print("\nLoading P1 features...")

if not P1_FILE.exists():
    raise FileNotFoundError(
        f"P1 file not found:\n{P1_FILE}"
    )

p1 = pd.read_parquet(P1_FILE)

p1["date"] = pd.to_datetime(
    p1["date"]
)

p1_selection = p1[
    (p1["date"] >= SELECTION_START)
    & (p1["date"] <= SELECTION_END)
].copy()

print(
    f"P1 rows in selection period: "
    f"{len(p1_selection)}"
)

print(
    f"Date range: "
    f"{p1_selection['date'].min().date()} "
    f"to "
    f"{p1_selection['date'].max().date()}"
)

print(
    f"Companies: "
    f"{p1_selection['company'].nunique()}"
)


# =============================================================================
# CHECK P1 FEATURES
# =============================================================================

missing_features = [
    feature
    for feature in EXPECTED_FEATURES
    if feature not in p1_selection.columns
]

if missing_features:

    raise ValueError(
        f"Missing P1 features: {missing_features}"
    )


# =============================================================================
# LOAD P2
# =============================================================================

print("\nLoading corrected P2 results...")

if not P2_FILE.exists():
    raise FileNotFoundError(
        f"P2 file not found:\n{P2_FILE}"
    )

p2 = pd.read_csv(P2_FILE)

# IMPORTANT:
# Actual P2 columns are:
# feature, RankIC, ICIR, abs_ICIR,
# valid_daily_IC_count, verdict

required_p2_columns = [
    "feature",
    "RankIC",
    "ICIR",
]

missing_p2_columns = [
    column
    for column in required_p2_columns
    if column not in p2.columns
]

if missing_p2_columns:

    raise ValueError(
        f"Missing P2 columns: {missing_p2_columns}\n"
        f"Actual columns: {list(p2.columns)}"
    )

print(
    f"P2 features found: {len(p2)}"
)


# =============================================================================
# VERIFY ALL 10 FEATURES ARE IN P2
# =============================================================================

p2_features = set(
    p2["feature"]
)

missing_from_p2 = (
    set(EXPECTED_FEATURES)
    - p2_features
)

if missing_from_p2:

    raise ValueError(
        "These Category 5 features are missing "
        f"from P2: {sorted(missing_from_p2)}"
    )


# =============================================================================
# LOAD P3
# =============================================================================

print("\nLoading P3 correlation results...")

if not P3_SURVIVOR_FILE.exists():

    raise FileNotFoundError(
        f"P3 survivor file not found:\n"
        f"{P3_SURVIVOR_FILE}"
    )

if not P3_DROPPED_FILE.exists():

    raise FileNotFoundError(
        f"P3 dropped file not found:\n"
        f"{P3_DROPPED_FILE}"
    )

p3_survivors = pd.read_csv(
    P3_SURVIVOR_FILE
)

p3_dropped = pd.read_csv(
    P3_DROPPED_FILE
)


# =============================================================================
# VERIFY P3 COLUMNS
# =============================================================================

if "feature" not in p3_survivors.columns:

    raise ValueError(
        "P3 survivor file does not contain "
        "'feature' column."
    )

p3_survivor_features = (
    p3_survivors["feature"]
    .dropna()
    .tolist()
)

print(
    f"P3 final survivors: "
    f"{len(p3_survivor_features)}"
)

for feature in p3_survivor_features:

    print(
        f"  KEEP: {feature}"
    )


# =============================================================================
# P3 DROPPED FEATURES
# =============================================================================

print("\nP3 dropped features:")

dropped_correlated_with = {
    feature: ""
    for feature in EXPECTED_FEATURES
}

if len(p3_dropped) > 0:

    # The P3 file is expected to contain:
    # dropped_feature
    # kept_feature
    #
    # If the exact columns differ, show the actual columns
    # and stop rather than guessing.

    required_p3_drop_columns = [
        "dropped_feature",
        "kept_feature",
    ]

    missing_p3_drop_columns = [
        column
        for column in required_p3_drop_columns
        if column not in p3_dropped.columns
    ]

    if missing_p3_drop_columns:

        raise ValueError(
            "Missing P3 dropped-feature columns: "
            f"{missing_p3_drop_columns}\n"
            f"Actual columns: "
            f"{list(p3_dropped.columns)}"
        )

    for _, row in p3_dropped.iterrows():

        dropped_feature = row[
            "dropped_feature"
        ]

        kept_feature = row[
            "kept_feature"
        ]

        dropped_correlated_with[
            dropped_feature
        ] = kept_feature

        print(
            f"  {dropped_feature} "
            f"-> {kept_feature}"
        )

else:

    print("  None")


# =============================================================================
# STATIONARITY TEST
# =============================================================================

print("\n" + "-" * 80)
print("STATIONARITY TEST")
print("-" * 80)

print(
    "Method: Augmented Dickey-Fuller"
)

print(
    "Threshold: p-value < 0.05"
)

print(
    "Period: Jan 2022 - Dec 2023"
)


stationarity_records = []

companies = sorted(
    p1_selection[
        "company"
    ]
    .dropna()
    .unique()
)


for company in companies:

    company_data = (
        p1_selection[
            p1_selection["company"] == company
        ]
        .sort_values("date")
    )

    for feature in EXPECTED_FEATURES:

        result = stationarity_test(
            company_data[feature]
        )

        stationarity_records.append(
            {
                "company": company,
                "feature": feature,
                "stationary": result[
                    "stationary"
                ],
                "p_value": result[
                    "p_value"
                ],
                "status": result[
                    "status"
                ],
                "observations": result[
                    "observations"
                ],
            }
        )


stationarity_df = pd.DataFrame(
    stationarity_records
)


# =============================================================================
# SAVE COMPANY-LEVEL STATIONARITY
# =============================================================================

stationarity_df.to_csv(
    STATIONARITY_FILE,
    index=False
)


# =============================================================================
# FEATURE-LEVEL STATIONARITY SUMMARY
# =============================================================================

stationarity_summary = []


for feature in EXPECTED_FEATURES:

    feature_results = stationarity_df[
        stationarity_df["feature"] == feature
    ].copy()

    valid = feature_results[
        feature_results["stationary"].notna()
    ]

    stationary_count = int(
        valid["stationary"].sum()
    )

    nonstationary_count = int(
        (~valid["stationary"]).sum()
    )

    total_tested = len(valid)

    label = stationarity_label(
        feature_results
    )

    stationarity_summary.append(
        {
            "feature": feature,
            "stationary_label": label,
            "stationary_companies": stationary_count,
            "nonstationary_companies": (
                nonstationary_count
            ),
            "companies_tested": total_tested,
        }
    )


stationarity_summary_df = pd.DataFrame(
    stationarity_summary
)


# =============================================================================
# CREATE FINAL PHASE 1 TABLE
# =============================================================================

print("\n" + "=" * 80)
print("CREATING FINAL PHASE 1 TABLE")
print("=" * 80)


final_records = []


for feature in EXPECTED_FEATURES:

    p2_row = p2[
        p2["feature"] == feature
    ]

    if len(p2_row) == 0:

        raise ValueError(
            f"Feature missing from P2: {feature}"
        )

    p2_row = p2_row.iloc[0]

    stationarity_row = (
        stationarity_summary_df[
            stationarity_summary_df[
                "feature"
            ] == feature
        ]
        .iloc[0]
    )

    rank_ic = float(
        p2_row["RankIC"]
    )

    icir = float(
        p2_row["ICIR"]
    )

    is_p3_survivor = (
        feature in p3_survivor_features
    )

    dropped_with = (
        dropped_correlated_with.get(
            feature,
            ""
        )
    )


    # -------------------------------------------------------------------------
    # FINAL VERDICT
    # -------------------------------------------------------------------------

    if is_p3_survivor:

        verdict = "KEEP"

    elif dropped_with:

        verdict = "DROP — CORRELATED"

    elif abs(icir) < 0.10:

        verdict = "DROP — LOW ICIR"

    else:

        verdict = "DROP"


    final_records.append(
        {
            "Feature": feature,
            "Stationary?": (
                stationarity_row[
                    "stationary_label"
                ]
            ),
            "RankIC": rank_ic,
            "ICIR": icir,
            "Dropped Correlated With": (
                dropped_with
            ),
            "Verdict": verdict,
        }
    )


final_df = pd.DataFrame(
    final_records
)


# =============================================================================
# SAVE FINAL TABLE
# =============================================================================

final_df.to_csv(
    FINAL_FILE,
    index=False
)


# =============================================================================
# PRINT FINAL RESULT
# =============================================================================

print("\n" + "=" * 80)
print("CATEGORY 5 — FINAL PHASE 1 RESULT")
print("=" * 80)

print(
    final_df.to_string(
        index=False
    )
)


# =============================================================================
# FINAL FEATURE LIST
# =============================================================================

kept_features = final_df[
    final_df["Verdict"] == "KEEP"
]["Feature"].tolist()

dropped_features = final_df[
    final_df["Verdict"] != "KEEP"
]["Feature"].tolist()


print("\n" + "-" * 80)

print(
    f"Total features: {len(final_df)}"
)

print(
    f"Final kept features: {len(kept_features)}"
)

print(
    f"Dropped features: {len(dropped_features)}"
)


print("\nFINAL CATEGORY 5 FEATURES:")

for i, feature in enumerate(
    kept_features,
    start=1
):

    print(
        f"  {i}. {feature}"
    )


# =============================================================================
# VALIDATION
# =============================================================================

print("\n" + "=" * 80)
print("VALIDATION")
print("=" * 80)


checks = []


checks.append(
    (
        "10 expected features present",
        len(final_df) == 10
    )
)


checks.append(
    (
        "4 P3 survivors present",
        len(kept_features) == 4
    )
)


checks.append(
    (
        "All kept features are P3 survivors",
        set(kept_features)
        == set(p3_survivor_features)
    )
)


checks.append(
    (
        "RankIC values are finite",
        np.isfinite(
            final_df["RankIC"]
        ).all()
    )
)


checks.append(
    (
        "ICIR values are finite",
        np.isfinite(
            final_df["ICIR"]
        ).all()
    )
)


checks.append(
    (
        "No duplicate features",
        final_df["Feature"].is_unique
    )
)


checks.append(
    (
        "Final output file exists",
        FINAL_FILE.exists()
    )
)


checks.append(
    (
        "Stationarity file exists",
        STATIONARITY_FILE.exists()
    )
)


all_pass = True


for name, passed in checks:

    if passed:

        print(
            f"PASS: {name}"
        )

    else:

        print(
            f"FAIL: {name}"
        )

        all_pass = False


# =============================================================================
# METADATA
# =============================================================================

metadata = {

    "category": 5,

    "phase": "Phase 1",

    "stage": (
        "Final Phase 1 Feature Selection"
    ),

    "selection_period": {

        "start": str(
            SELECTION_START.date()
        ),

        "end": str(
            SELECTION_END.date()
        ),
    },

    "feature_count_initial": 10,

    "p2_survivor_count": 7,

    "p3_survivor_count": len(
        kept_features
    ),

    "final_features": kept_features,

    "stationarity_method": (
        "Augmented Dickey-Fuller"
    ),

    "stationarity_pvalue_threshold": (
        ADF_PVALUE_THRESHOLD
    ),

    "rankic_method": (
        "Daily cross-sectional "
        "Spearman correlation"
    ),

    "p3_correlation_method": (
        "Spearman"
    ),

    "p3_high_correlation_threshold": (
        0.70
    ),

    "p3_selection_basis": (
        "Higher absolute ICIR"
    ),

    "model_training": False,

    "lightgbm": False,

    "phase_2": False,

    "continuous_learning": False,

    "validation": {
        name: bool(passed)
        for name, passed in checks
    },

    "overall_status": (
        "PASS"
        if all_pass
        else "FAIL"
    ),
}


with open(
    METADATA_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )


# =============================================================================
# FINAL STATUS
# =============================================================================

print("\n" + "=" * 80)

if all_pass:

    print(
        "CATEGORY 5 PHASE 1 "
        "COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "CATEGORY 5 PHASE 1 "
        "VALIDATION FAILED"
    )

print("=" * 80)


print(
    f"\nFinal table:\n{FINAL_FILE}"
)

print(
    f"\nStationarity results:\n"
    f"{STATIONARITY_FILE}"
)

print(
    f"\nMetadata:\n{METADATA_FILE}"
)

print("\nModel training: NO")
print("LightGBM: NO")
print("Phase 2: NO")
print("Continuous learning: NO")

print("=" * 80)
