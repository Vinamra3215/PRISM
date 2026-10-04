import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CATEGORY 5 — PHASE 1 — P3
# WITHIN-CATEGORY SPEARMAN CORRELATION
# ============================================================

BASE_DIR = Path(
    "/home/soq/__shutupandbendover/yatin/final features"
)

P1_FILE = (
    BASE_DIR
    / "category_5"
    / "phase1_final"
    / "category5_all_companies_features.parquet"
)

P2_SUMMARY_FILE = (
    BASE_DIR
    / "category_5"
    / "p2_rankic_icir"
    / "category5_p2_summary.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "category_5"
    / "p3_correlation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PERIOD
# ============================================================

START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2023-12-31")


# ============================================================
# CORRELATION THRESHOLDS
# ============================================================

HIGH_CORRELATION = 0.70
MODERATE_CORRELATION = 0.40


# ============================================================
# LOAD P1 DATA
# ============================================================

print("=" * 80)
print("CATEGORY 5 — P3 WITHIN-CATEGORY CORRELATION")
print("=" * 80)

print("\nLoading P1 features...")

p1 = pd.read_parquet(P1_FILE)

p1["date"] = pd.to_datetime(
    p1["date"]
)

p1 = p1[
    (p1["date"] >= START_DATE)
    & (p1["date"] <= END_DATE)
].copy()

print(
    f"P1 rows in P3 period: {len(p1)}"
)

print(
    f"Date range: "
    f"{p1['date'].min().date()} "
    f"to "
    f"{p1['date'].max().date()}"
)

print(
    f"Companies: "
    f"{p1['company'].nunique()}"
)


# ============================================================
# LOAD P2 RESULTS
# ============================================================

print("\nLoading corrected P2 results...")

p2 = pd.read_csv(
    P2_SUMMARY_FILE
)

p2["abs_ICIR"] = p2["ICIR"].abs()


# ============================================================
# P2 SURVIVORS
# ============================================================

# P2 threshold:
# Keep features with |ICIR| >= 0.1
#
# This corresponds to:
# Strong Keep       > 0.4
# Decent Keep       0.2–0.4
# Weak Keep         0.1–0.2
#
# Features below 0.1 are discarded.

survivors = p2[
    p2["abs_ICIR"] >= 0.10
].copy()

survivors = survivors.sort_values(
    "abs_ICIR",
    ascending=False
).reset_index(
    drop=True
)

survivor_features = (
    survivors["feature"]
    .tolist()
)

print("\n" + "-" * 80)
print("P2 SURVIVORS ENTERING P3")
print("-" * 80)

for i, row in survivors.iterrows():

    print(
        f"{i + 1}. "
        f"{row['feature']:<35} "
        f"ICIR={row['ICIR']:.6f}"
    )

print(
    f"\nNumber of P2 survivors: "
    f"{len(survivor_features)}"
)


if len(survivor_features) == 0:

    raise ValueError(
        "No features survived P2."
    )


# ============================================================
# VERIFY FEATURES EXIST
# ============================================================

missing_features = [
    feature
    for feature in survivor_features
    if feature not in p1.columns
]

if missing_features:

    raise ValueError(
        f"Missing P2 survivor columns in P1: "
        f"{missing_features}"
    )


# ============================================================
# CREATE CORRELATION DATASET
# ============================================================

corr_data = p1[
    ["company", "date"]
    + survivor_features
].copy()

print("\nCorrelation dataset:")
print(
    f"Rows: {len(corr_data)}"
)

print(
    f"Feature columns: "
    f"{len(survivor_features)}"
)


# ============================================================
# SPEARMAN CORRELATION MATRIX
# ============================================================

print("\n" + "-" * 80)
print("CALCULATING SPEARMAN FEATURE-FEATURE CORRELATION")
print("-" * 80)

correlation_matrix = corr_data[
    survivor_features
].corr(
    method="spearman"
)

print("\nCorrelation matrix:")
print(
    correlation_matrix.to_string()
)


# ============================================================
# SAVE CORRELATION MATRIX
# ============================================================

matrix_output = (
    OUTPUT_DIR
    / "category5_p3_correlation_matrix.csv"
)

correlation_matrix.to_csv(
    matrix_output
)


# ============================================================
# CREATE ALL UNIQUE FEATURE PAIRS
# ============================================================

pairs = []

for i in range(
    len(survivor_features)
):

    feature_a = survivor_features[i]

    for j in range(
        i + 1,
        len(survivor_features)
    ):

        feature_b = survivor_features[j]

        corr = correlation_matrix.loc[
            feature_a,
            feature_b
        ]

        pairs.append(
            {
                "feature_a": feature_a,
                "feature_b": feature_b,
                "spearman_correlation": corr,
                "absolute_correlation": abs(corr),
            }
        )


pairs_df = pd.DataFrame(
    pairs
)

pairs_df = pairs_df.sort_values(
    "absolute_correlation",
    ascending=False
).reset_index(
    drop=True
)


# ============================================================
# CLASSIFY CORRELATION
# ============================================================

def correlation_class(abs_corr):

    if pd.isna(abs_corr):
        return "UNDEFINED"

    if abs_corr > HIGH_CORRELATION:
        return "HIGH > 0.70"

    if abs_corr >= MODERATE_CORRELATION:
        return "MODERATE 0.40–0.70"

    return "LOW < 0.40"


pairs_df[
    "correlation_class"
] = pairs_df[
    "absolute_correlation"
].apply(
    correlation_class
)


# ============================================================
# SAVE ALL PAIRS
# ============================================================

pairs_output = (
    OUTPUT_DIR
    / "category5_p3_correlation_pairs.csv"
)

pairs_df.to_csv(
    pairs_output,
    index=False
)


# ============================================================
# IDENTIFY HIGH-CORRELATION PAIRS
# ============================================================

high_pairs = pairs_df[
    pairs_df[
        "absolute_correlation"
    ] > HIGH_CORRELATION
].copy()

print("\n" + "-" * 80)
print("HIGH-CORRELATION PAIRS |corr| > 0.70")
print("-" * 80)

if len(high_pairs) == 0:

    print(
        "No high-correlation pairs found."
    )

else:

    print(
        high_pairs[
            [
                "feature_a",
                "feature_b",
                "spearman_correlation",
                "absolute_correlation",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# DETERMINE DROPS
# ============================================================

# We use the P2 |ICIR| to decide which feature survives
# when |correlation| > 0.70.
#
# IMPORTANT:
# A feature already marked for dropping is not selected again.
# We process pairs from highest correlation downward.

icir_map = dict(
    zip(
        survivors["feature"],
        survivors["ICIR"]
    )
)

abs_icir_map = {
    feature: abs(value)
    for feature, value
    in icir_map.items()
}


dropped_features = {}

for _, row in high_pairs.iterrows():

    feature_a = row[
        "feature_a"
    ]

    feature_b = row[
        "feature_b"
    ]

    # If both already dropped, nothing to do.
    if (
        feature_a in dropped_features
        and feature_b in dropped_features
    ):
        continue

    # If A already dropped, B survives.
    if feature_a in dropped_features:
        continue

    # If B already dropped, A survives.
    if feature_b in dropped_features:
        continue

    icir_a = abs_icir_map[
        feature_a
    ]

    icir_b = abs_icir_map[
        feature_b
    ]

    # Drop lower |ICIR|.
    if icir_a < icir_b:

        dropped_features[
            feature_a
        ] = {
            "dropped_feature": feature_a,
            "kept_feature": feature_b,
            "correlation": row[
                "spearman_correlation"
            ],
            "absolute_correlation": row[
                "absolute_correlation"
            ],
            "dropped_abs_ICIR": icir_a,
            "kept_abs_ICIR": icir_b,
            "reason": (
                "|correlation| > 0.70 and "
                "lower |ICIR|"
            ),
        }

    elif icir_b < icir_a:

        dropped_features[
            feature_b
        ] = {
            "dropped_feature": feature_b,
            "kept_feature": feature_a,
            "correlation": row[
                "spearman_correlation"
            ],
            "absolute_correlation": row[
                "absolute_correlation"
            ],
            "dropped_abs_ICIR": icir_b,
            "kept_abs_ICIR": icir_a,
            "reason": (
                "|correlation| > 0.70 and "
                "lower |ICIR|"
            ),
        }

    else:

        # Exact tie:
        # keep feature_a because both have identical |ICIR|.
        dropped_features[
            feature_b
        ] = {
            "dropped_feature": feature_b,
            "kept_feature": feature_a,
            "correlation": row[
                "spearman_correlation"
            ],
            "absolute_correlation": row[
                "absolute_correlation"
            ],
            "dropped_abs_ICIR": icir_b,
            "kept_abs_ICIR": icir_a,
            "reason": (
                "|correlation| > 0.70 and "
                "equal |ICIR|; first feature kept"
            ),
        }


# ============================================================
# DROPPED FEATURES TABLE
# ============================================================

if dropped_features:

    dropped_df = pd.DataFrame(
        list(
            dropped_features.values()
        )
    )

else:

    dropped_df = pd.DataFrame(
        columns=[
            "dropped_feature",
            "kept_feature",
            "correlation",
            "absolute_correlation",
            "dropped_abs_ICIR",
            "kept_abs_ICIR",
            "reason",
        ]
    )


dropped_output = (
    OUTPUT_DIR
    / "category5_p3_dropped_features.csv"
)

dropped_df.to_csv(
    dropped_output,
    index=False
)


# ============================================================
# FINAL P3 SURVIVORS
# ============================================================

final_survivors = [
    feature
    for feature in survivor_features
    if feature not in dropped_features
]


survivors_output_df = survivors[
    survivors["feature"].isin(
        final_survivors
    )
].copy()

survivors_output_df[
    "P3_status"
] = "KEEP"


survivors_output_df = survivors_output_df[
    [
        "feature",
        "RankIC",
        "ICIR",
        "abs_ICIR",
        "verdict",
        "P3_status",
    ]
].sort_values(
    "abs_ICIR",
    ascending=False
).reset_index(
    drop=True
)


survivors_output = (
    OUTPUT_DIR
    / "category5_p3_survivors.csv"
)

survivors_output_df.to_csv(
    survivors_output,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("CATEGORY 5 — P3 FINAL RESULT")
print("=" * 80)

print(
    f"\nP2 survivors: "
    f"{len(survivor_features)}"
)

print(
    f"High-correlation pairs: "
    f"{len(high_pairs)}"
)

print(
    f"Features dropped as redundant: "
    f"{len(dropped_features)}"
)

print(
    f"Final P3 survivors: "
    f"{len(final_survivors)}"
)

if dropped_features:

    print("\nDROPPED FEATURES:")

    for feature, info in dropped_features.items():

        print(
            f"  {feature}"
            f"  → kept {info['kept_feature']}"
            f" | corr={info['correlation']:.6f}"
            f" | |ICIR| dropped="
            f"{info['dropped_abs_ICIR']:.6f}"
            f" | |ICIR| kept="
            f"{info['kept_abs_ICIR']:.6f}"
        )

print("\nFINAL P3 SURVIVORS:")

for i, feature in enumerate(
    final_survivors,
    start=1
):

    print(
        f"  {i}. {feature}"
    )


# ============================================================
# METADATA
# ============================================================

metadata = {

    "category":
        "Category 5 — Candlestick & Price Structure",

    "phase":
        "Phase 1",

    "step":
        "P3 — Within-Category Correlation",

    "period": {
        "start": "2022-01-01",
        "end": "2023-12-31",
    },

    "input_p2_survivor_count":
        len(survivor_features),

    "input_p2_survivors":
        survivor_features,

    "high_correlation_threshold":
        HIGH_CORRELATION,

    "moderate_correlation_threshold":
        MODERATE_CORRELATION,

    "high_correlation_action":
        "Drop feature with lower absolute ICIR.",

    "final_p3_survivor_count":
        len(final_survivors),

    "final_p3_survivors":
        final_survivors,

    "dropped_features":
        list(dropped_features.values()),

    "uses_model_training":
        False,

    "uses_lightgbm":
        False,

    "uses_phase_2":
        False,

    "continuous_learning":
        False,
}


metadata_output = (
    OUTPUT_DIR
    / "category5_p3_metadata.json"
)

with open(
    metadata_output,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metadata,
        f,
        indent=2,
        default=str
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print("\n" + "-" * 80)
print("OUTPUT FILES")
print("-" * 80)

print(
    matrix_output
)

print(
    pairs_output
)

print(
    dropped_output
)

print(
    survivors_output
)

print(
    metadata_output
)

print("\n" + "=" * 80)
print("P3 COMPLETED SUCCESSFULLY")
print("=" * 80)
print("Period: Jan 2022 – Dec 2023")
print("Correlation: Spearman")
print("High correlation threshold: |corr| > 0.70")
print("Selection basis: higher |ICIR|")
print("Model training: NO")
print("LightGBM: NO")
print("Phase 2: NO")
print("Continuous learning: NO")
print("=" * 80)
