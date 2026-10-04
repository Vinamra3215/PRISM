import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

OUTPUT_DIR = os.path.join(RESULTS_DIR, "phase2_c1")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

CATEGORY_DIRS = {
    "Category 1": "p1_cat1",
    "Category 2": "p1_cat2",
    "Category 3": "p1_cat3",
    "Category 4": "p1_cat4",
    "Category 5": "p1_cat5",
    "Category 6": "p1_cat6",
    "Category 7": "p1_cat7",
    "Category 8": "p1_cat8",
}

SELECTION_START = "2022-01-01"
SELECTION_END = "2023-12-31"

CORRELATION_THRESHOLD = 0.70

TARGET_FEATURES = 35


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates, description, filename):
    """
    Find a column using case-insensitive matching.
    """
    lower_map = {str(col).strip().lower(): col for col in df.columns}

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in lower_map:
            return lower_map[key]

    raise ValueError(
        f"\nCould not find {description} in {filename}.\n"
        f"Expected one of: {candidates}\n"
        f"Actual columns: {df.columns.tolist()}\n"
    )


def load_category(category_name, directory):
    """
    Load:
      - selected_features.csv
      - icir_metrics.csv
      - features_full.parquet
    """

    category_path = os.path.join(RESULTS_DIR, directory)

    selected_file = os.path.join(
        category_path,
        "selected_features.csv"
    )

    icir_file = os.path.join(
        category_path,
        "icir_metrics.csv"
    )

    features_file = os.path.join(
        category_path,
        "features_full.parquet"
    )

    if not os.path.exists(selected_file):
        raise FileNotFoundError(
            f"{category_name}: missing {selected_file}"
        )

    if not os.path.exists(icir_file):
        raise FileNotFoundError(
            f"{category_name}: missing {icir_file}"
        )

    if not os.path.exists(features_file):
        raise FileNotFoundError(
            f"{category_name}: missing {features_file}"
        )

    # --------------------------------------------------------
    # Selected features
    # --------------------------------------------------------

    selected_df = pd.read_csv(selected_file)

    selected_col = find_column(
        selected_df,
        [
            "selected_feature",
            "feature",
            "features"
        ],
        "selected feature column",
        selected_file
    )

    selected_features = (
        selected_df[selected_col]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    # --------------------------------------------------------
    # ICIR metrics
    # --------------------------------------------------------

    icir_df = pd.read_csv(icir_file)

    feature_col = find_column(
        icir_df,
        [
            "feature",
            "selected_feature",
            "features"
        ],
        "feature column in ICIR file",
        icir_file
    )

    icir_col = find_column(
        icir_df,
        [
            "ICIR",
            "icir",
            "ICIR_value",
            "icir_value"
        ],
        "ICIR column",
        icir_file
    )

    icir_df = icir_df[[feature_col, icir_col]].copy()

    icir_df.columns = ["feature", "ICIR"]

    icir_df["feature"] = (
        icir_df["feature"]
        .astype(str)
        .str.strip()
    )

    icir_df["ICIR"] = pd.to_numeric(
        icir_df["ICIR"],
        errors="coerce"
    )

    icir_lookup = dict(
        zip(
            icir_df["feature"],
            icir_df["ICIR"]
        )
    )

    # --------------------------------------------------------
    # Full feature data
    # --------------------------------------------------------

    features_df = pd.read_parquet(features_file)

    if "date" not in features_df.columns:
        raise ValueError(
            f"{category_name}: {features_file} does not contain 'date'."
        )

    if "symbol" not in features_df.columns:
        raise ValueError(
            f"{category_name}: {features_file} does not contain 'symbol'."
        )

    features_df["date"] = pd.to_datetime(
        features_df["date"]
    )

    # Keep only features selected during Phase 1
    missing_features = [
        feature
        for feature in selected_features
        if feature not in features_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"\n{category_name}: selected features are missing "
            f"from features_full.parquet:\n"
            f"{missing_features}\n"
            f"Available columns:\n"
            f"{features_df.columns.tolist()}"
        )

    keep_columns = [
        "date",
        "symbol"
    ] + selected_features

    features_df = features_df[keep_columns].copy()

    # Rename feature columns with category prefix
    rename_map = {
        feature: f"{category_name}_{feature}"
        for feature in selected_features
    }

    features_df = features_df.rename(
        columns=rename_map
    )

    # ICIR dictionary with same prefixed names
    feature_info = {}

    for feature in selected_features:
        feature_info[f"{category_name}_{feature}"] = {
            "category": category_name,
            "original_feature": feature,
            "ICIR": icir_lookup.get(feature, np.nan)
        }

    return (
        features_df,
        feature_info
    )


# ============================================================
# LOAD ALL CATEGORIES
# ============================================================

print("=" * 75)
print("C1: CROSS-CATEGORY CORRELATION CHECK")
print("=" * 75)

print("\nLoading selected features, ICIR metrics and feature data...")


category_frames = []
feature_info = {}

for category_name, directory in CATEGORY_DIRS.items():

    print(f"\n{category_name}")

    df, info = load_category(
        category_name,
        directory
    )

    category_frames.append(df)
    feature_info.update(info)

    print(
        f"  Selected features: {len(info)}"
    )

    print(
        f"  Columns loaded: {df.shape[1] - 2}"
    )


# ============================================================
# MERGE ALL CATEGORIES
# ============================================================

print("\n" + "=" * 75)
print("MERGING CATEGORY FEATURES")
print("=" * 75)

merged_df = category_frames[0].copy()

for df in category_frames[1:]:

    merged_df = merged_df.merge(
        df,
        on=["date", "symbol"],
        how="outer"
    )

print(
    f"\nMerged feature table shape: {merged_df.shape}"
)


# ============================================================
# SELECTION PERIOD
# ============================================================

merged_df = merged_df[
    (merged_df["date"] >= SELECTION_START)
    & (merged_df["date"] <= SELECTION_END)
].copy()

print(
    f"Selection-period table shape: {merged_df.shape}"
)

print(
    f"Selection period: "
    f"{SELECTION_START} -> {SELECTION_END}"
)


# ============================================================
# FEATURE LIST
# ============================================================

feature_columns = [
    col
    for col in merged_df.columns
    if col not in ["date", "symbol"]
]

print(
    f"\nTotal Phase-1 survivors: "
    f"{len(feature_columns)}"
)

if len(feature_columns) == 0:
    raise ValueError("No feature columns found.")


# ============================================================
# ICIR TABLE
# ============================================================

icir_records = []

for feature in feature_columns:

    info = feature_info.get(
        feature,
        {}
    )

    icir_records.append(
        {
            "feature": feature,
            "category": info.get(
                "category",
                ""
            ),
            "original_feature": info.get(
                "original_feature",
                feature
            ),
            "ICIR": info.get(
                "ICIR",
                np.nan
            ),
            "abs_ICIR": abs(
                info.get(
                    "ICIR",
                    np.nan
                )
            )
        }
    )

icir_table = pd.DataFrame(
    icir_records
)

# Missing ICIR values are ranked last
icir_table["abs_ICIR"] = (
    pd.to_numeric(
        icir_table["abs_ICIR"],
        errors="coerce"
    )
)

icir_table = icir_table.sort_values(
    "abs_ICIR",
    ascending=False,
    na_position="last"
).reset_index(drop=True)


# ============================================================
# SAVE ICIR TABLE
# ============================================================

icir_table.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "all_phase1_survivors_icir.csv"
    ),
    index=False
)


# ============================================================
# CROSS-CATEGORY SPEARMAN CORRELATION
# ============================================================

print("\n" + "=" * 75)
print("COMPUTING CROSS-CATEGORY SPEARMAN CORRELATION")
print("=" * 75)

# Spearman correlation across stock-day observations.
#
# Each row represents one stock on one trading day.
# This is exactly the required Phase-2 cross-category setup.

feature_data = merged_df[
    feature_columns
].copy()

correlation_matrix = feature_data.corr(
    method="spearman"
)

correlation_matrix.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "cross_category_spearman_correlation.csv"
    )
)

print(
    f"\nCorrelation matrix saved."
)

print(
    f"Matrix size: "
    f"{correlation_matrix.shape[0]} x "
    f"{correlation_matrix.shape[1]}"
)


# ============================================================
# CROSS-CATEGORY REDUNDANCY REMOVAL
# ============================================================

print("\n" + "=" * 75)
print("REMOVING CROSS-CATEGORY REDUNDANCY")
print("=" * 75)

# Start with the strongest |ICIR| feature.
#
# For every later feature:
#
#     if |corr| > 0.70 with an already-selected feature:
#         compare |ICIR|
#         keep the stronger one
#
# This implements the Phase-2 C1 rule.

sorted_features = (
    icir_table["feature"]
    .tolist()
)

selected = []
dropped_records = []


for feature in sorted_features:

    if feature not in correlation_matrix.columns:
        continue

    current_icir = feature_info.get(
        feature,
        {}
    ).get(
        "ICIR",
        np.nan
    )

    if not selected:

        selected.append(feature)
        continue

    redundant_with = None

    for kept_feature in selected:

        if (
            kept_feature not in
            correlation_matrix.columns
        ):
            continue

        corr = correlation_matrix.loc[
            feature,
            kept_feature
        ]

        if pd.isna(corr):
            continue

        if abs(corr) > CORRELATION_THRESHOLD:

            redundant_with = kept_feature

            kept_icir = feature_info.get(
                kept_feature,
                {}
            ).get(
                "ICIR",
                np.nan
            )

            current_abs_icir = (
                abs(current_icir)
                if pd.notna(current_icir)
                else -np.inf
            )

            kept_abs_icir = (
                abs(kept_icir)
                if pd.notna(kept_icir)
                else -np.inf
            )

            if current_abs_icir > kept_abs_icir:

                # Current feature is stronger.
                selected.remove(
                    kept_feature
                )

                selected.append(
                    feature
                )

                dropped_records.append(
                    {
                        "dropped_feature": kept_feature,
                        "kept_feature": feature,
                        "correlation": corr,
                        "dropped_ICIR": kept_icir,
                        "kept_ICIR": current_icir,
                        "reason": (
                            "Higher absolute ICIR"
                        )
                    }
                )

            else:

                dropped_records.append(
                    {
                        "dropped_feature": feature,
                        "kept_feature": kept_feature,
                        "correlation": corr,
                        "dropped_ICIR": current_icir,
                        "kept_ICIR": kept_icir,
                        "reason": (
                            "Higher absolute ICIR"
                        )
                    }
                )

            break

    if redundant_with is None:

        selected.append(feature)


# Remove accidental duplicates while preserving order
selected = list(
    dict.fromkeys(selected)
)


# ============================================================
# SAVE REDUNDANCY RESULTS
# ============================================================

dropped_df = pd.DataFrame(
    dropped_records
)

dropped_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dropped_cross_category.csv"
    ),
    index=False
)


# ============================================================
# BUILD C1 SURVIVOR TABLE
# ============================================================

c1_records = []

for rank, feature in enumerate(
    selected,
    start=1
):

    info = feature_info.get(
        feature,
        {}
    )

    icir = info.get(
        "ICIR",
        np.nan
    )

    c1_records.append(
        {
            "C1_rank": rank,
            "feature": feature,
            "category": info.get(
                "category",
                ""
            ),
            "original_feature": info.get(
                "original_feature",
                feature
            ),
            "ICIR": icir,
            "abs_ICIR": (
                abs(icir)
                if pd.notna(icir)
                else np.nan
            )
        }
    )

c1_df = pd.DataFrame(
    c1_records
)

c1_df = c1_df.sort_values(
    "abs_ICIR",
    ascending=False,
    na_position="last"
).reset_index(drop=True)

c1_df["C1_rank"] = (
    np.arange(len(c1_df)) + 1
)


# ============================================================
# TARGET ~30-35 FEATURES
# ============================================================

if len(c1_df) > TARGET_FEATURES:

    final_c1_df = c1_df.head(
        TARGET_FEATURES
    ).copy()

else:

    final_c1_df = c1_df.copy()


# ============================================================
# SAVE FINAL C1 OUTPUT
# ============================================================

final_c1_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "c1_selected_features.csv"
    ),
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = pd.DataFrame(
    [
        {
            "phase1_survivors": len(
                feature_columns
            ),
            "after_cross_category_redundancy": len(
                c1_df
            ),
            "final_c1_features": len(
                final_c1_df
            ),
            "correlation_threshold": (
                CORRELATION_THRESHOLD
            ),
            "selection_start": (
                SELECTION_START
            ),
            "selection_end": (
                SELECTION_END
            )
        }
    ]
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "c1_summary.csv"
    ),
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 75)
print("C1 COMPLETE")
print("=" * 75)

print(
    f"\nPhase-1 survivors: "
    f"{len(feature_columns)}"
)

print(
    f"After cross-category redundancy: "
    f"{len(c1_df)}"
)

print(
    f"Final C1 features: "
    f"{len(final_c1_df)}"
)

print(
    f"\nCorrelation threshold: "
    f"|Spearman| > {CORRELATION_THRESHOLD}"
)

print("\nFinal C1 feature list:")

for _, row in final_c1_df.iterrows():

    print(
        f"{int(row['C1_rank']):2d}. "
        f"{row['feature']} "
        f"| {row['category']} "
        f"| ICIR = {row['ICIR']:.6f}"
        if pd.notna(row["ICIR"])
        else
        f"{int(row['C1_rank']):2d}. "
        f"{row['feature']} "
        f"| {row['category']} "
        f"| ICIR = NaN"
    )


print("\nOutput files:")
print(
    os.path.join(
        OUTPUT_DIR,
        "cross_category_spearman_correlation.csv"
    )
)
print(
    os.path.join(
        OUTPUT_DIR,
        "dropped_cross_category.csv"
    )
)
print(
    os.path.join(
        OUTPUT_DIR,
        "c1_selected_features.csv"
    )
)
print(
    os.path.join(
        OUTPUT_DIR,
        "all_phase1_survivors_icir.csv"
    )
)
print(
    os.path.join(
        OUTPUT_DIR,
        "c1_summary.csv"
    )
)

print("\n" + "=" * 75)