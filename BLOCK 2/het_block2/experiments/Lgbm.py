import os
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

C1_DIR = os.path.join(
    RESULTS_DIR,
    "cross_category_feature_filtering"
)

OUTPUT_DIR = os.path.join(
    RESULTS_DIR,
    "lightgbm_feature_selection"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# CONFIGURATION
# ============================================================

C1_FILE = os.path.join(
    C1_DIR,
    "c1_selected_features.csv"
)

TRAIN_START = "2022-01-01"
TRAIN_END = "2022-12-31"

VALID_START = "2023-01-01"
VALID_END = "2023-12-31"

# Mapping from C1 category names to their directories
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


# ============================================================
# HELPER
# ============================================================

def get_category_from_feature(feature_name):
    """
    Extract category name from a C1-prefixed feature.

    Example:
        Category 1_return_10d
        -> Category 1
    """

    for category in CATEGORY_DIRS:

        prefix = category + "_"

        if feature_name.startswith(prefix):
            return category

    raise ValueError(
        f"Could not determine category for feature: "
        f"{feature_name}"
    )


def get_original_feature_name(feature_name):
    """
    Remove the C1 category prefix.

    Example:
        Category 1_return_10d
        -> return_10d
    """

    category = get_category_from_feature(
        feature_name
    )

    prefix = category + "_"

    return feature_name[
        len(prefix):
    ]


# ============================================================
# START
# ============================================================

print("=" * 75)
print("C2: LIGHTGBM FEATURE IMPORTANCE")
print("=" * 75)


# ============================================================
# CHECK C1 OUTPUT
# ============================================================

if not os.path.exists(C1_FILE):

    raise FileNotFoundError(
        f"C1 output not found:\n{C1_FILE}"
    )


c1_df = pd.read_csv(
    C1_FILE
)

print(
    f"\nLoaded C1 feature list: "
    f"{len(c1_df)} features"
)


# ------------------------------------------------------------
# Find feature column
# ------------------------------------------------------------

if "feature" in c1_df.columns:

    feature_column = "feature"

elif "selected_feature" in c1_df.columns:

    feature_column = "selected_feature"

else:

    raise ValueError(
        "C1 file does not contain "
        "'feature' or 'selected_feature'.\n"
        f"Columns: {c1_df.columns.tolist()}"
    )


c1_features = (
    c1_df[feature_column]
    .dropna()
    .astype(str)
    .str.strip()
    .tolist()
)


print("\nC1 features:")

for i, feature in enumerate(
    c1_features,
    start=1
):

    print(
        f"{i:2d}. {feature}"
    )


# ============================================================
# LOAD TARGET
# ============================================================

print("\n" + "=" * 75)
print("LOADING TARGET")
print("=" * 75)

cat1_file = os.path.join(
    RESULTS_DIR,
    CATEGORY_DIRS["Category 1"],
    "features_full.parquet"
)

if not os.path.exists(cat1_file):

    raise FileNotFoundError(
        f"Category 1 file not found:\n{cat1_file}"
    )


target_df = pd.read_parquet(
    cat1_file,
    columns=[
        "date",
        "symbol",
        "next_day_return"
    ]
)

target_df["date"] = pd.to_datetime(
    target_df["date"]
)

print(
    f"Target rows: {len(target_df)}"
)


# ============================================================
# LOAD C1 FEATURES
# ============================================================

print("\n" + "=" * 75)
print("LOADING C1 SURVIVOR FEATURES")
print("=" * 75)


# Group requested features by category
features_by_category = {}

for feature in c1_features:

    category = get_category_from_feature(
        feature
    )

    original_feature = (
        get_original_feature_name(feature)
    )

    if category not in features_by_category:

        features_by_category[category] = []

    features_by_category[
        category
    ].append(
        (
            feature,
            original_feature
        )
    )


# Start with date and symbol
model_df = target_df.copy()


for category, feature_pairs in (
    features_by_category.items()
):

    print(
        f"\nLoading {category}..."
    )

    directory = CATEGORY_DIRS[
        category
    ]

    filepath = os.path.join(
        RESULTS_DIR,
        directory,
        "features_full.parquet"
    )

    if not os.path.exists(filepath):

        raise FileNotFoundError(
            f"Missing feature file:\n{filepath}"
        )

    required_columns = [
        "date",
        "symbol"
    ]

    original_names = [
        original
        for _, original in feature_pairs
    ]

    required_columns.extend(
        original_names
    )

    category_df = pd.read_parquet(
        filepath,
        columns=required_columns
    )

    category_df["date"] = pd.to_datetime(
        category_df["date"]
    )

    # Rename original feature names to the
    # C1-prefixed names.
    rename_map = {}

    for prefixed, original in feature_pairs:

        rename_map[
            original
        ] = prefixed

    category_df = category_df.rename(
        columns=rename_map
    )

    model_df = model_df.merge(
        category_df,
        on=[
            "date",
            "symbol"
        ],
        how="left",
        validate="one_to_one"
    )

    print(
        f"  Loaded {len(feature_pairs)} features"
    )


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in c1_features
    if feature not in model_df.columns
]

if missing_features:

    raise ValueError(
        "\nThe following C1 features were not found "
        "after merging:\n"
        + "\n".join(missing_features)
    )


print(
    f"\nFinal modeling table shape: "
    f"{model_df.shape}"
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

model_df = model_df.sort_values(
    [
        "date",
        "symbol"
    ]
).reset_index(
    drop=True
)


# ============================================================
# CREATE TRAIN / VALIDATION SETS
# ============================================================

train_df = model_df[
    (
        model_df["date"]
        >= TRAIN_START
    )
    &
    (
        model_df["date"]
        <= TRAIN_END
    )
].copy()


valid_df = model_df[
    (
        model_df["date"]
        >= VALID_START
    )
    &
    (
        model_df["date"]
        <= VALID_END
    )
].copy()


print("\n" + "=" * 75)
print("DATA SPLIT")
print("=" * 75)

print(
    f"\nTraining period: "
    f"{TRAIN_START} -> {TRAIN_END}"
)

print(
    f"Training rows before cleaning: "
    f"{len(train_df)}"
)

print(
    f"\nValidation period: "
    f"{VALID_START} -> {VALID_END}"
)

print(
    f"Validation rows before cleaning: "
    f"{len(valid_df)}"
)


# ============================================================
# PREPARE X / Y
# ============================================================

X_train = train_df[
    c1_features
].copy()

y_train = train_df[
    "next_day_return"
].copy()


X_valid = valid_df[
    c1_features
].copy()

y_valid = valid_df[
    "next_day_return"
].copy()


# ------------------------------------------------------------
# Remove rows where target is unavailable.
#
# Feature NaNs are NOT removed globally.
# LightGBM handles missing feature values natively.
# ------------------------------------------------------------

train_target_mask = (
    y_train.notna()
    &
    np.isfinite(y_train)
)

valid_target_mask = (
    y_valid.notna()
    &
    np.isfinite(y_valid)
)


X_train = X_train.loc[
    train_target_mask
].copy()

y_train = y_train.loc[
    train_target_mask
].copy()


X_valid = X_valid.loc[
    valid_target_mask
].copy()

y_valid = y_valid.loc[
    valid_target_mask
].copy()


print(
    f"\nTraining rows after target cleaning: "
    f"{len(X_train)}"
)

print(
    f"Validation rows after target cleaning: "
    f"{len(X_valid)}"
)


# ============================================================
# CHECK DATA TYPES
# ============================================================

for feature in c1_features:

    X_train[feature] = pd.to_numeric(
        X_train[feature],
        errors="coerce"
    )

    X_valid[feature] = pd.to_numeric(
        X_valid[feature],
        errors="coerce"
    )


# ============================================================
# LIGHTGBM MODEL
# ============================================================

print("\n" + "=" * 75)
print("TRAINING LIGHTGBM")
print("=" * 75)

model = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=30,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)


model.fit(
    X_train,
    y_train,
    eval_set=[
        (
            X_train,
            y_train
        ),
        (
            X_valid,
            y_valid
        )
    ],
    eval_names=[
        "train",
        "validation"
    ],
    callbacks=[
        lgb.early_stopping(
            stopping_rounds=75,
            verbose=False
        )
    ]
)


print(
    "\nTraining complete."
)

print(
    f"Best iteration: "
    f"{model.best_iteration_}"
)


# ============================================================
# GAIN IMPORTANCE
# ============================================================

print("\n" + "=" * 75)
print("CALCULATING GAIN IMPORTANCE")
print("=" * 75)


importance_df = pd.DataFrame(
    {
        "feature": c1_features,
        "gain_importance": (
            model.booster_
            .feature_importance(
                importance_type="gain"
            )
        ),
        "split_importance": (
            model.booster_
            .feature_importance(
                importance_type="split"
            )
        )
    }
)


# Add category information
importance_df["category"] = (
    importance_df["feature"]
    .apply(
        get_category_from_feature
    )
)

importance_df["original_feature"] = (
    importance_df["feature"]
    .apply(
        get_original_feature_name
    )
)


# Normalize gain to percentage
total_gain = (
    importance_df["gain_importance"]
    .sum()
)

if total_gain > 0:

    importance_df[
        "gain_importance_pct"
    ] = (
        importance_df["gain_importance"]
        / total_gain
        * 100
    )

else:

    importance_df[
        "gain_importance_pct"
    ] = 0.0


# Rank by gain
importance_df = (
    importance_df
    .sort_values(
        "gain_importance",
        ascending=False
    )
    .reset_index(
        drop=True
    )
)

importance_df[
    "gain_rank"
] = np.arange(
    1,
    len(importance_df) + 1
)


# ============================================================
# SAVE IMPORTANCE
# ============================================================

importance_file = os.path.join(
    OUTPUT_DIR,
    "lightgbm_gain_importance.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)


# ============================================================
# VALIDATION METRICS
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION PERFORMANCE")
print("=" * 75)


valid_predictions = model.predict(
    X_valid,
    num_iteration=model.best_iteration_
)


# RMSE
rmse = np.sqrt(
    np.mean(
        (
            y_valid.values
            - valid_predictions
        ) ** 2
    )
)


# MAE
mae = np.mean(
    np.abs(
        y_valid.values
        - valid_predictions
    )
)


# Information coefficient
prediction_series = pd.Series(
    valid_predictions,
    index=y_valid.index
)

ic = (
    prediction_series
    .corr(
        y_valid,
        method="spearman"
    )
)


validation_metrics = pd.DataFrame(
    [
        {
            "best_iteration": (
                model.best_iteration_
            ),
            "validation_rmse": rmse,
            "validation_mae": mae,
            "validation_spearman_ic": ic,
            "train_rows": len(X_train),
            "validation_rows": len(X_valid),
            "num_features": len(c1_features)
        }
    ]
)


validation_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "validation_metrics.csv"
    ),
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model.booster_.save_model(
    os.path.join(
        OUTPUT_DIR,
        "lightgbm_model.txt"
    )
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    f"\nValidation RMSE: "
    f"{rmse:.8f}"
)

print(
    f"Validation MAE: "
    f"{mae:.8f}"
)

print(
    f"Validation Spearman IC: "
    f"{ic:.8f}"
)


print("\n" + "=" * 75)
print("LIGHTGBM GAIN RANKING")
print("=" * 75)

for _, row in importance_df.iterrows():

    print(
        f"{int(row['gain_rank']):2d}. "
        f"{row['feature']:<45} "
        f"Gain = "
        f"{row['gain_importance']:.4f} "
        f"({row['gain_importance_pct']:.2f}%)"
    )


# ============================================================
# FINAL OUTPUTS
# ============================================================

print("\n" + "=" * 75)
print("C2 COMPLETE")
print("=" * 75)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nFiles created:")

print(
    "  lightgbm_gain_importance.csv"
)

print(
    "  validation_metrics.csv"
)

print(
    "  lightgbm_model.txt"
)

print("\nNext stage:")
print(
    "C3: combine ICIR ranking + "
    "LightGBM gain ranking"
)

print("=" * 75)