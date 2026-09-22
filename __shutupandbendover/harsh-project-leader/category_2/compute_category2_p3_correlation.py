"""PRISM Category 2, Phase 1/P3: remove redundant within-category features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
P1_FILE = HERE / "phase1_final" / "category2_all_companies_features.parquet"
P2_FILE = HERE / "p2_rankic_icir" / "category2_p2_summary.csv"
OUTPUT_DIR = HERE / "p3_correlation"
START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2023-12-31")
HIGH_CORRELATION = 0.70
MODERATE_CORRELATION = 0.40
ICIR_SURVIVAL_THRESHOLD = 0.018


def correlation_class(value: float) -> str:
    if pd.isna(value):
        return "undefined"
    if value > HIGH_CORRELATION:
        return "high"
    if value >= MODERATE_CORRELATION:
        return "moderate"
    return "low"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = pd.read_parquet(P1_FILE)
    p1["date"] = pd.to_datetime(p1["date"])
    p1 = p1[p1["date"].between(START_DATE, END_DATE)]
    p2 = pd.read_csv(P2_FILE)
    p2["abs_ICIR"] = p2["ICIR"].abs()
    candidates = p2[p2["abs_ICIR"] >= ICIR_SURVIVAL_THRESHOLD].sort_values(
        "abs_ICIR", ascending=False
    ).copy()
    candidate_features = candidates["feature"].tolist()

    correlation = p1[candidate_features].corr(method="spearman")
    correlation.to_csv(OUTPUT_DIR / "category2_p3_correlation_matrix.csv")
    pairs = []
    for index, feature_a in enumerate(candidate_features):
        for feature_b in candidate_features[index + 1 :]:
            value = correlation.loc[feature_a, feature_b]
            pairs.append(
                {
                    "feature_a": feature_a,
                    "feature_b": feature_b,
                    "spearman_correlation": value,
                    "absolute_correlation": abs(value) if np.isfinite(value) else np.nan,
                    "correlation_class": correlation_class(abs(value)),
                }
            )
    pair_columns = [
        "feature_a", "feature_b", "spearman_correlation",
        "absolute_correlation", "correlation_class",
    ]
    pairs_frame = pd.DataFrame(pairs, columns=pair_columns).sort_values(
        "absolute_correlation", ascending=False
    )
    pairs_frame.to_csv(OUTPUT_DIR / "category2_p3_correlation_pairs.csv", index=False)

    icir = candidates.set_index("feature")["abs_ICIR"].to_dict()
    dropped = {}
    high_pairs = pairs_frame[pairs_frame["absolute_correlation"] > HIGH_CORRELATION]
    for _, row in high_pairs.iterrows():
        feature_a, feature_b = row["feature_a"], row["feature_b"]
        if feature_a in dropped or feature_b in dropped:
            continue
        if icir[feature_a] >= icir[feature_b]:
            kept, removed = feature_a, feature_b
        else:
            kept, removed = feature_b, feature_a
        dropped[removed] = {
            "dropped_feature": removed,
            "kept_feature": kept,
            "correlation": row["spearman_correlation"],
            "absolute_correlation": row["absolute_correlation"],
            "dropped_abs_ICIR": icir[removed],
            "kept_abs_ICIR": icir[kept],
            "reason": "|correlation| > 0.70 and lower |ICIR|",
        }
    dropped_columns = [
        "dropped_feature", "kept_feature", "correlation", "absolute_correlation",
        "dropped_abs_ICIR", "kept_abs_ICIR", "reason",
    ]
    dropped_frame = pd.DataFrame(list(dropped.values()), columns=dropped_columns)
    dropped_frame.to_csv(OUTPUT_DIR / "category2_p3_dropped_features.csv", index=False)
    survivors = candidates[~candidates["feature"].isin(dropped)].copy()
    survivors.to_csv(OUTPUT_DIR / "category2_p3_survivors.csv", index=False)
    metadata = {
        "category": "Category 2 - Volatility Features",
        "selection_period": {"start": "2022-01-01", "end": "2023-12-31"},
        "p2_survival_threshold": ICIR_SURVIVAL_THRESHOLD,
        "threshold_note": (
            "User-requested calibrated cutoff; P3 still applies the guide's "
            "absolute Spearman correlation threshold of 0.70."
        ),
        "high_correlation_threshold": HIGH_CORRELATION,
        "moderate_correlation_threshold": MODERATE_CORRELATION,
        "selection_basis": "Higher absolute ICIR",
        "input_p2_survivors": candidate_features,
        "final_p3_survivors": survivors["feature"].tolist(),
        "dropped_features": list(dropped.values()),
        "uses_2024_or_later_for_selection": False,
    }
    (OUTPUT_DIR / "category2_p3_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Final P3 survivors:")
    print(survivors[["feature", "RankIC", "ICIR", "abs_ICIR"]].to_string(index=False))


if __name__ == "__main__":
    main()
