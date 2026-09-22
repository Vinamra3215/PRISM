"""Create PRISM Category 2's final Phase-1 selection and stationarity reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from compute_category2_p1 import FEATURES


HERE = Path(__file__).resolve().parent
P1_FILE = HERE / "phase1_final" / "category2_all_companies_features.parquet"
P2_FILE = HERE / "p2_rankic_icir" / "category2_p2_summary.csv"
P3_SURVIVORS = HERE / "p3_correlation" / "category2_p3_survivors.csv"
P3_DROPPED = HERE / "p3_correlation" / "category2_p3_dropped_features.csv"
OUTPUT_DIR = HERE / "phase1_final"
START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2023-12-31")
ADF_THRESHOLD = 0.05
ICIR_SURVIVAL_THRESHOLD = 0.018


def adf_result(values: pd.Series) -> dict:
    clean = values.replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 20:
        return {"stationary": None, "p_value": np.nan, "status": "INSUFFICIENT_DATA", "observations": len(clean)}
    if clean.nunique() <= 1:
        return {"stationary": True, "p_value": 0.0, "status": "CONSTANT_SERIES", "observations": len(clean)}
    try:
        p_value = float(adfuller(clean, autolag="AIC", result_object=False)[1])
        return {"stationary": p_value < ADF_THRESHOLD, "p_value": p_value, "status": "TESTED", "observations": len(clean)}
    except Exception as error:
        return {"stationary": None, "p_value": np.nan, "status": f"ERROR: {error}", "observations": len(clean)}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    p1 = pd.read_parquet(P1_FILE)
    p1["date"] = pd.to_datetime(p1["date"])
    p1 = p1[p1["date"].between(START_DATE, END_DATE)]
    p2 = pd.read_csv(P2_FILE)
    survivors = pd.read_csv(P3_SURVIVORS)["feature"].tolist()
    dropped = pd.read_csv(P3_DROPPED)
    correlated_with = dict(zip(dropped.get("dropped_feature", []), dropped.get("kept_feature", [])))

    stationarity_rows = []
    for company, company_data in p1.groupby("company", sort=True):
        company_data = company_data.sort_values("date")
        for feature in FEATURES:
            stationarity_rows.append({"company": company, "feature": feature, **adf_result(company_data[feature])})
    stationarity = pd.DataFrame(stationarity_rows)
    stationarity.to_csv(OUTPUT_DIR / "category2_stationarity_by_company.csv", index=False)

    final_rows = []
    for feature in FEATURES:
        metric = p2[p2["feature"].eq(feature)].iloc[0]
        tested = stationarity[stationarity["feature"].eq(feature) & stationarity["stationary"].notna()]
        stationary_count = int(tested["stationary"].sum())
        if len(tested) == 0:
            stationary_label = "Unknown"
        elif stationary_count == len(tested):
            stationary_label = "Yes"
        elif stationary_count == 0:
            stationary_label = "No"
        else:
            stationary_label = "Mixed"
        if feature in survivors:
            verdict = "KEEP"
        elif feature in correlated_with:
            verdict = "DISCARD (Redundant)"
        elif abs(metric["ICIR"]) < ICIR_SURVIVAL_THRESHOLD:
            verdict = "DISCARD (Low ICIR)"
        else:
            verdict = "DISCARD"
        final_rows.append(
            {
                "Feature": feature,
                "Stationary?": stationary_label,
                "RankIC": metric["RankIC"],
                "ICIR": metric["ICIR"],
                "Dropped Correlated With": correlated_with.get(feature, ""),
                "Verdict": verdict,
            }
        )
    final = pd.DataFrame(final_rows)
    final_path = OUTPUT_DIR / "category2_phase1_final_feature_selection.csv"
    final.to_csv(final_path, index=False)
    kept = final.loc[final["Verdict"].eq("KEEP"), "Feature"].tolist()
    metadata = {
        "category": 2,
        "phase": "Phase 1",
        "selection_period": {"start": "2022-01-01", "end": "2023-12-31"},
        "feature_count_initial": len(FEATURES),
        "p2_icir_survival_threshold": ICIR_SURVIVAL_THRESHOLD,
        "p2_threshold_note": (
            "User-requested override of 0.15, calibrated to produce 6-7 "
            "features after P3 correlation pruning."
        ),
        "p2_survivor_count": int((p2["ICIR"].abs() >= ICIR_SURVIVAL_THRESHOLD).sum()),
        "p3_survivor_count": len(kept),
        "final_features": kept,
        "stationarity_method": "Augmented Dickey-Fuller by company",
        "stationarity_pvalue_threshold": ADF_THRESHOLD,
        "rankic_method": "Daily cross-sectional Spearman correlation",
        "p3_high_correlation_threshold": 0.70,
        "p3_selection_basis": "Higher absolute ICIR",
        "uses_2024_or_later_for_selection": False,
        "model_training": False,
        "guide_target_final_feature_count": "6-7",
        "target_achieved": 6 <= len(kept) <= 7,
        "overall_status": "PASS" if 6 <= len(kept) <= 7 else "TARGET_NOT_ACHIEVED",
    }
    (OUTPUT_DIR / "category2_phase1_final_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report_lines = [
        "PRISM CATEGORY 2 - PHASE 1 REPORT",
        "=================================",
        "",
        "P1 feature computation: all 50 stocks, January 2022 through latest available date.",
        "P2 RankIC/ICIR selection: January 2022 through December 2023 only.",
        "P3 within-category correlation: January 2022 through December 2023 only.",
        "P2 rule used: discard features with |ICIR| below 0.018.",
        "This is a user-requested calibrated override of the guide's original 0.15 cutoff.",
        "P3 rule from guide: for |Spearman correlation| above 0.70, keep higher |ICIR|.",
        "No model training or LightGBM was performed in Phase 1.",
        "No 2024-or-later observations were used for feature selection.",
        "",
        f"Initial features: {len(FEATURES)}",
        f"P2 survivors: {int((p2['ICIR'].abs() >= ICIR_SURVIVAL_THRESHOLD).sum())}",
        f"P3/final survivors: {len(kept)}",
        "",
        "RESULT:",
        f"The calibrated cutoff admits {int((p2['ICIR'].abs() >= ICIR_SURVIVAL_THRESHOLD).sum())} features to P3.",
        f"Correlation pruning leaves {len(kept)} final Category 2 features.",
        "",
        final.to_string(index=False),
        "",
    ]
    (OUTPUT_DIR / "category2_phase1_report.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    print(final.to_string(index=False))
    print(f"\nFinal kept features ({len(kept)}): {', '.join(kept)}")
    print(f"Saved: {final_path}")


if __name__ == "__main__":
    main()
