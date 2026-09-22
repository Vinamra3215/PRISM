"""PRISM Category 2, Phase 1/P2: daily cross-sectional RankIC and ICIR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from compute_category2_p1 import DATA_DIR, FEATURES, company_name, load_ohlcv


HERE = Path(__file__).resolve().parent
P1_FILE = HERE / "phase1_final" / "category2_all_companies_features.parquet"
OUTPUT_DIR = HERE / "p2_rankic_icir"
START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2023-12-31")
# User-requested calibrated cutoff.  On the current 2022-2023 data this sends
# eight features into P3, where correlation pruning leaves six final features.
ICIR_SURVIVAL_THRESHOLD = 0.018


def verdict(icir: float) -> str:
    if not np.isfinite(icir) or abs(icir) < ICIR_SURVIVAL_THRESHOLD:
        return "DISCARD"
    if abs(icir) > 0.4:
        return "STRONG KEEP"
    if abs(icir) >= 0.2:
        return "DECENT KEEP"
    return "WEAK KEEP IF NEEDED"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(P1_FILE)
    features["date"] = pd.to_datetime(features["date"])
    features = features[features["date"].between(START_DATE, END_DATE)].copy()
    if features["company"].nunique() != 50:
        raise ValueError("P1 input must contain all 50 companies")

    target_parts = []
    files = sorted(DATA_DIR.glob("*.parquet"))
    if len(files) != 50:
        raise ValueError(f"Expected 50 raw files, found {len(files)}")
    for path in files:
        raw = load_ohlcv(path)
        raw["next_day_return"] = raw["close"].shift(-1) / raw["close"] - 1
        raw["target_date"] = raw["date"].shift(-1)
        raw["company"] = company_name(path)
        target_parts.append(raw[["company", "date", "target_date", "next_day_return"]])
    targets = pd.concat(target_parts, ignore_index=True)
    merged = features.merge(targets, on=["company", "date"], how="left", validate="one_to_one")
    # Both the feature date and the realized target date must remain inside the
    # guide's training/validation boundary.  In particular, the last 2023 row
    # must not use a next-trading-day return realized in January 2024.
    merged = merged[
        merged["target_date"].between(START_DATE, END_DATE)
    ].dropna(subset=["next_day_return"])

    records = []
    for date, day in merged.groupby("date", sort=True):
        for feature in FEATURES:
            valid = day[[feature, "next_day_return"]].replace([np.inf, -np.inf], np.nan).dropna()
            correlation = np.nan
            if len(valid) >= 3 and valid[feature].nunique() > 1 and valid["next_day_return"].nunique() > 1:
                correlation = float(spearmanr(valid[feature], valid["next_day_return"]).statistic)
            records.append(
                {"date": date, "feature": feature, "daily_rankic": correlation, "n_stocks": len(valid)}
            )
    daily = pd.DataFrame(records)
    daily.to_csv(OUTPUT_DIR / "category2_p2_daily_rankic.csv", index=False)

    rows = []
    for feature in FEATURES:
        values = daily.loc[daily["feature"].eq(feature), "daily_rankic"].dropna()
        rank_ic = float(values.mean()) if len(values) else np.nan
        standard_deviation = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        icir = rank_ic / standard_deviation if np.isfinite(standard_deviation) and standard_deviation != 0 else np.nan
        rows.append(
            {
                "feature": feature,
                "RankIC": rank_ic,
                "ICIR": icir,
                "abs_ICIR": abs(icir) if np.isfinite(icir) else np.nan,
                "valid_daily_IC_count": len(values),
                "verdict": verdict(icir),
            }
        )
    summary = pd.DataFrame(rows).sort_values("abs_ICIR", ascending=False, na_position="last")
    summary.to_csv(OUTPUT_DIR / "category2_p2_summary.csv", index=False)
    metadata = {
        "category": "Category 2 - Volatility Features",
        "phase": "Phase 1",
        "step": "P2 - RankIC and ICIR",
        "selection_period": {"start": "2022-01-01", "end": "2023-12-31"},
        "number_of_companies": 50,
        "features": FEATURES,
        "method": "Daily cross-sectional Spearman correlation with next-trading-day actual return",
        "target_boundary_rule": "Both feature date and next-trading-day target date are within 2022-2023",
        "icir_definition": "Mean daily RankIC divided by sample standard deviation of daily RankIC",
        "survival_threshold": ICIR_SURVIVAL_THRESHOLD,
        "discard_rule": "Discard features with absolute ICIR below 0.018",
        "threshold_note": (
            "User-requested override of the guide's 0.15 cutoff, calibrated "
            "so P3 retains 6-7 Category 2 features on the supplied data."
        ),
        "uses_2024_or_later_for_selection": False,
        "model_training": False,
    }
    (OUTPUT_DIR / "category2_p2_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
