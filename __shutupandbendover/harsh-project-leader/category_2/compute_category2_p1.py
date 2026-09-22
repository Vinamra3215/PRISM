"""PRISM Category 2, Phase 1/P1: compute volatility features for 50 stocks."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "yatin" / "final features" / "NIFTY50_OHLCV"
FEATURE_DIR = HERE / "feature_computation"
VALIDATION_DIR = HERE / "validation"
METADATA_DIR = HERE / "metadata"
FINAL_DIR = HERE / "phase1_final"

FEATURES = [
    "rolling_volatility_5d",
    "rolling_volatility_10d",
    "rolling_volatility_20d",
    "rolling_volatility_60d",
    "volatility_ratio_5d_20d",
    "volatility_ratio_20d_60d",
    "atr_14d",
    "normalized_atr_14d",
    "garman_klass_volatility_20d",
    "parkinson_volatility_20d",
    "rolling_skewness_20d",
    "rolling_kurtosis_20d",
]
ANNUALIZATION_DAYS = 252


def company_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("INDIGO"):
        return "INDIGO"
    if stem.startswith("JIOFIN"):
        return "JIOFIN"
    return stem.removesuffix("_OHLCV")


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def load_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    frame.columns = [str(column).strip().lower().replace("_", " ") for column in frame.columns]
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    frame = frame[required]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in required[1:]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError(f"{path.name}: invalid dates")
    return frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def compute_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = frame["close"]
    log_return = np.log(close / close.shift(1))
    annualizer = np.sqrt(ANNUALIZATION_DAYS)
    output = pd.DataFrame(index=frame.index)

    for window in (5, 10, 20, 60):
        output[f"rolling_volatility_{window}d"] = (
            log_return.rolling(window, min_periods=window).std(ddof=1) * annualizer
        )

    output["volatility_ratio_5d_20d"] = safe_divide(
        output["rolling_volatility_5d"], output["rolling_volatility_20d"]
    )
    output["volatility_ratio_20d_60d"] = safe_divide(
        output["rolling_volatility_20d"], output["rolling_volatility_60d"]
    )

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    output["atr_14d"] = true_range.rolling(14, min_periods=14).mean()
    output["normalized_atr_14d"] = safe_divide(output["atr_14d"], close)

    log_hl = np.log(frame["high"] / frame["low"])
    log_co = np.log(frame["close"] / frame["open"])
    gk_daily_variance = 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)
    gk_variance = gk_daily_variance.rolling(20, min_periods=20).mean().clip(lower=0)
    output["garman_klass_volatility_20d"] = np.sqrt(gk_variance * ANNUALIZATION_DAYS)

    parkinson_daily_variance = log_hl.pow(2) / (4 * np.log(2))
    parkinson_variance = parkinson_daily_variance.rolling(20, min_periods=20).mean().clip(lower=0)
    output["parkinson_volatility_20d"] = np.sqrt(parkinson_variance * ANNUALIZATION_DAYS)

    output["rolling_skewness_20d"] = log_return.rolling(20, min_periods=20).skew()
    output["rolling_kurtosis_20d"] = log_return.rolling(20, min_periods=20).kurt()
    return output.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    for directory in (FEATURE_DIR, VALIDATION_DIR, METADATA_DIR, FINAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_DIR.glob("*.parquet"))
    if len(files) != 50:
        raise RuntimeError(f"Expected exactly 50 raw Parquet files, found {len(files)}")

    all_frames, company_rows = [], []
    for index, path in enumerate(files, 1):
        company = company_name(path)
        raw = load_ohlcv(path)
        features = compute_features(raw)
        result = pd.concat([raw[["date"]], features], axis=1)
        result.insert(0, "company", company)
        output_path = FEATURE_DIR / f"{company}_category2_features.parquet"
        result.to_parquet(output_path, index=False)
        all_frames.append(result)
        row = {
            "company": company,
            "source_file": path.name,
            "output_file": output_path.name,
            "rows": len(result),
            "start_date": result["date"].min(),
            "end_date": result["date"].max(),
            "feature_nan_total": int(result[FEATURES].isna().sum().sum()),
            "feature_inf_total": int(np.isinf(result[FEATURES].to_numpy(float)).sum()),
        }
        row.update({f"{feature}_nan": int(result[feature].isna().sum()) for feature in FEATURES})
        company_rows.append(row)
        print(f"[{index:02d}/50] {company:<15} rows={len(result)}")

    combined = pd.concat(all_frames, ignore_index=True).sort_values(["date", "company"])
    combined.to_parquet(FINAL_DIR / "category2_all_companies_features.parquet", index=False)
    validation = pd.DataFrame(company_rows).sort_values("company")
    validation.to_csv(VALIDATION_DIR / "p1_company_validation.csv", index=False)
    combined.groupby("company").agg(
        rows=("date", "count"), start_date=("date", "min"), end_date=("date", "max")
    ).reset_index().to_csv(VALIDATION_DIR / "p1_company_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "feature": feature,
                "total_rows": len(combined),
                "valid_values": int(combined[feature].notna().sum()),
                "nan_values": int(combined[feature].isna().sum()),
                "nan_percentage": float(combined[feature].isna().mean() * 100),
            }
            for feature in FEATURES
        ]
    ).to_csv(VALIDATION_DIR / "p1_feature_nan_report.csv", index=False)

    definitions = {
        "rolling_volatility_5d": "Annualized sample standard deviation of daily log returns over 5 days",
        "rolling_volatility_10d": "Annualized sample standard deviation of daily log returns over 10 days",
        "rolling_volatility_20d": "Annualized sample standard deviation of daily log returns over 20 days",
        "rolling_volatility_60d": "Annualized sample standard deviation of daily log returns over 60 days",
        "volatility_ratio_5d_20d": "5-day rolling volatility divided by 20-day rolling volatility",
        "volatility_ratio_20d_60d": "20-day rolling volatility divided by 60-day rolling volatility",
        "atr_14d": "14-day simple average of True Range",
        "normalized_atr_14d": "14-day ATR divided by Close",
        "garman_klass_volatility_20d": "20-day annualized Garman-Klass OHLC volatility",
        "parkinson_volatility_20d": "20-day annualized Parkinson high-low volatility",
        "rolling_skewness_20d": "20-day skewness of daily log returns",
        "rolling_kurtosis_20d": "20-day excess kurtosis of daily log returns",
    }
    pd.DataFrame(
        [{"feature": key, "definition": value, "category": "Volatility Features"} for key, value in definitions.items()]
    ).to_csv(METADATA_DIR / "category2_feature_definitions.csv", index=False)
    metadata = {
        "project": "PRISM",
        "category": 2,
        "category_name": "Volatility Features",
        "phase": 1,
        "step": "P1",
        "raw_data_directory": str(DATA_DIR),
        "data_period": "Full available data: January 2022 to present",
        "number_of_companies": 50,
        "number_of_features": len(FEATURES),
        "features": FEATURES,
        "parameters": {
            "return_type": "daily log return",
            "annualization_days": ANNUALIZATION_DAYS,
            "atr_window": 14,
            "range_estimator_window": 20,
            "skewness_and_kurtosis_window": 20,
        },
        "uses_2024_or_later_for_selection": False,
        "model_training": False,
        "continuous_learning": False,
    }
    (METADATA_DIR / "category2_p1_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"P1 complete: {len(combined)} rows -> {FINAL_DIR}")


if __name__ == "__main__":
    main()
