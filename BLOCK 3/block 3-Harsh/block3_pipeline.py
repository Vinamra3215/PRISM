#!/usr/bin/env python3
"""Prepare data, tune/train three Block 3 models, and evaluate the test once."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import plotly.graph_objects as go
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from models import build_model


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
ROOT = PROJECT.parent
DEFAULT_PANEL = ROOT / "tanishq/phase2_data/nifty50_phase2_all_features_full.parquet"
DEFAULT_BLOCK2 = PROJECT / "block 2/c3_final_feature_set.csv"
DEFAULT_BLOCK1_PREDICTIONS = PROJECT / "nifty50individual/predictions_v5"
MARKET_FEATURES = [
    "nifty_daily_return", "nifty_5d_return", "nifty_vol_20d", "vix_level",
    "vix_change", "bank_return", "pharma_return",
]
TUNE_END = pd.Timestamp("2023-07-31")
EARLY_START = pd.Timestamp("2023-08-01")
TEST_START = pd.Timestamp("2024-01-01")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--block2-selection", type=Path, default=DEFAULT_BLOCK2)
    parser.add_argument("--embedding-dir", type=Path, default=HERE / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--block1-predictions", type=Path, default=DEFAULT_BLOCK1_PREDICTIONS)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--tune-epochs", type=int, default=10)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def strict_json(value):
    if isinstance(value, dict):
        return {str(k): strict_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp,)):
        return str(value)
    return value


def markdown_table(frame):
    display = frame.copy().fillna("—")
    lines = ["| " + " | ".join(display.columns) + " |",
             "| " + " | ".join(["---"] * len(display.columns)) + " |"]
    for row in display.itertuples(index=False, name=None):
        formatted = []
        for value in row:
            if isinstance(value, (float, np.floating)):
                formatted.append(f"{value:.6f}")
            else:
                formatted.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(formatted) + " |")
    return "\n".join(lines)


def write_json(path, value):
    path.write_text(json.dumps(strict_json(value), indent=2, allow_nan=False) + "\n")


def prepare_data(args):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = pd.read_csv(args.block2_selection)
    stock_features = selected.loc[selected.Category != "Market-Level", "Feature"].tolist()
    selected_market = selected.loc[selected.Category == "Market-Level", "Feature"].tolist()
    panel = pd.read_parquet(args.panel)
    panel.date = pd.to_datetime(panel.date).dt.tz_localize(None)
    panel = panel.sort_values(["date", "symbol"]).drop_duplicates(["date", "symbol"]).reset_index(drop=True)
    missing = [feature for feature in stock_features + [m for m in MARKET_FEATURES if m != "vix_change"] if feature not in panel]
    if missing:
        raise ValueError(f"Feature panel is missing: {missing}")

    daily_vix = panel.groupby("date", sort=True).vix_level.first().sort_index()
    vix_change = daily_vix.pct_change(fill_method=None).rename("vix_change")
    panel = panel.merge(vix_change, left_on="date", right_index=True, how="left")

    # Cross-sectional normalization uses only stocks on that date.
    stock_groups = panel.groupby("date", sort=False)[stock_features]
    stock_mean = stock_groups.transform("mean")
    stock_std = stock_groups.transform("std", ddof=0).replace(0.0, np.nan)
    normalized_stock = (panel[stock_features] - stock_mean) / stock_std
    normalized_stock = normalized_stock.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    normalized_stock.columns = [f"stock__{c}" for c in stock_features]

    # Rolling normalization is computed on one market row per day and never uses future days.
    market_daily = panel.groupby("date", sort=True)[MARKET_FEATURES].first()
    rolling_mean = market_daily.rolling(60, min_periods=2).mean()
    rolling_std = market_daily.rolling(60, min_periods=2).std(ddof=0).replace(0.0, np.nan)
    normalized_market = ((market_daily - rolling_mean) / rolling_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    normalized_market.columns = [f"market__{c}" for c in MARKET_FEATURES]
    panel = pd.concat([panel[["date", "symbol", "target_return_1d"]], normalized_stock], axis=1)
    panel = panel.merge(normalized_market, left_on="date", right_index=True, how="left")

    embedding_index = pd.read_parquet(args.embedding_dir / "kronos_embedding_index.parquet")
    embedding_index.date = pd.to_datetime(embedding_index.date).dt.tz_localize(None)
    panel = panel.merge(embedding_index, on=["date", "symbol"], how="left", validate="one_to_one")
    if panel.embedding_row.isna().any():
        raise RuntimeError(f"Missing embeddings for {int(panel.embedding_row.isna().sum())} rows")
    panel.embedding_row = panel.embedding_row.astype(int)
    panel = panel[panel.target_return_1d.notna()].reset_index(drop=True)
    feature_columns = [c for c in panel if c.startswith("stock__") or c.startswith("market__")]
    panel[feature_columns] = panel[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    raw_embedding = np.load(args.embedding_dir / "kronos_embeddings_float16.npy", mmap_mode="r")
    normalized_path = args.embedding_dir / "kronos_embeddings_layernorm_float16.npy"
    normalized_embedding = np.lib.format.open_memmap(
        normalized_path, mode="w+", dtype=np.float16, shape=raw_embedding.shape
    )
    for start in range(0, len(raw_embedding), 2048):
        block = np.asarray(raw_embedding[start:start + 2048], dtype=np.float32)
        mean = block.mean(axis=1, keepdims=True)
        variance = block.var(axis=1, keepdims=True)
        normalized_embedding[start:start + len(block)] = ((block - mean) / np.sqrt(variance + 1e-5)).astype(np.float16)
    normalized_embedding.flush()
    if not np.isfinite(np.asarray(normalized_embedding)).all():
        raise RuntimeError("Layer-normalized embedding contains NaN or infinity")

    panel.to_parquet(args.output_dir / "prepared_panel.parquet", index=False)
    metadata = {
        "rows": len(panel), "stocks": int(panel.symbol.nunique()),
        "date_min": str(panel.date.min().date()), "date_max": str(panel.date.max().date()),
        "embedding_dim": int(raw_embedding.shape[1]), "stock_features": stock_features,
        "market_features": MARKET_FEATURES, "block2_selected_market_features": selected_market,
        "normalization": {
            "stock": "same-day cross-sectional z-score; zero-variance/missing -> 0",
            "market": "causal rolling 60-day z-score; early/zero-variance/missing -> 0",
            "embedding": "per-sample layer normalization across actual 832 dimensions",
        },
        "splits": {
            "train": "2022-01-01..2022-12-31",
            "optuna_validation": "2023-01-01..2023-07-31",
            "early_stopping": "2023-08-01..2023-12-31",
            "test": f"2024-01-01..{panel.date.max().date()}",
        },
        "finite_feature_values": bool(np.isfinite(panel[feature_columns].to_numpy()).all()),
        "target_definition": "precomputed next-day stock return; feature row is dated t",
    }
    write_json(args.output_dir / "data_validation.json", metadata)
    return panel, stock_features, MARKET_FEATURES, raw_embedding.shape[1], normalized_path


class TensorPanel:
    def __init__(self, panel, normalized_embedding_path, stock_features, market_features, device):
        embedding = np.load(normalized_embedding_path, mmap_mode="r")
        rows = panel.embedding_row.to_numpy(int)
        self.embedding = torch.from_numpy(np.asarray(embedding[rows], dtype=np.float32).copy()).to(device)
        self.stock = torch.from_numpy(
            panel[[f"stock__{c}" for c in stock_features]].to_numpy(np.float32).copy()
        ).to(device)
        self.market = torch.from_numpy(
            panel[[f"market__{c}" for c in market_features]].to_numpy(np.float32).copy()
        ).to(device)
        self.target = torch.from_numpy(panel.target_return_1d.to_numpy(np.float32).copy()).to(device)
        self.frame = panel[["date", "symbol", "target_return_1d"]].copy()
        self.date_groups = {
            date: torch.tensor(indices, dtype=torch.long, device=device)
            for date, indices in self.frame.groupby("date", sort=True).indices.items()
        }

    def indices_between(self, start, end):
        return [idx for date, idx in self.date_groups.items() if pd.Timestamp(start) <= date <= pd.Timestamp(end)]


def mean_daily_rank_ic(actual, predicted, dates):
    frame = pd.DataFrame({"date": dates, "actual": actual, "predicted": predicted})
    values = []
    for _, day in frame.groupby("date", sort=True):
        day = day.replace([np.inf, -np.inf], np.nan).dropna()
        if len(day) >= 10 and day.actual.nunique() > 1 and day.predicted.nunique() > 1:
            rho = spearmanr(day.actual, day.predicted).statistic
            if np.isfinite(rho):
                values.append(float(rho))
    required_days = math.ceil(0.90 * frame.date.nunique())
    if len(values) < required_days:
        return -1.0, np.nan
    standard_deviation = np.std(values, ddof=1) if len(values) > 1 else np.nan
    icir = np.mean(values) / standard_deviation if np.isfinite(standard_deviation) and standard_deviation > 0 else np.nan
    return float(np.mean(values)), float(icir)


@torch.inference_mode()
def predict_indices(model, data, groups, return_aux=False, film_mode="full"):
    model.eval()
    predictions, actuals, dates, symbols = [], [], [], []
    aux_store = []
    for idx in groups:
        if model.__class__.__name__ == "FiLMConditioning":
            output = model(data.embedding[idx], data.stock[idx], data.market[idx],
                           return_aux=return_aux, mode=film_mode)
        else:
            output = model(data.embedding[idx], data.stock[idx], data.market[idx], return_aux=return_aux)
        if return_aux:
            prediction, auxiliary = output
            aux_store.append({key: value.detach().cpu().numpy() for key, value in auxiliary.items()})
        else:
            prediction = output
        cpu_idx = idx.detach().cpu().numpy()
        predictions.append(prediction.detach().cpu().numpy())
        actuals.append(data.target[idx].detach().cpu().numpy())
        dates.extend(data.frame.iloc[cpu_idx].date.tolist())
        symbols.extend(data.frame.iloc[cpu_idx].symbol.tolist())
    return pd.DataFrame({"date": dates, "symbol": symbols,
                         "actual_return": np.concatenate(actuals),
                         "predicted_return": np.concatenate(predictions)}), aux_store


def validation_score(model, data, groups):
    prediction, _ = predict_indices(model, data, groups)
    return mean_daily_rank_ic(prediction.actual_return, prediction.predicted_return, prediction.date)[0]


def train_model(name, params, dimensions, data, train_groups, validation_groups,
                max_epochs, patience, seed, trial=None):
    seed_everything(seed)
    model = build_model(name, dimensions, params).to(data.embedding.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    best_score, best_epoch, wait = -np.inf, 0, 0
    best_state = copy.deepcopy(model.state_dict())
    history = []
    generator = np.random.default_rng(seed)
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = generator.permutation(len(train_groups))
        losses = []
        for group_number in order:
            idx = train_groups[group_number]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(data.embedding[idx], data.stock[idx], data.market[idx])
            loss = torch.mean((prediction - data.target[idx]) ** 2)
            if not torch.isfinite(loss):
                return model, -1.0, epoch, pd.DataFrame(history)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = validation_score(model, data, validation_groups)
        history.append({"epoch": epoch, "train_mse": float(np.mean(losses)), "validation_rank_ic": score})
        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if score > best_score + 1e-6:
            best_score, best_epoch, wait = score, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_score, best_epoch, pd.DataFrame(history)


def suggest_params(trial, name):
    common = {
        "dropout": trial.suggest_float("dropout", 0.1, 0.4 if name != "attention" else 0.3),
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
    }
    if name == "concat":
        common.update(hidden1=trial.suggest_categorical("hidden1", [128, 256, 512]),
                      hidden2=trial.suggest_categorical("hidden2", [32, 64, 128]))
    elif name == "attention":
        common.update(d_model=trial.suggest_categorical("d_model", [64, 128, 256]),
                      n_heads=trial.suggest_categorical("n_heads", [2, 4, 8]),
                      head_hidden=trial.suggest_categorical("head_hidden", [16, 32, 64]))
    else:
        common.update(film_hidden=trial.suggest_categorical("film_hidden", [32, 64, 128, 256]),
                      hidden1=trial.suggest_categorical("hidden1", [128, 256, 512]),
                      hidden2=trial.suggest_categorical("hidden2", [32, 64, 128]))
    return common


def evaluate_predictions(predictions, model_name, output_dir):
    predictions = predictions.sort_values(["date", "predicted_return"], ascending=[True, False]).copy()
    daily_rows, quantile_rows = [], []
    previous_top, previous_bottom = None, None
    for date, day in predictions.groupby("date", sort=True):
        day = day.sort_values("predicted_return", ascending=False).copy()
        if len(day) < 20 or day.actual_return.nunique() < 2 or day.predicted_return.nunique() < 2:
            continue
        rank_ic = float(spearmanr(day.predicted_return, day.actual_return).statistic)
        top = day.head(10)
        bottom = day.tail(10)
        top_set, bottom_set = set(top.symbol), set(bottom.symbol)
        turnover = np.nan if previous_top is None else (len(top_set - previous_top) + len(bottom_set - previous_bottom)) / 20.0
        previous_top, previous_bottom = top_set, bottom_set
        ls_return = float(top.actual_return.mean() - bottom.actual_return.mean())
        daily_rows.append({"date": date, "rank_ic": rank_ic, "long_short_return": ls_return,
                           "top_quintile_return": float(top.actual_return.mean()), "turnover": turnover})
        day["quintile"] = pd.qcut(day.predicted_return.rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
        for quintile, group in day.groupby("quintile", observed=True):
            quantile_rows.append({"date": date, "quintile": int(quintile),
                                  "actual_return": float(group.actual_return.mean())})
    daily = pd.DataFrame(daily_rows)
    quantile = pd.DataFrame(quantile_rows)
    daily["cumulative_ic"] = daily.rank_ic.cumsum()
    daily["equity"] = (1.0 + daily.long_short_return).cumprod()
    daily["drawdown"] = daily.equity / daily.equity.cummax() - 1.0
    ls_std = daily.long_short_return.std(ddof=1)
    downside = daily.loc[daily.long_short_return < 0, "long_short_return"].std(ddof=1)
    annual_return = float(daily.equity.iloc[-1] ** (252 / len(daily)) - 1) if len(daily) else np.nan
    max_drawdown = float(daily.drawdown.min()) if len(daily) else np.nan
    metrics = {
        "Model": model_name,
        "RankIC": float(daily.rank_ic.mean()),
        "ICIR": float(daily.rank_ic.mean() / daily.rank_ic.std(ddof=1)),
        "RMSE": float(np.sqrt(np.mean((predictions.predicted_return - predictions.actual_return) ** 2))),
        "MAE": float(np.mean(np.abs(predictions.predicted_return - predictions.actual_return))),
        "Dir Acc": float(np.mean(np.sign(predictions.predicted_return) == np.sign(predictions.actual_return))),
        "L/S Spread": float(daily.long_short_return.mean()),
        "Top Hit Rate": float(np.mean(daily.top_quintile_return > 0)),
        "Sharpe": float(daily.long_short_return.mean() / ls_std * np.sqrt(252)) if ls_std else np.nan,
        "Sortino": float(daily.long_short_return.mean() / downside * np.sqrt(252)) if downside else np.nan,
        "MaxDD": max_drawdown,
        "Ann Ret": annual_return,
        "Calmar": annual_return / abs(max_drawdown) if max_drawdown else np.nan,
        "Win Rate": float(np.mean(daily.long_short_return > 0)),
        "Turnover": float(daily.turnover.mean()),
        "Test Days": int(len(daily)),
        "Stock Days": int(len(predictions)),
    }
    prefix = model_name.lower().replace(" ", "_").replace("+", "plus").replace(":", "")
    predictions.to_parquet(output_dir / f"{prefix}_test_predictions.parquet", index=False)
    daily.to_csv(output_dir / f"{prefix}_daily_metrics.csv", index=False)
    quantile.groupby("quintile", observed=True).actual_return.mean().reset_index().to_csv(
        output_dir / f"{prefix}_quantile_returns.csv", index=False)
    daily.assign(month=daily.date.dt.to_period("M").astype(str)).groupby("month").rank_ic.mean().reset_index().to_csv(
        output_dir / f"{prefix}_monthly_ic.csv", index=False)
    predictions.assign(correct=np.sign(predictions.predicted_return) == np.sign(predictions.actual_return)).groupby("symbol").correct.mean().reset_index(name="direction_accuracy").to_csv(
        output_dir / f"{prefix}_per_stock_direction_accuracy.csv", index=False)
    return metrics, daily


def ridge_baseline(panel, stock_features, selected_market, output_dir):
    columns = [f"stock__{c}" for c in stock_features] + [f"market__{c}" for c in selected_market]
    train = panel.date.dt.year == 2022
    tune = (panel.date.dt.year == 2023) & (panel.date <= TUNE_END)
    fit = panel.date < pd.Timestamp("2024-01-01")
    test = panel.date >= TEST_START
    best_alpha, best_score = None, -np.inf
    for alpha in np.logspace(-4, 4, 17):
        model = Ridge(alpha=alpha).fit(panel.loc[train, columns], panel.loc[train, "target_return_1d"])
        prediction = model.predict(panel.loc[tune, columns])
        score, _ = mean_daily_rank_ic(panel.loc[tune, "target_return_1d"], prediction, panel.loc[tune, "date"])
        if score > best_score:
            best_alpha, best_score = float(alpha), score
    model = Ridge(alpha=best_alpha).fit(panel.loc[fit, columns], panel.loc[fit, "target_return_1d"])
    predictions = panel.loc[test, ["date", "symbol", "target_return_1d"]].rename(columns={"target_return_1d": "actual_return"})
    predictions["predicted_return"] = model.predict(panel.loc[test, columns])
    metrics, _ = evaluate_predictions(predictions, "Block 2 Features Ridge", output_dir)
    write_json(output_dir / "block2_ridge_configuration.json",
               {"alpha": best_alpha, "optuna_period_rank_ic": best_score, "features": columns})
    return metrics


def block1_baseline(prediction_dir, output_dir):
    aliases = {"BAJAJ-AUTO": "BAJAJ_AUTO", "MM": "M&M"}
    rows = []
    for path in sorted(prediction_dir.glob("*_pred.parquet")):
        symbol = aliases.get(path.stem.replace("_pred", ""), path.stem.replace("_pred", ""))
        frame = pd.read_parquet(path).sort_values("date")
        frame.date = pd.to_datetime(frame.date).dt.tz_localize(None)
        previous = frame.actual_close.shift(1)
        part = pd.DataFrame({"date": frame.date, "symbol": symbol,
                             "actual_return": frame.actual_close / previous - 1.0,
                             "predicted_return": frame.pred_close / previous - 1.0})
        rows.append(part)
    predictions = pd.concat(rows, ignore_index=True).dropna()
    predictions = predictions[predictions.date >= TEST_START]
    return evaluate_predictions(predictions, "Block 1 Kronos LoRA Only", output_dir)[0]


def market_benchmark(panel):
    daily = panel[panel.date >= TEST_START].groupby("date").nifty_daily_return.first().dropna()
    equity = (1 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1
    annual = float(equity.iloc[-1] ** (252 / len(daily)) - 1)
    downside = daily[daily < 0].std(ddof=1)
    return {"Model": "Buy and Hold NIFTY 50", "RankIC": np.nan, "ICIR": np.nan,
            "Dir Acc": np.nan, "L/S Spread": np.nan,
            "Sharpe": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)),
            "Sortino": float(daily.mean() / downside * np.sqrt(252)),
            "MaxDD": float(drawdown.min()), "Ann Ret": annual,
            "Turnover": 0.0, "Test Days": len(daily), "Stock Days": np.nan}


def save_interpretability(name, model, data, test_groups, market_features, output_dir):
    _, auxiliaries = predict_indices(model, data, test_groups, return_aux=True)
    if name == "attention":
        attention = np.concatenate([item["attention"] for item in auxiliaries], axis=0).mean(axis=1)
        pd.DataFrame({"market_feature": market_features, "mean_attention": attention.mean(axis=0),
                      "std_attention": attention.std(axis=0)}).to_csv(output_dir / "attention_market_weights.csv", index=False)
        gates = np.concatenate([item["gate"] for item in auxiliaries], axis=0)
        write_json(output_dir / "gate_analysis.json", {
            "mean": gates.mean(), "std": gates.std(), "minimum": gates.min(), "maximum": gates.max(),
            "fraction_below_0.1": np.mean(gates < 0.1), "fraction_above_0.9": np.mean(gates > 0.9),
        })
    elif name == "film":
        gamma = np.concatenate([item["gamma"] for item in auxiliaries], axis=0)
        beta = np.concatenate([item["beta"] for item in auxiliaries], axis=0)
        write_json(output_dir / "film_parameter_analysis.json", {
            "gamma_mean": gamma.mean(), "gamma_std": gamma.std(), "gamma_mean_absolute": np.abs(gamma).mean(),
            "beta_mean": beta.mean(), "beta_std": beta.std(), "beta_mean_absolute": np.abs(beta).mean(),
        })
        rows = []
        for mode in ["full", "gamma_only", "beta_only", "none"]:
            prediction, _ = predict_indices(model, data, test_groups, film_mode=mode)
            rank_ic, icir = mean_daily_rank_ic(prediction.actual_return, prediction.predicted_return, prediction.date)
            rows.append({"mode": mode, "RankIC": rank_ic, "ICIR": icir,
                         "RMSE": np.sqrt(np.mean((prediction.predicted_return - prediction.actual_return) ** 2))})
        pd.DataFrame(rows).to_csv(output_dir / "film_ablation.csv", index=False)


def make_plots(daily_by_model, comparison, output_dir):
    cumulative = go.Figure()
    for name, daily in daily_by_model.items():
        cumulative.add_trace(go.Scatter(x=daily.date, y=daily.cumulative_ic, mode="lines", name=name))
    cumulative.update_layout(title="Block 3 Test Cumulative RankIC", xaxis_title="Date",
                             yaxis_title="Cumulative daily RankIC", template="plotly_white")
    cumulative.write_html(output_dir / "cumulative_ic.html", include_plotlyjs="cdn")
    chart = comparison[comparison.Model.str.contains("Block 3")].copy()
    architecture = go.Figure(go.Bar(
        x=chart.Model.str.replace("Block 3 ", "", regex=False), y=chart.RankIC
    ))
    architecture.update_layout(title="Architecture Test RankIC", yaxis_title="RankIC", template="plotly_white")
    architecture.write_html(output_dir / "architecture_rankic_comparison.html", include_plotlyjs="cdn")


def main():
    args = arguments()
    seed_everything(args.seed)
    panel, stock_features, market_features, embedding_dim, normalized_path = prepare_data(args)
    dimensions = {"embedding": embedding_dim, "stock": len(stock_features), "market": len(market_features)}
    data = TensorPanel(panel, normalized_path, stock_features, market_features, args.device)
    train_2022 = data.indices_between("2022-01-01", "2022-12-31")
    tune_validation = data.indices_between("2023-01-01", TUNE_END)
    final_train = data.indices_between("2022-01-01", TUNE_END)
    early_validation = data.indices_between(EARLY_START, "2023-12-31")
    test_groups = data.indices_between(TEST_START, "2100-01-01")
    split_counts = {"train_2022_days": len(train_2022), "optuna_validation_days": len(tune_validation),
                    "final_train_days": len(final_train), "early_stopping_days": len(early_validation),
                    "test_days": len(test_groups)}
    write_json(args.output_dir / "split_counts.json", split_counts)

    locked_models, locked_records = {}, []
    for architecture_number, name in enumerate(["concat", "attention", "film"]):
        print(f"\n=== OPTUNA: {name} ({args.trials} trials) ===", flush=True)
        sampler = optuna.samplers.TPESampler(seed=args.seed + architecture_number)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3)
        study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

        def objective(trial):
            params = suggest_params(trial, name)
            _, score, _, _ = train_model(name, params, dimensions, data, train_2022,
                                          tune_validation, args.tune_epochs, 4,
                                          args.seed + trial.number, trial)
            return score

        study.optimize(objective, n_trials=args.trials, gc_after_trial=True, show_progress_bar=False)
        trials = study.trials_dataframe()
        trials.to_csv(args.output_dir / f"optuna_{name}_trials.csv", index=False)
        best_params = study.best_params
        write_json(args.output_dir / f"best_{name}_params.json",
                   {"best_value": study.best_value, "params": best_params})
        print(f"=== FINAL TRAIN: {name}, params={best_params} ===", flush=True)
        model, score, epoch, history = train_model(
            name, best_params, dimensions, data, final_train, early_validation,
            args.final_epochs, 15, args.seed + 100 + architecture_number,
        )
        history.to_csv(args.output_dir / f"final_{name}_training_history.csv", index=False)
        torch.save({"architecture": name, "dimensions": dimensions, "params": best_params,
                    "state_dict": model.state_dict(), "best_epoch": epoch,
                    "early_stopping_rank_ic": score}, args.output_dir / f"best_{name}_model.pt")
        locked_models[name] = model
        locked_records.append({"architecture": name, "best_epoch": epoch,
                               "early_stopping_rank_ic": score, **best_params})
    pd.DataFrame(locked_records).to_csv(args.output_dir / "locked_model_summary.csv", index=False)

    # Test is first touched only after all architectures and hyperparameters are locked.
    comparison_rows, daily_by_model = [], {}
    display_names = {"concat": "Block 3A Concat + MLP",
                     "attention": "Block 3B Cross-Attention Gating",
                     "film": "Block 3C FiLM Conditioning"}
    for name, model in locked_models.items():
        predictions, _ = predict_indices(model, data, test_groups)
        result, daily = evaluate_predictions(predictions, display_names[name], args.output_dir)
        comparison_rows.append(result)
        daily_by_model[display_names[name]] = daily
        save_interpretability(name, model, data, test_groups, market_features, args.output_dir)

    comparison_rows.insert(0, ridge_baseline(
        panel, stock_features,
        pd.read_csv(args.block2_selection).query("Category == 'Market-Level'").Feature.tolist(),
        args.output_dir,
    ))
    comparison_rows.insert(0, block1_baseline(args.block1_predictions, args.output_dir))
    comparison_rows.append({"Model": "TimesFM 3 Zero-Shot", "status": "Not available: no TimesFM predictions/checkpoint supplied"})
    comparison_rows.append(market_benchmark(pd.read_parquet(args.panel)))
    comparison = pd.DataFrame(comparison_rows)
    ordered = ["Model", "RankIC", "ICIR", "Dir Acc", "L/S Spread", "Sharpe", "Sortino",
               "MaxDD", "Ann Ret", "Turnover", "RMSE", "MAE", "Test Days", "Stock Days", "status"]
    comparison = comparison.reindex(columns=ordered)
    comparison.to_csv(args.output_dir / "final_comparison_table.csv", index=False)
    make_plots(daily_by_model, comparison, args.output_dir)

    best_block3 = comparison[comparison.Model.str.contains("Block 3")].sort_values("RankIC", ascending=False).iloc[0]
    report = ["# Block 3 Final Report", "",
              "All three architectures were tuned on January-July 2023, retrained on data through July 2023, and early-stopped on August-December 2023. Test data beginning January 2024 was evaluated only after all three models were locked.", "",
              f"The actual Block 1 checkpoint hidden size is **{embedding_dim}**, so genuine 832-dimensional final-layer states were used instead of truncating to the specification's assumed 768 dimensions.", "",
              "## Final comparison", "", markdown_table(comparison), "",
              f"Best Block 3 architecture by untouched-test RankIC: **{best_block3.Model}** ({best_block3.RankIC:.6f}).", "",
              "TimesFM is explicitly marked unavailable because no TimesFM checkpoint or prediction output was supplied; inventing that benchmark would invalidate the comparison.", "",
              "Detailed daily metrics, monthly IC, quantile returns, per-stock direction accuracy, attention/gate analysis, FiLM ablations, checkpoints, Optuna histories, and plots are stored in this results folder."]
    (args.output_dir / "FINAL_REPORT.md").write_text("\n".join(report) + "\n")
    print("\nBlock 3 completed. Results:", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
