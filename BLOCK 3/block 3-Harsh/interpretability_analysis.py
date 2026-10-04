#!/usr/bin/env python3
"""Generate detailed test-period attention, gate, and FiLM regime analyses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from block3_pipeline import TensorPanel, predict_indices, strict_json
from models import build_model


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
RESULTS = HERE / "results"


def load_model(name, data):
    checkpoint = torch.load(RESULTS / f"best_{name}_model.pt", map_location=data.embedding.device, weights_only=False)
    model = build_model(name, checkpoint["dimensions"], checkpoint["params"]).to(data.embedding.device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def main():
    metadata = json.loads((RESULTS / "data_validation.json").read_text())
    panel = pd.read_parquet(RESULTS / "prepared_panel.parquet")
    panel.date = pd.to_datetime(panel.date)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = TensorPanel(
        panel, ARTIFACTS / "kronos_embeddings_layernorm_float16.npy",
        metadata["stock_features"], metadata["market_features"], device,
    )
    test_groups = data.indices_between("2024-01-01", "2100-01-01")
    test_dates = [date for date in data.date_groups if date >= pd.Timestamp("2024-01-01")]

    attention_model = load_model("attention", data)
    _, attention_aux = predict_indices(attention_model, data, test_groups, return_aux=True)
    attention_rows, gate_rows = [], []
    for date, auxiliary in zip(test_dates, attention_aux):
        # attention shape: stocks x heads x market tokens
        token_weight = auxiliary["attention"].mean(axis=(0, 1))
        attention_rows.append({"date": date, **dict(zip(metadata["market_features"], token_weight))})
        gate_rows.append({"date": date, "gate_mean": auxiliary["gate"].mean(),
                          "gate_std": auxiliary["gate"].std(),
                          "gate_min": auxiliary["gate"].min(), "gate_max": auxiliary["gate"].max()})
    attention_daily = pd.DataFrame(attention_rows)
    gate_daily = pd.DataFrame(gate_rows)
    attention_daily.to_csv(RESULTS / "attention_by_test_day.csv", index=False)
    gate_daily.to_csv(RESULTS / "gate_by_test_day.csv", index=False)

    sample_positions = np.unique(np.linspace(0, len(attention_daily) - 1, 16, dtype=int))
    sample = attention_daily.iloc[sample_positions]
    heatmap = go.Figure(go.Heatmap(
        z=sample[metadata["market_features"]].to_numpy().T,
        x=sample.date.dt.strftime("%Y-%m-%d"), y=metadata["market_features"],
        colorscale="Viridis", colorbar_title="Attention",
    ))
    heatmap.update_layout(title="Cross-Attention Weights on Sample Test Days",
                          xaxis_title="Test date", yaxis_title="Market token", template="plotly_white")
    heatmap.write_html(RESULTS / "attention_sample_days.html", include_plotlyjs="cdn")

    film_model = load_model("film", data)
    _, film_aux = predict_indices(film_model, data, test_groups, return_aux=True)
    gamma = np.concatenate([item["gamma"] for item in film_aux], axis=0)
    beta = np.concatenate([item["beta"] for item in film_aux], axis=0)
    market_rows = np.concatenate([idx.detach().cpu().numpy() for idx in test_groups])
    vix_column = metadata["market_features"].index("vix_level")
    vix = data.market[torch.tensor(market_rows, device=data.market.device), vix_column].detach().cpu().numpy()
    low_cut, high_cut = np.quantile(vix, [0.25, 0.75])
    low, high = vix <= low_cut, vix >= high_cut
    dimensions = np.arange(gamma.shape[1])
    regime = pd.DataFrame({
        "embedding_dimension": dimensions,
        "gamma_low_vix": gamma[low].mean(axis=0), "gamma_high_vix": gamma[high].mean(axis=0),
        "gamma_high_minus_low": gamma[high].mean(axis=0) - gamma[low].mean(axis=0),
        "beta_low_vix": beta[low].mean(axis=0), "beta_high_vix": beta[high].mean(axis=0),
        "beta_high_minus_low": beta[high].mean(axis=0) - beta[low].mean(axis=0),
    })
    regime.to_csv(RESULTS / "film_high_low_vix_by_dimension.csv", index=False)
    summary = {
        "vix_normalized_low_quartile_cutoff": low_cut,
        "vix_normalized_high_quartile_cutoff": high_cut,
        "low_vix_stock_days": int(low.sum()), "high_vix_stock_days": int(high.sum()),
        "mean_absolute_gamma_change_high_vs_low": np.abs(regime.gamma_high_minus_low).mean(),
        "mean_absolute_beta_change_high_vs_low": np.abs(regime.beta_high_minus_low).mean(),
        "gamma_correlation_high_vs_low": np.corrcoef(regime.gamma_low_vix, regime.gamma_high_vix)[0, 1],
        "beta_correlation_high_vs_low": np.corrcoef(regime.beta_low_vix, regime.beta_high_vix)[0, 1],
    }
    (RESULTS / "film_vix_regime_summary.json").write_text(
        json.dumps(strict_json(summary), indent=2, allow_nan=False) + "\n"
    )
    modulation = go.Figure()
    modulation.add_trace(go.Scatter(x=dimensions, y=regime.gamma_high_minus_low,
                                    mode="lines", name="Gamma: high minus low VIX"))
    modulation.add_trace(go.Scatter(x=dimensions, y=regime.beta_high_minus_low,
                                    mode="lines", name="Beta: high minus low VIX"))
    modulation.update_layout(title="FiLM Parameter Change Across VIX Regimes",
                             xaxis_title="Kronos embedding dimension", yaxis_title="High - low VIX",
                             template="plotly_white")
    modulation.write_html(RESULTS / "film_vix_regime_modulation.html", include_plotlyjs="cdn")
    print("Detailed interpretability outputs written to", RESULTS)


if __name__ == "__main__":
    main()
