# Block 3 Final Report

All three architectures were tuned on January-July 2023, retrained on data through July 2023, and early-stopped on August-December 2023. Test data beginning January 2024 was evaluated only after all three models were locked.

The actual Block 1 checkpoint hidden size is **832**, so genuine 832-dimensional final-layer states were used instead of truncating to the specification's assumed 768 dimensions.

## Final comparison

| Model | RankIC | ICIR | Dir Acc | L/S Spread | Sharpe | Sortino | MaxDD | Ann Ret | Turnover | RMSE | MAE | Test Days | Stock Days | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Block 1 Kronos LoRA Only | -0.001041 | -0.005980 | 0.500017 | -0.000153 | -0.279613 | -0.476975 | -0.229904 | -0.047003 | 0.623051 | 0.026039 | 0.018692 | 591.000000 | 28773.000000 | — |
| Block 2 Features Ridge | 0.000289 | 0.001617 | 0.491834 | 0.000223 | 0.417062 | 0.618375 | -0.171562 | 0.048325 | 0.434493 | 0.017044 | 0.011833 | 662.000000 | 33249.000000 | — |
| Block 3A Concat + MLP | -0.009594 | -0.057358 | 0.496346 | -0.000444 | -0.956379 | -1.399496 | -0.307067 | -0.111894 | 0.442965 | 0.017133 | 0.011995 | 662.000000 | 33249.000000 | — |
| Block 3B Cross-Attention Gating | -0.002509 | -0.014862 | 0.496647 | 0.000017 | 0.031821 | 0.045074 | -0.243793 | -0.004880 | 0.471558 | 0.017438 | 0.012024 | 662.000000 | 33249.000000 | — |
| Block 3C FiLM Conditioning | 0.002558 | 0.015089 | 0.496256 | 0.000238 | 0.517687 | 0.828930 | -0.110552 | 0.054631 | 0.594402 | 0.017005 | 0.011786 | 662.000000 | 33249.000000 | — |
| TimesFM 3 Zero-Shot | — | — | — | — | — | — | — | — | — | — | — | — | — | Not available: no TimesFM predictions/checkpoint supplied |
| Buy and Hold NIFTY 50 | — | — | — | — | 0.335174 | 0.456525 | -0.157667 | 0.036834 | 0.000000 | — | — | 662.000000 | — | — |

Best Block 3 architecture by untouched-test RankIC: **Block 3C FiLM Conditioning** (0.002558).

TimesFM is explicitly marked unavailable because no TimesFM checkpoint or prediction output was supplied; inventing that benchmark would invalidate the comparison.

The specification mentions a historical Block 1 RankIC of `0.008`. Recomputing the metric from the available per-stock Block 1 prediction files over their common January 2024-June 2026 coverage gives `-0.00104`; the comparison table reports this reproducible value rather than copying the historical headline number.

Detailed daily metrics, monthly IC, quantile returns, per-stock direction accuracy, attention/gate analysis, FiLM ablations, checkpoints, Optuna histories, and plots are stored in this results folder.

## Interpretability findings

- The cross-attention gate is dynamic rather than saturated: mean `0.5560`, standard deviation `0.0824`, with no values below `0.1` or above `0.9`.
- Mean attention is close to uniform across the seven market tokens (`0.1405` to `0.1460`). The largest average weight is on `nifty_vol_20d`, but the narrow range indicates weak token differentiation.
- FiLM changes materially across VIX regimes. The mean absolute high-minus-low-VIX change is `0.1945` for gamma and `0.1774` for beta across the 832 Kronos dimensions.
- The post-training FiLM ablation gives RankIC `0.00718` for gamma-only, `-0.00536` for beta-only, and `-0.00965` without modulation. Gamma carries the useful modulation signal in this checkpoint; beta weakens it.

See `attention_by_test_day.csv`, `attention_sample_days.html`, `gate_by_test_day.csv`, `film_high_low_vix_by_dimension.csv`, and `film_vix_regime_modulation.html` for the detailed analyses.
