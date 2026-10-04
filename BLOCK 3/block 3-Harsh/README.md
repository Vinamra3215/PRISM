# Block 3 - Kronos + Validated Features + Market Gating

This folder implements and compares:

1. Concat + MLP
2. Cross-Attention Gating (MASTER-style)
3. FiLM Conditioning

The pipeline uses 2022 for initial training, January-July 2023 for Optuna,
August-December 2023 for final early stopping, and January 2024 onward as the
one-time test. Training batches are grouped by trading date, preserving the
cross-section of stocks.

The Block 1 checkpoint has a real hidden size of 832. The extractor therefore
stores genuine 832-dimensional final-transformer-layer states rather than
forcing the document's assumed 768 dimensions.

Run:

```bash
../.venv/bin/python run_all.py --trials 50
```

All generated artifacts and results stay under this `block 3` directory.

`interpretability_analysis.py` can be rerun after training to regenerate the
per-day attention/gate outputs and high-VIX versus low-VIX FiLM analysis.

Key outputs:

- `artifacts/`: 57,545 genuine Kronos embeddings, their index, normalized copy,
  and extraction metadata.
- `results/FINAL_REPORT.md`: final findings and comparison table.
- `results/final_comparison_table.csv`: Block 1, Block 2, all three Block 3
  architectures, TimesFM availability, and NIFTY benchmark.
- `results/best_*_model.pt`: locked architecture checkpoints.
- `results/optuna_*_trials.csv`: all 50 trials per architecture.
- `results/*_test_predictions.parquet`: one-time test predictions.
- `results/cumulative_ic.html` and interpretability HTML/CSV files: cumulative
  IC, attention, gating, and FiLM regime analysis.
