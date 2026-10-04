# Block 2 - C2/C3 Feature Selection

`final15.py` trains a LightGBM feature-ranking model on 2022 rows, validates
it on 2023 rows, and writes the complete C2 gain ranking and balanced C3 final
feature table. It never performs a random split.

Run from this directory:

```bash
python3 -m pip install -r requirements.txt
python3 final15.py
```

The default input is the existing merged eight-category panel at
`../../tanishq/phase2_data/nifty50_phase2_calibration_2022_2023.parquet`.
Use `--data /path/to/panel.parquet` to override it.

Outputs:

- `c2_lightgbm_feature_importance.csv`: all 43 candidates ranked by gain.
- `c2_validation_metrics.json`: time-split boundaries and validation metrics.
- `c3_final_feature_set.csv`: machine-readable final 19 features.
- `c3_final_feature_set.md`: requested final table and selection notes.
