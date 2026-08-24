# ~/Kronos/het/compare_predictions.py
#
# Forecasts the 80-bar validation window using the 320-bar training window
# as context, once with the original pretrained model and once with the
# LoRA-adapted model, and compares both against the actual validation prices.
#
# This does NOT touch the future (post-2021) test period -- it's purely a
# sanity check that LoRA fine-tuning improved forecasting on held-out
# historical data before we move on to predicting the future period.

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model import Kronos, KronosTokenizer, KronosPredictor

# ---------------------------------------------------------------------------
# Config -- must match train_lora.py
# ---------------------------------------------------------------------------
TOKENIZER_PATH = "/home/soq/__shutupandbendover/het-uchiha/weights/Kronos-Tokenizer-base"
MODEL_PATH     = "/home/soq/__shutupandbendover/het-uchiha/weights/Kronos-base"

DATA_PATH   = os.path.join(os.path.dirname(__file__), "data", "RIL_19-26.parquet")
CUTOFF_DATE = "2021-01-01"
WINDOW_SIZE = 400
TRAIN_SIZE  = 320
VAL_SIZE    = 80

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "lora_best.pt")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

PRICE_COLS = ["open", "high", "low", "close"]
VOL_COL    = "volume"
AMT_COL    = "amount"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# LoRA layer -- identical to train_lora.py, needed to reconstruct the
# adapted architecture before loading the saved checkpoint into it.
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False
        in_features = base_linear.in_features
        out_features = base_linear.out_features
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        base_out = self.base(x)
        lora_out = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return base_out + self.scaling * lora_out


def inject_lora(model, r, alpha, dropout, target_names):
    for p in model.parameters():
        p.requires_grad = False
    for block in model.transformer:
        attn = block.self_attn
        for name in target_names:
            orig = getattr(attn, name)
            setattr(attn, name, LoRALinear(orig, r=r, alpha=alpha, dropout=dropout))


# ---------------------------------------------------------------------------
# Data prep -- same windowing as train_lora.py
# ---------------------------------------------------------------------------
def load_ohlcv(path):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    if AMT_COL not in df.columns:
        df[AMT_COL] = df[VOL_COL] * df[PRICE_COLS].mean(axis=1)
    return df


def get_train_val_windows(df):
    before = df[df.index < CUTOFF_DATE]
    window = before.tail(WINDOW_SIZE)
    train = window.iloc[:TRAIN_SIZE]
    val = window.iloc[TRAIN_SIZE:]
    return train, val


def run_forecast(predictor, train_df, val_df, pred_len):
    x_df = train_df[PRICE_COLS + [VOL_COL]]
    x_timestamp = pd.Series(pd.to_datetime(train_df.index)).reset_index(drop=True)
    y_timestamp = pd.Series(pd.to_datetime(val_df.index)).reset_index(drop=True)
    pred_df = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=1.0,
        top_p=0.9,
        sample_count=5,   # average a few sampled paths for a steadier forecast
    )
    return pred_df


def score(pred_df, val_df):
    actual = val_df["close"].values
    predicted = pred_df["close"].values[:len(actual)]
    mae = np.mean(np.abs(predicted - actual))
    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mape = np.mean(np.abs((predicted - actual) / actual)) * 100
    return mae, rmse, mape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")

    df = load_ohlcv(DATA_PATH)
    train_df, val_df = get_train_val_windows(df)
    print(f"Train (context): {train_df.index.min()} -> {train_df.index.max()} ({len(train_df)} bars)")
    print(f"Val (target):    {val_df.index.min()} -> {val_df.index.max()} ({len(val_df)} bars)")

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    print(f"\nLoaded checkpoint from {CHECKPOINT_PATH}")
    print(f"  trained to step {ckpt['step']}, val loss {ckpt['val_loss']:.4f}")
    print(f"  r={ckpt['r']} alpha={ckpt['alpha']} targets={ckpt['target_names']}")

    # --- Pretrained (no LoRA) ---
    print("\nLoading pretrained model (no LoRA)...")
    tokenizer_base = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    model_base = Kronos.from_pretrained(MODEL_PATH)
    predictor_base = KronosPredictor(model_base, tokenizer_base, max_context=512)

    print("Forecasting with pretrained model...")
    pred_base = run_forecast(predictor_base, train_df, val_df, VAL_SIZE)
    mae_base, rmse_base, mape_base = score(pred_base, val_df)

    del model_base, tokenizer_base, predictor_base
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- LoRA-adapted ---
    print("\nLoading LoRA-adapted model...")
    tokenizer_lora = KronosTokenizer.from_pretrained(TOKENIZER_PATH)
    model_lora = Kronos.from_pretrained(MODEL_PATH)
    inject_lora(model_lora, ckpt["r"], ckpt["alpha"], ckpt["dropout"], ckpt["target_names"])
    missing, unexpected = model_lora.load_state_dict(ckpt["lora_state_dict"], strict=False)
    # missing here is expected: it's every frozen base param not in the checkpoint
    if unexpected:
        print(f"  WARNING unexpected keys in checkpoint: {unexpected}")
    predictor_lora = KronosPredictor(model_lora, tokenizer_lora, max_context=512)

    print("Forecasting with LoRA-adapted model...")
    pred_lora = run_forecast(predictor_lora, train_df, val_df, VAL_SIZE)
    mae_lora, rmse_lora, mape_lora = score(pred_lora, val_df)

    # --- Results ---
    print("\n" + "=" * 60)
    print(f"{'Metric':<10} {'Pretrained':>15} {'LoRA-adapted':>15}")
    print("-" * 60)
    print(f"{'MAE':<10} {mae_base:>15.4f} {mae_lora:>15.4f}")
    print(f"{'RMSE':<10} {rmse_base:>15.4f} {rmse_lora:>15.4f}")
    print(f"{'MAPE %':<10} {mape_base:>15.4f} {mape_lora:>15.4f}")
    print("=" * 60)
    improvement = (mae_base - mae_lora) / mae_base * 100
    print(f"\nMAE change from LoRA fine-tuning: {improvement:+.2f}% "
          f"({'improvement' if improvement > 0 else 'regression'})")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "val_comparison.parquet")
    comparison = pd.DataFrame({
        "actual_close": val_df["close"].values,
        "pretrained_pred": pred_base["close"].values[:len(val_df)],
        "lora_pred": pred_lora["close"].values[:len(val_df)],
    }, index=val_df.index)
    comparison.to_parquet(out_path)
    print(f"\nSaved comparison table to {out_path}")


if __name__ == "__main__":
    main()