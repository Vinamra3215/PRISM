import os
import sys
import time
import json
import webbrowser
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from peft import LoraConfig, get_peft_model
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

# ---------------------------------------------------------------------------
# Path Configuration
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
MY_KRONOS_PROJECT = os.path.join(PARENT_DIR, "my_kronos_project")
KRONOS_REPO_PATH = "/home/soq/Kronos"
NIFTY_DATA_PATH = os.path.join(CURRENT_DIR, "nifty50_2022_2026.parquet")
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for p in [CURRENT_DIR, MY_KRONOS_PROJECT, KRONOS_REPO_PATH]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from model.kronos import Kronos, KronosTokenizer, KronosPredictor
from metrics_engine import calculate_return_and_kline_metrics, save_and_print_metrics

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Ingest Optuna Tuned Hyperparameters (Direct Values from Completed Run)
# ---------------------------------------------------------------------------
ROLLING_TRAIN_WINDOW = 380
LORA_R = 16
LORA_ALPHA = 8
LORA_DROPOUT = 0.05
ADAPT_EPOCHS = 3
LEARNING_RATE = 8.33407096399951e-05
RECENCY_DECAY = 0.25394571349665224
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16

PARAMS_JSON_PATH = os.path.join(CURRENT_DIR, "best_hyperparameters.json")
if os.path.exists(PARAMS_JSON_PATH):
    try:
        with open(PARAMS_JSON_PATH, "r") as f:
            cfg = json.load(f).get("optimal_hyperparameters", {})
        ROLLING_TRAIN_WINDOW = int(cfg.get("rolling_window", ROLLING_TRAIN_WINDOW))
        LORA_R = int(cfg.get("lora_r", LORA_R))
        LORA_ALPHA = int(cfg.get("lora_alpha", LORA_ALPHA))
        ADAPT_EPOCHS = int(cfg.get("adapt_epochs", ADAPT_EPOCHS))
        LEARNING_RATE = float(cfg.get("lr", LEARNING_RATE))
        RECENCY_DECAY = float(cfg.get("recency_decay", RECENCY_DECAY))
    except Exception:
        pass

print("=" * 65)
print("✓ OPTUNA TUNED HYPERPARAMETERS INGESTED:")
print(f"  • Rolling Window : {ROLLING_TRAIN_WINDOW} days")
print(f"  • LoRA Rank (r)  : {LORA_R} | Alpha: {LORA_ALPHA}")
print(f"  • Learning Rate  : {LEARNING_RATE:.2e} | Adapt Epochs: {ADAPT_EPOCHS}")
print(f"  • Recency Decay  : {RECENCY_DECAY:.4f}")
print("=" * 65 + "\n")

# ---------------------------------------------------------------------------
# Robust Model & Tokenizer Loaders
# ---------------------------------------------------------------------------
def get_base_model():
    try:
        return Kronos.from_pretrained("NeoQuasar/Kronos-small")
    except Exception:
        try:
            return Kronos.from_pretrained("NeoQuasar/Kronos-base")
        except Exception:
            snap_path = "/home/soq/.cache/huggingface/hub/models--NeoQuasar--Kronos-small/snapshots"
            if os.path.exists(snap_path) and len(os.listdir(snap_path)) > 0:
                target = os.path.join(snap_path, os.listdir(snap_path)[0])
                return Kronos.from_pretrained(target)
            raise RuntimeError("Could not load Kronos base model.")

def get_tokenizer():
    try:
        config_file = hf_hub_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", filename="config.json")
        with open(config_file, "r") as f:
            cfg = json.load(f)
        for ignored in ["model_type", "architectures", "auto_map", "torch_dtype", "transformers_version"]:
            cfg.pop(ignored, None)
        
        tokenizer = KronosTokenizer(**cfg)
        
        try:
            weights_file = hf_hub_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", filename="model.safetensors")
            state_dict = load_safetensors(weights_file)
        except Exception:
            weights_file = hf_hub_download(repo_id="NeoQuasar/Kronos-Tokenizer-base", filename="pytorch_model.bin")
            state_dict = torch.load(weights_file, map_location="cpu")
            
        tokenizer.load_state_dict(state_dict)
        return tokenizer
    except Exception:
        snap_path = "/home/soq/.cache/huggingface/hub/models--NeoQuasar--Kronos-Tokenizer-base/snapshots"
        if os.path.exists(snap_path) and len(os.listdir(snap_path)) > 0:
            target = os.path.join(snap_path, os.listdir(snap_path)[0])
            with open(os.path.join(target, "config.json"), "r") as f:
                cfg = json.load(f)
            for ignored in ["model_type", "architectures", "auto_map", "torch_dtype", "transformers_version"]:
                cfg.pop(ignored, None)
            tokenizer = KronosTokenizer(**cfg)
            w_file = os.path.join(target, "model.safetensors") if os.path.exists(os.path.join(target, "model.safetensors")) else os.path.join(target, "pytorch_model.bin")
            tokenizer.load_state_dict(load_safetensors(w_file) if w_file.endswith(".safetensors") else torch.load(w_file, map_location="cpu"))
            return tokenizer
            
        raise RuntimeError("Could not find or load Kronos Tokenizer.")

# ---------------------------------------------------------------------------
# Data Preprocessing & Tokenizer Pipeline
# ---------------------------------------------------------------------------
def normalize_dates(series: pd.Series) -> pd.Series:
    dt_series = pd.to_datetime(series)
    if dt_series.dt.tz is not None:
        dt_series = dt_series.dt.tz_localize(None)
    return dt_series

def load_local_nifty_data() -> pd.DataFrame:
    df = pd.read_parquet(NIFTY_DATA_PATH)
    df.columns = df.columns.str.lower()
    df['date'] = normalize_dates(df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'amount' not in df.columns or df['amount'].isnull().all():
        df['amount'] = df['close'] * df['volume']
    else:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    return df.ffill().bfill().sort_values('date').reset_index(drop=True)

def tokenize_window(df_window: pd.DataFrame, tokenizer, patch_size: int = 16):
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    windows = []
    for i in range(len(df_window) - patch_size + 1):
        windows.append(df_window[feature_cols].iloc[i : i + patch_size].values)
    raw_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(DEVICE)
    
    all_s1, all_s2 = [], []
    with torch.no_grad():
        for b in range(0, len(raw_tensor), 32):
            slice_tensor = raw_tensor[b : b + 32]
            enc = tokenizer.encode(slice_tensor)
            if isinstance(enc, (tuple, list)):
                s1 = enc[0].detach().cpu().flatten()
                s2 = enc[1].detach().cpu().flatten()
            elif isinstance(enc, dict):
                s1 = enc['s1_ids'].detach().cpu().flatten()
                s2 = enc['s2_ids'].detach().cpu().flatten()
            elif isinstance(enc, torch.Tensor) and enc.shape[-1] >= 2:
                s1 = enc[..., 0].detach().cpu().flatten()
                s2 = enc[..., 1].detach().cpu().flatten()
            else:
                s1 = enc.detach().cpu().flatten()
                s2 = enc.detach().cpu().flatten()
            all_s1.append(s1)
            all_s2.append(s2)
    s1_all = torch.clamp(torch.cat(all_s1, dim=0), 0, 1023)
    s2_all = torch.clamp(torch.cat(all_s2, dim=0), 0, 1023)
    return s1_all, s2_all

class RollingPatchDataset(Dataset):
    def __init__(self, s1_seq: torch.Tensor, s2_seq: torch.Tensor, seq_len: int = 32):
        self.seq_len = seq_len
        self.s1_seq = torch.clamp(s1_seq.flatten().long(), 0, 1023)
        self.s2_seq = torch.clamp(s2_seq.flatten().long(), 0, 1023)

    def __len__(self) -> int:
        return max(1, len(self.s1_seq) - self.seq_len)

    def __getitem__(self, idx: int):
        idx_bounded = min(idx, max(0, len(self.s1_seq) - self.seq_len - 1))
        return {
            "s1_ids": self.s1_seq[idx_bounded : idx_bounded + self.seq_len],
            "s2_ids": self.s2_seq[idx_bounded : idx_bounded + self.seq_len],
            "s1_targets": self.s1_seq[idx_bounded + 1 : idx_bounded + self.seq_len + 1],
            "s2_targets": self.s2_seq[idx_bounded + 1 : idx_bounded + self.seq_len + 1]
        }

def build_lora_kline(base_model):
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=None
    )
    return get_peft_model(base_model, lora_config)

def adapt_lora_kline(model, dataloader, epochs=ADAPT_EPOCHS):
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    for _ in range(epochs):
        for batch in dataloader:
            s1_in = batch['s1_ids'].to(DEVICE)
            s2_in = batch['s2_ids'].to(DEVICE)
            s1_tgt = batch['s1_targets'].to(DEVICE)
            
            # Recency weighting using Optuna-tuned decay
            time_weights = torch.linspace(RECENCY_DECAY, 1.0, steps=s1_in.size(1), device=DEVICE)
            
            optimizer.zero_grad()
            outputs = model(s1_ids=s1_in, s2_ids=s2_in)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            raw_loss = criterion(logits.reshape(-1, logits.size(-1)), s1_tgt.reshape(-1)).view(s1_in.size(0), s1_in.size(1))
            loss = (raw_loss * time_weights).mean()
            loss.backward()
            optimizer.step()

# ---------------------------------------------------------------------------
# Main Walk-Forward Execution Engine (Mid-2023 to 2026)
# ---------------------------------------------------------------------------
def run_kline_pipeline():
    start_total_time = time.perf_counter()
    
    df_nifty = load_local_nifty_data()
    window_size = ROLLING_TRAIN_WINDOW
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    
    eval_start_date = pd.to_datetime("2023-07-01")
    eval_indices = df_nifty[df_nifty['date'] >= eval_start_date].index
    
    if len(eval_indices) == 0:
        raise ValueError(f"No dates found >= {eval_start_date.date()} in dataset!")
        
    start_eval_idx = eval_indices[0]
    total_candles = len(df_nifty)
    total_steps = total_candles - start_eval_idx

    print(f"--> Total Nifty 50 Candles Ingested: {total_candles}")
    print(f"--> Rolling Context Size: {window_size} days")
    print(f"--> Evaluation Horizon: {df_nifty['date'].iloc[start_eval_idx].date()} to {df_nifty['date'].iloc[-1].date()} ({total_steps} trading days)\n")
    
    tokenizer = get_tokenizer()
    if hasattr(tokenizer, 'to'):
        tokenizer = tokenizer.to(DEVICE)
    tokenizer.eval()

    base_model = get_base_model().to(DEVICE)

    pred_records = []
    for _ in range(start_eval_idx):
        pred_records.append({'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan})

    adapt_times = []
    infer_times = []
    step_times = []

    print("--> Starting Walk-Forward Loop for all trading days from mid-2023 to 2026...")

    for step in range(total_steps):
        predict_idx = start_eval_idx + step
        train_end = predict_idx
        train_start = max(0, train_end - window_size)
        
        target_date = df_nifty['date'].iloc[predict_idx]
        actual_close = df_nifty['close'].iloc[predict_idx]
        
        window_df = df_nifty.iloc[train_start:train_end].reset_index(drop=True)
        
        t_step_start = time.perf_counter()
        
        s1_stream, s2_stream = tokenize_window(window_df, tokenizer)
        train_ds = RollingPatchDataset(s1_stream, s2_stream, seq_len=32)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        # 1. Measure LoRA Adaptation Time
        t_adapt_start = time.perf_counter()
        model = build_lora_kline(base_model)
        adapt_lora_kline(model, train_loader, epochs=ADAPT_EPOCHS)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_adapt = time.perf_counter() - t_adapt_start
        adapt_times.append(t_adapt)
        
        # 2. Measure 1-Day Prediction Inference Time
        t_infer_start = time.perf_counter()
        model.eval()
        predictor = KronosPredictor(model, tokenizer, max_context=window_size)
        
        with torch.no_grad():
            pred_candle = predictor.predict(
                df=window_df[feature_cols],
                x_timestamp=pd.to_datetime(window_df['date']),
                y_timestamp=pd.to_datetime(df_nifty['date'].iloc[predict_idx:predict_idx+1]),
                pred_len=1
            )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_infer = time.perf_counter() - t_infer_start
        infer_times.append(t_infer)
            
        if isinstance(pred_candle, pd.DataFrame):
            p_o = float(pred_candle['open'].iloc[-1])
            p_h = float(pred_candle['high'].iloc[-1])
            p_l = float(pred_candle['low'].iloc[-1])
            p_c = float(pred_candle['close'].iloc[-1])
        else:
            p_o = float(pred_candle[-1, 0])
            p_h = float(pred_candle[-1, 1])
            p_l = float(pred_candle[-1, 2])
            p_c = float(pred_candle[-1, 3])
            
        # Candlestick structural constraints
        p_h = max(p_h, p_o, p_c)
        p_l = min(p_l, p_o, p_c)
        
        pred_records.append({'open': p_o, 'high': p_h, 'low': p_l, 'close': p_c})
        
        # Cleanly unload PEFT adapter so base model remains pristine
        if hasattr(model, 'unload'):
            base_model = model.unload()
        del model
        torch.cuda.empty_cache()

        t_step = time.perf_counter() - t_step_start
        step_times.append(t_step)

        if (step + 1) % 25 == 0 or step == total_steps - 1:
            print(f"Step [{step+1:03d}/{total_steps:03d}] Date: {target_date.date()} | "
                  f"Actual Close: {actual_close:.2f} | Pred Close: {p_c:.2f} | "
                  f"1-Day Infer: {t_infer*1000:.1f}ms | Step Time: {t_step:.2f}s")

    total_elapsed_sec = time.perf_counter() - start_total_time
    total_net_infer_sec = sum(infer_times)
    total_net_adapt_sec = sum(adapt_times)

    df_preds = pd.DataFrame(pred_records)
    
    # Save predictions CSV
    pred_export_df = df_nifty[['date', 'open', 'high', 'low', 'close']].copy()
    pred_export_df['pred_open'] = df_preds['open']
    pred_export_df['pred_high'] = df_preds['high']
    pred_export_df['pred_low'] = df_preds['low']
    pred_export_df['pred_close'] = df_preds['close']
    csv_out_path = os.path.join(OUTPUT_DIR, "nifty50_kline_predictions.csv")
    pred_export_df.to_csv(csv_out_path, index=False)
    print(f"\n✓ Saved OHLC predictions dataset to: {csv_out_path}")

    # Compute and persist risk report
    metrics = calculate_return_and_kline_metrics(df_nifty, df_preds)
    save_and_print_metrics(metrics, output_dir=OUTPUT_DIR, title="NIFTY 50 (MID 2023 - 2026) RISK REPORT")
    
    # Latency & runtime profiling report
    print(f"{'=' * 22} TOTAL TIME & LATENCY PROFILING REPORT {'=' * 22}")
    print(f"  • Horizon Evaluated                   : Mid-2023 to 2026 ({total_steps} trading days)")
    print(f"  • TOTAL NET PREDICTION INFERENCE TIME : {int(total_net_infer_sec // 60)}m {total_net_infer_sec % 60:.2f}s ({total_net_infer_sec:.2f}s)")
    print(f"  • TOTAL NET LORA ADAPTATION TIME      : {int(total_net_adapt_sec // 60)}m {total_net_adapt_sec % 60:.2f}s ({total_net_adapt_sec:.2f}s)")
    print(f"  • TOTAL END-TO-END PIPELINE TIME      : {int(total_elapsed_sec // 60)}m {total_elapsed_sec % 60:.2f}s ({total_elapsed_sec:.2f}s)")
    print(f"{'-' * 74}")
    print(f"  • Average 1-Day Prediction Latency    : {np.mean(infer_times)*1000:.2f} ms / day")
    print(f"  • Average 1-Day LoRA Adaptation Time  : {np.mean(adapt_times):.2f} s / day")
    print(f"  • Average Total Step Cycle Time       : {np.mean(step_times):.2f} s / day")
    print(f"{'=' * 74}\n")

    # Plotly Dual Candlestick Graph
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df_nifty['date'],
        open=df_nifty['open'],
        high=df_nifty['high'],
        low=df_nifty['low'],
        close=df_nifty['close'],
        name='Actual NIFTY 50 Candlesticks',
        increasing_line_color='#10b981',
        decreasing_line_color='#ef4444',
        opacity=0.65
    ))

    valid_mask = ~df_preds['close'].isna()
    fig.add_trace(go.Candlestick(
        x=df_nifty.loc[valid_mask, 'date'],
        open=df_preds.loc[valid_mask, 'open'],
        high=df_preds.loc[valid_mask, 'high'],
        low=df_preds.loc[valid_mask, 'low'],
        close=df_preds.loc[valid_mask, 'close'],
        name='LoRA Predicted Candlesticks',
        increasing_line_color='#3b82f6',
        decreasing_line_color='#f59e0b',
        opacity=0.85
    ))

    split_date = df_nifty['date'].iloc[start_eval_idx]
    fig.add_vline(
        x=split_date,
        line_width=1.5,
        line_dash="dash",
        line_color="#64748b",
        annotation_text="Mid-2023 Prediction Start",
        annotation_position="top right"
    )

    fig.update_layout(
        title="NIFTY 50 (Mid-2023 to 2026): Actual vs LoRA Walk-Forward Candlesticks",
        xaxis_title="Date",
        yaxis_title="Index Level (Points)",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    out_html = os.path.join(OUTPUT_DIR, "nifty50_kline_walk_forward.html")
    fig.write_html(out_html)
    print(f"✓ Saved interactive dual-candlestick chart to: {out_html}")

    try:
        webbrowser.open_new_tab(f"file://{os.path.abspath(out_html)}")
    except Exception:
        pass

if __name__ == "__main__":
    run_kline_pipeline()