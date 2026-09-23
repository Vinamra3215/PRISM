import os
import sys
import time
import json
import webbrowser
import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file as load_safetensors

# ---------------------------------------------------------------------------
# Path Configuration & Environment Setup
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
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Robust Base Foundation Model & Tokenizer Loaders
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
# Data Preprocessing & Tokenization Engine
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

# ---------------------------------------------------------------------------
# Daily Full-Parameter Training Routine
# ---------------------------------------------------------------------------
def train_full_parameter_step(model, dataloader, epochs, lr, weight_decay, recency_decay):
    for p in model.parameters():
        p.requires_grad = True
        
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    for _ in range(epochs):
        for batch in dataloader:
            s1_in = batch['s1_ids'].to(DEVICE)
            s2_in = batch['s2_ids'].to(DEVICE)
            s1_tgt = batch['s1_targets'].to(DEVICE)
            
            time_weights = torch.linspace(recency_decay, 1.0, steps=s1_in.size(1), device=DEVICE)
            optimizer.zero_grad()
            outputs = model(s1_ids=s1_in, s2_ids=s2_in)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            
            raw_loss = criterion(logits.reshape(-1, logits.size(-1)), s1_tgt.reshape(-1)).view(s1_in.size(0), s1_in.size(1))
            loss = (raw_loss * time_weights).mean()
            loss.backward()
            optimizer.step()

# ---------------------------------------------------------------------------
# STAGE 1: Optuna Objective Function for Full-Parameter FT
# ---------------------------------------------------------------------------
def full_ft_objective(trial):
    rolling_window = trial.suggest_int("rolling_window", 220, 360, step=20)
    adapt_epochs = trial.suggest_int("adapt_epochs", 2, 4)
    lr = trial.suggest_float("lr", 8e-6, 4e-5, log=True)
    recency_decay = trial.suggest_float("recency_decay", 0.2, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-3, log=True)

    df_nifty = load_local_nifty_data()
    
    # Validation Window: Early 2023 to Mid 2023
    val_slice = df_nifty[(df_nifty['date'] >= '2023-01-01') & (df_nifty['date'] <= '2023-06-30')].reset_index(drop=True)
    if len(val_slice) == 0:
        val_slice = df_nifty.iloc[rolling_window : rolling_window + 50].reset_index(drop=True)
        
    start_eval_idx = df_nifty[df_nifty['date'] == val_slice['date'].iloc[0]].index[0]
    total_val_steps = min(35, len(val_slice))

    tokenizer = get_tokenizer()
    if hasattr(tokenizer, 'to'):
        tokenizer = tokenizer.to(DEVICE)
    tokenizer.eval()

    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    y_true_returns = []
    y_pred_returns = []

    for step in range(total_val_steps):
        target_idx = start_eval_idx + step
        train_start = target_idx - rolling_window
        train_end = target_idx
        
        if train_start < 0:
            continue
            
        window_train_df = df_nifty.iloc[train_start:train_end].reset_index(drop=True)
        actual_prev_close = df_nifty['close'].iloc[target_idx - 1]
        actual_target_close = df_nifty['close'].iloc[target_idx]
        
        s1_stream, s2_stream = tokenize_window(window_train_df, tokenizer)
        train_ds = RollingPatchDataset(s1_stream, s2_stream, seq_len=32)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        
        # Fresh base model instance for isolated full-parameter tuning
        model = get_base_model().to(DEVICE)
        train_full_parameter_step(model, train_loader, epochs=adapt_epochs, lr=lr, 
                                  weight_decay=weight_decay, recency_decay=recency_decay)
        
        model.eval()
        predictor = KronosPredictor(model, tokenizer, max_context=rolling_window)
        
        with torch.no_grad():
            pred_res = predictor.predict(
                df=window_train_df[feature_cols],
                x_timestamp=pd.to_datetime(window_train_df['date']),
                y_timestamp=pd.to_datetime(df_nifty['date'].iloc[target_idx:target_idx+1]),
                pred_len=1
            )
            
        pred_c = float(pred_res['close'].iloc[-1]) if isinstance(pred_res, pd.DataFrame) else float(pred_res[-1, 3])
        
        r_actual = (actual_target_close - actual_prev_close) / actual_prev_close
        r_pred = (pred_c - actual_prev_close) / actual_prev_close
        
        y_true_returns.append(r_actual)
        y_pred_returns.append(r_pred)
        
        del model
        torch.cuda.empty_cache()

    if len(y_true_returns) == 0:
        return float("inf")

    return_mae = float(np.mean(np.abs(np.array(y_true_returns) - np.array(y_pred_returns)))) * 100
    print(f"  [Trial #{trial.number:02d}] Window: {rolling_window}d | Epochs: {adapt_epochs} | LR: {lr:.2e} | Return MAE: {return_mae:.4f} %")
    return return_mae

def run_optuna_tuning_stage(n_trials: int = 10):
    print("=" * 70)
    print(f" STAGE 1: OPTUNA HYPERPARAMETER TUNING (2022 LOOKBACK -> MID-2023 VAL)")
    print("=" * 70)
    t_start = time.perf_counter()
    
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(full_ft_objective, n_trials=n_trials)
    
    elapsed = time.perf_counter() - t_start
    mins, secs = divmod(elapsed, 60)
    
    best_params = study.best_params
    print("\n" + "=" * 70)
    print("✓ STAGE 1 COMPLETED: OPTIMAL FULL-FT HYPERPARAMETERS IDENTIFIED")
    print(f"  • Total Search Time     : {int(mins)}m {secs:.2f}s")
    print(f"  • Best Validation Return MAE : {study.best_value:.4f} %")
    for k, v in best_params.items():
        print(f"     - {k}: {v}")
    print("=" * 70 + "\n")
    
    # Save optimal hyperparameters to JSON
    json_path = os.path.join(OUTPUT_DIR, "best_full_ft_hyperparameters.json")
    with open(json_path, "w") as f:
        json.dump({"optimal_hyperparameters": best_params, "best_val_mae": study.best_value}, f, indent=4)
    print(f"✓ Saved optimal parameters to: {json_path}\n")
    return best_params

# ---------------------------------------------------------------------------
# STAGE 2: Walk-Forward Full-Parameter Prediction (Mid-2023 to 2026)
# ---------------------------------------------------------------------------
def run_walk_forward_prediction_stage(params):
    start_total_time = time.perf_counter()
    
    rolling_window = int(params.get("rolling_window", 280))
    adapt_epochs = int(params.get("adapt_epochs", 3))
    lr = float(params.get("lr", 2e-5))
    recency_decay = float(params.get("recency_decay", 0.3))
    weight_decay = float(params.get("weight_decay", 1e-3))
    batch_size = 16

    df_nifty = load_local_nifty_data()
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    
    eval_start_date = pd.to_datetime("2023-07-01")
    eval_indices = df_nifty[df_nifty['date'] >= eval_start_date].index
    
    if len(eval_indices) == 0:
        raise ValueError(f"No dates found >= {eval_start_date.date()} in dataset!")
        
    start_eval_idx = eval_indices[0]
    total_candles = len(df_nifty)
    total_steps = total_candles - start_eval_idx

    print("=" * 70)
    print(" STAGE 2: WALK-FORWARD FULL-PARAMETER PREDICTIONS (MID-2023 TO 2026)")
    print("=" * 70)
    print(f"  • Total Ingested Candles : {total_candles}")
    print(f"  • Rolling Context Window : {rolling_window} trading days")
    print(f"  • Evaluation Horizon     : {df_nifty['date'].iloc[start_eval_idx].date()} to {df_nifty['date'].iloc[-1].date()} ({total_steps} trading days)")
    print(f"  • Active Full-FT Params  : Epochs={adapt_epochs} | LR={lr:.2e} | Recency Decay={recency_decay:.4f}")
    print("=" * 70 + "\n")
    
    tokenizer = get_tokenizer()
    if hasattr(tokenizer, 'to'):
        tokenizer = tokenizer.to(DEVICE)
    tokenizer.eval()

    pred_records = []
    for _ in range(start_eval_idx):
        pred_records.append({'open': np.nan, 'high': np.nan, 'low': np.nan, 'close': np.nan})

    train_times = []
    infer_times = []
    step_times = []

    print("--> Commencing Daily Walk-Forward Full-Parameter Loop...")

    for step in range(total_steps):
        predict_idx = start_eval_idx + step
        train_end = predict_idx
        train_start = max(0, train_end - rolling_window)
        
        target_date = df_nifty['date'].iloc[predict_idx]
        actual_close = df_nifty['close'].iloc[predict_idx]
        window_df = df_nifty.iloc[train_start:train_end].reset_index(drop=True)
        
        t_step_start = time.perf_counter()
        
        s1_stream, s2_stream = tokenize_window(window_df, tokenizer)
        train_ds = RollingPatchDataset(s1_stream, s2_stream, seq_len=32)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        
        # 1. Instantiate fresh base foundation model for this rolling window
        t_train_start = time.perf_counter()
        model = get_base_model().to(DEVICE)
        
        # 2. Execute Full-Parameter Fine-Tuning across all transformer layers
        train_full_parameter_step(model, train_loader, epochs=adapt_epochs, lr=lr, 
                                  weight_decay=weight_decay, recency_decay=recency_decay)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_train = time.perf_counter() - t_train_start
        train_times.append(t_train)
        
        # 3. Autoregressive Candlestick Inference
        t_infer_start = time.perf_counter()
        model.eval()
        predictor = KronosPredictor(model, tokenizer, max_context=rolling_window)
        
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
        
        del model
        torch.cuda.empty_cache()

        t_step = time.perf_counter() - t_step_start
        step_times.append(t_step)

        if (step + 1) % 25 == 0 or step == total_steps - 1:
            print(f"Step [{step+1:03d}/{total_steps:03d}] Date: {target_date.date()} | "
                  f"Actual: {actual_close:.2f} | Pred: {p_c:.2f} | "
                  f"Full-FT: {t_train:.2f}s | Infer: {t_infer*1000:.1f}ms | Step Time: {t_step:.2f}s")

    total_elapsed_sec = time.perf_counter() - start_total_time
    total_net_infer_sec = sum(infer_times)
    total_net_train_sec = sum(train_times)

    df_preds = pd.DataFrame(pred_records)
    
    # Save full predictions dataset to CSV
    pred_export_df = df_nifty[['date', 'open', 'high', 'low', 'close']].copy()
    pred_export_df['pred_open'] = df_preds['open']
    pred_export_df['pred_high'] = df_preds['high']
    pred_export_df['pred_low'] = df_preds['low']
    pred_export_df['pred_close'] = df_preds['close']
    csv_out_path = os.path.join(OUTPUT_DIR, "nifty50_full_param_predictions.csv")
    pred_export_df.to_csv(csv_out_path, index=False)
    print(f"\n✓ Saved OHLC predictions dataset to: {csv_out_path}")

    # Compute risk report
    metrics = calculate_return_and_kline_metrics(df_nifty, df_preds)
    save_and_print_metrics(metrics, output_dir=OUTPUT_DIR, title="NIFTY 50 (TUNED FULL-FT ROLLING) RISK REPORT")
    
    # Latency and profiling summary
    print(f"{'=' * 22} TOTAL TIME & LATENCY PROFILING REPORT {'=' * 22}")
    print(f"  • Horizon Evaluated                   : Mid-2023 to 2026 ({total_steps} trading days)")
    print(f"  • TOTAL NET PREDICTION INFERENCE TIME : {int(total_net_infer_sec // 60)}m {total_net_infer_sec % 60:.2f}s ({total_net_infer_sec:.2f}s)")
    print(f"  • TOTAL NET FULL-FT TRAINING TIME     : {int(total_net_train_sec // 60)}m {total_net_train_sec % 60:.2f}s ({total_net_train_sec:.2f}s)")
    print(f"  • TOTAL END-TO-END PIPELINE TIME      : {int(total_elapsed_sec // 60)}m {total_elapsed_sec % 60:.2f}s ({total_elapsed_sec:.2f}s)")
    print(f"{'-' * 74}")
    print(f"  • Average 1-Day Prediction Latency    : {np.mean(infer_times)*1000:.2f} ms / day")
    print(f"  • Average 1-Day Full-FT Train Time    : {np.mean(train_times):.2f} s / day")
    print(f"  • Average Total Step Cycle Time       : {np.mean(step_times):.2f} s / day")
    print(f"{'=' * 74}\n")

    # Plotly Visual Dual Candlestick Graph
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
        name='Tuned Full-FT Predicted Candlesticks',
        increasing_line_color='#06b6d4',
        decreasing_line_color='#f43f5e',
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
        title="NIFTY 50 (Mid-2023 to 2026): Actual vs Tuned Full-FT Rolling Candlesticks",
        xaxis_title="Date",
        yaxis_title="Index Level (Points)",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    out_html = os.path.join(OUTPUT_DIR, "nifty50_tuned_full_ft_walk_forward.html")
    fig.write_html(out_html)
    print(f"✓ Saved interactive dual-candlestick chart to: {out_html}")

    try:
        webbrowser.open_new_tab(f"file://{os.path.abspath(out_html)}")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Main Execution Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Run Stage 1: Full-Parameter Optuna Hyperparameter Optimization
    optimal_hyperparams = run_optuna_tuning_stage(n_trials=10)
    
    # 2. Run Stage 2: Walk-Forward Rolling Prediction using the Tuned Hyperparameters
    run_walk_forward_prediction_stage(optimal_hyperparams)