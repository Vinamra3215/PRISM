import os
import sys
import time
import json
import optuna
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

for p in [CURRENT_DIR, MY_KRONOS_PROJECT, KRONOS_REPO_PATH]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from model.kronos import Kronos, KronosTokenizer, KronosPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NIFTY_DATA_PATH = os.path.join(CURRENT_DIR, "nifty50_2022_2026.parquet")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Robust Model & Tokenizer Loaders (Hub + Cache Discovery)
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

def build_trial_lora(base_model, r, alpha, dropout):
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
        lora_dropout=dropout,
        bias="none",
        task_type=None
    )
    return get_peft_model(base_model, lora_config)

# ---------------------------------------------------------------------------
# Optuna Tuning Objective
# ---------------------------------------------------------------------------
def objective(trial):
    rolling_window = trial.suggest_int("rolling_window", 380, 480, step=20)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32])
    adapt_epochs = trial.suggest_int("adapt_epochs", 2, 4)
    lr = trial.suggest_float("lr", 8e-5, 4e-4, log=True)
    recency_decay = trial.suggest_float("recency_decay", 0.2, 0.7)

    df_nifty = load_local_nifty_data()
    total_len = len(df_nifty)
    
    start_eval_idx = max(rolling_window, 400)
    total_val_steps = min(40, total_len - start_eval_idx)

    if total_val_steps <= 0:
        raise ValueError(f"Dataset length {total_len} is too short for rolling window {rolling_window}")

    tokenizer = get_tokenizer()
    if hasattr(tokenizer, 'to'):
        tokenizer = tokenizer.to(DEVICE)
    tokenizer.eval()

    base_model = get_base_model().to(DEVICE)
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    
    y_true_returns = []
    y_pred_returns = []

    for step in range(total_val_steps):
        target_idx = start_eval_idx + step
        train_start = target_idx - rolling_window
        train_end = target_idx
        
        window_train_df = df_nifty.iloc[train_start:train_end].reset_index(drop=True)
        actual_prev_close = df_nifty['close'].iloc[target_idx - 1]
        actual_target_close = df_nifty['close'].iloc[target_idx]
        
        s1_stream, s2_stream = tokenize_window(window_train_df, tokenizer)
        train_ds = RollingPatchDataset(s1_stream, s2_stream, seq_len=32)
        train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
        
        model = build_trial_lora(base_model, lora_r, lora_alpha, dropout=0.05)
        model.train()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        for _ in range(adapt_epochs):
            for batch in train_loader:
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
        
        # Cleanly unload PEFT adapter to prevent multiple adapter stacking
        if hasattr(model, 'unload'):
            base_model = model.unload()
        del model
        torch.cuda.empty_cache()

    del base_model
    torch.cuda.empty_cache()

    return_mae = float(np.mean(np.abs(np.array(y_true_returns) - np.array(y_pred_returns)))) * 100
    print(f"Trial #{trial.number:02d} | Window: {rolling_window}d | r: {lora_r} | alpha: {lora_alpha} | LR: {lr:.2e} | Return MAE: {return_mae:.4f} %")
    return return_mae

def run_optuna_study(n_trials: int = 12):
    start_total_time = time.perf_counter()
    print("=" * 65)
    print(f"--> STARTING OPTUNA STUDY ON NIFTY 50 DATA ({n_trials} Trials)...")
    print("=" * 65)
    
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials)
    
    total_elapsed_sec = time.perf_counter() - start_total_time
    minutes, seconds = divmod(total_elapsed_sec, 60)
    
    print("\n" + "=" * 65)
    print("✓ OPTUNA HYPERPARAMETER TUNING COMPLETED")
    print(f"  • Total Tuning Wall-Clock Time : {int(minutes)}m {seconds:.2f}s ({total_elapsed_sec:.2f} total seconds)")
    print(f"  • Best Validation Return MAE   : {study.best_value:.4f} %")
    print("  • Optimal Hyperparameter Set:")
    for k, v in study.best_params.items():
        print(f"     - {k}: {v}")
    print("=" * 65)
    
    return study.best_params

if __name__ == "__main__":
    run_optuna_study(n_trials=12)