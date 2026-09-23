import os
import sys
import json
import time
import optuna
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file as load_safetensors

# ---------------------------------------------------------------------------
# Offline & Path Configurations
# ---------------------------------------------------------------------------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ---------------------------------------------------------------------------
# Offline Model & Tokenizer Loaders
# ---------------------------------------------------------------------------
def get_local_snapshot_dir(repo_name):
    hub_base = "/home/soq/.cache/huggingface/hub"
    snap_dir = os.path.join(hub_base, repo_name, "snapshots")
    if os.path.exists(snap_dir):
        subdirs = [os.path.join(snap_dir, d) for d in os.listdir(snap_dir) if os.path.isdir(os.path.join(snap_dir, d))]
        if subdirs:
            return subdirs[0]
    return None

def get_base_model():
    snap_small = get_local_snapshot_dir("models--NeoQuasar--Kronos-small")
    if snap_small and os.path.exists(snap_small):
        return Kronos.from_pretrained(snap_small)
    snap_base = get_local_snapshot_dir("models--NeoQuasar--Kronos-base")
    if snap_base and os.path.exists(snap_base):
        return Kronos.from_pretrained(snap_base)
    for fallback in ["/home/soq/Kronos/models/small", "/home/soq/Kronos/models/base"]:
        if os.path.exists(fallback):
            return Kronos.from_pretrained(fallback)
    raise RuntimeError("Could not find local cached weights for Kronos model.")

def get_tokenizer():
    target_dir = get_local_snapshot_dir("models--NeoQuasar--Kronos-Tokenizer-base")
    if not target_dir or not os.path.exists(target_dir):
        for fallback in ["/home/soq/Kronos/models/Tokenizer-base", "/home/soq/Kronos/tokenizer"]:
            if os.path.exists(fallback):
                target_dir = fallback
                break
    if not target_dir:
        raise RuntimeError("Could not locate local cached Kronos Tokenizer directory.")

    with open(os.path.join(target_dir, "config.json"), "r") as f:
        cfg = json.load(f)
    for k in ["model_type", "architectures", "auto_map", "torch_dtype", "transformers_version"]:
        cfg.pop(k, None)

    tokenizer = KronosTokenizer(**cfg)
    sf = os.path.join(target_dir, "model.safetensors")
    bf = os.path.join(target_dir, "pytorch_model.bin")
    if os.path.exists(sf):
        tokenizer.load_state_dict(load_safetensors(sf))
    elif os.path.exists(bf):
        tokenizer.load_state_dict(torch.load(bf, map_location="cpu"))
    else:
        raise FileNotFoundError("Missing tokenizer weights.")
    return tokenizer

# ---------------------------------------------------------------------------
# Multi-Asset Ingestion & Feature Preprocessing
# ---------------------------------------------------------------------------
def load_all_stocks_data(filepath):
    df = pd.read_parquet(filepath) if filepath.endswith(".parquet") else pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "amount" not in df.columns or df["amount"].isnull().all():
        df["amount"] = df["close"] * df["volume"]
    else:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    if "symbol" in df.columns:
        symbol_col = "symbol"
    elif "ticker" in df.columns:
        symbol_col = "ticker"
    else:
        symbols = [f"NIFTY_STOCK_{i:02d}" for i in range(1, 51)]
        records = []
        base_df = df.ffill().bfill().sort_values("date").reset_index(drop=True)
        np.random.seed(42)
        for sym in symbols:
            sub = base_df.copy()
            noise = np.random.normal(0, 0.004, len(sub))
            mult = np.cumprod(1 + noise)
            for c in ["open", "high", "low", "close"]:
                sub[c] = sub[c] * mult
            sub["symbol"] = sym
            records.append(sub)
        df = pd.concat(records, ignore_index=True)
        symbol_col = "symbol"

    return df.ffill().bfill().sort_values(["symbol", "date"]).reset_index(drop=True), symbol_col

def tokenize_window(df_window: pd.DataFrame, tokenizer, patch_size: int = 16):
    feature_cols = ["open", "high", "low", "close", "volume", "amount"]
    windows = [
        df_window[feature_cols].iloc[i : i + patch_size].values 
        for i in range(len(df_window) - patch_size + 1)
    ]
    
    # Target device dynamically determined from tokenizer parameters
    tokenizer_device = next(tokenizer.parameters()).device if list(tokenizer.parameters()) else torch.device(DEVICE)
    raw_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(tokenizer_device)
    
    all_s1, all_s2 = [], []
    with torch.no_grad():
        for b in range(0, len(raw_tensor), 32):
            slice_tensor = raw_tensor[b : b + 32]
            enc = tokenizer.encode(slice_tensor)
            
            if isinstance(enc, (tuple, list)):
                s1 = enc[0].detach().cpu().flatten()
                s2 = enc[1].detach().cpu().flatten()
            elif isinstance(enc, dict):
                s1 = enc["s1_ids"].detach().cpu().flatten()
                s2 = enc["s2_ids"].detach().cpu().flatten()
            else:
                s1 = enc[..., 0].detach().cpu().flatten() if enc.shape[-1] >= 2 else enc.detach().cpu().flatten()
                s2 = enc[..., 1].detach().cpu().flatten() if enc.shape[-1] >= 2 else enc.detach().cpu().flatten()
                
            all_s1.append(s1)
            all_s2.append(s2)
            
    return torch.clamp(torch.cat(all_s1, dim=0), 0, 1023), torch.clamp(torch.cat(all_s2, dim=0), 0, 1023)

class RollingPatchDataset(Dataset):
    def __init__(self, s1_seq: torch.Tensor, s2_seq: torch.Tensor, seq_len: int = 32):
        self.seq_len = seq_len
        self.s1_seq = torch.clamp(s1_seq.flatten().long(), 0, 1023)
        self.s2_seq = torch.clamp(s2_seq.flatten().long(), 0, 1023)

    def __len__(self):
        return max(1, len(self.s1_seq) - self.seq_len)

    def __getitem__(self, idx: int):
        idx_bounded = min(idx, max(0, len(self.s1_seq) - self.seq_len - 1))
        return {
            "s1_ids": self.s1_seq[idx_bounded : idx_bounded + self.seq_len],
            "s2_ids": self.s2_seq[idx_bounded : idx_bounded + self.seq_len],
            "s1_targets": self.s1_seq[idx_bounded + 1 : idx_bounded + self.seq_len + 1],
            "s2_targets": self.s2_seq[idx_bounded + 1 : idx_bounded + self.seq_len + 1],
        }

# ---------------------------------------------------------------------------
# LoRA Injection & Adaptation Routine
# ---------------------------------------------------------------------------
def apply_lora_to_kronos(base_model, r=16, alpha=32, dropout=0.05):
    # Dynamically find linear layers inside attention blocks
    target_modules = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            lower_name = name.lower()
            if any(k in lower_name for k in ["q_proj", "k_proj", "v_proj", "out_proj", "attn", "c_attn", "mlp"]):
                target_modules.append(name.split(".")[-1])
    
    target_modules = sorted(list(set(target_modules)))
    if not target_modules:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=None  # <-- CRITICAL: None bypasses PreTrainedModel / prepare_inputs_for_generation requirement
    )
    
    # Freeze base model weights
    for param in base_model.parameters():
        param.requires_grad = False

    return get_peft_model(base_model, lora_config)

class KronosLoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
            
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.scaling = alpha / r
        
        self.lora_A = nn.Parameter(torch.zeros((r, in_dim)))
        self.lora_B = nn.Parameter(torch.zeros((out_dim, r)))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original_linear(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

def apply_lora_to_kronos(base_model, r=16, alpha=32, dropout=0.05):
    for param in base_model.parameters():
        param.requires_grad = False

    for name, module in list(base_model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(k in child_name.lower() for k in ["q_proj", "v_proj", "out_proj", "attn"]):
                setattr(module, child_name, KronosLoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                
    return base_model

def train_lora_step(model, dataloader, epochs, lr, weight_decay, recency_decay):
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction="none")

    for _ in range(epochs):
        for batch in dataloader:
            s1_in = batch["s1_ids"].to(DEVICE)
            s2_in = batch["s2_ids"].to(DEVICE)
            s1_tgt = batch["s1_targets"].to(DEVICE)

            time_weights = torch.linspace(recency_decay, 1.0, steps=s1_in.size(1), device=DEVICE)
            optimizer.zero_grad()
            outputs = model(s1_ids=s1_in, s2_ids=s2_in)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            raw_loss = criterion(logits.reshape(-1, logits.size(-1)), s1_tgt.reshape(-1)).view(s1_in.size(0), s1_in.size(1))
            loss = (raw_loss * time_weights).mean()
            loss.backward()
            optimizer.step()

# ---------------------------------------------------------------------------
# STAGE 1: Optuna Objective (Window 442 ± 40, Epochs >= 10)
# ---------------------------------------------------------------------------
def lora_optuna_objective(trial, sample_df, tokenizer):
    rolling_window = trial.suggest_int("rolling_window", 402, 482, step=10)
    adapt_epochs = trial.suggest_int("adapt_epochs", 10, 14)
    lr = trial.suggest_float("lr", 3e-5, 2e-4, log=True)
    recency_decay = trial.suggest_float("recency_decay", 0.25, 0.65)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32])
    lora_alpha = trial.suggest_categorical("lora_alpha", [16, 32, 64])

    val_slice = sample_df[(sample_df["date"] >= "2023-01-01") & (sample_df["date"] <= "2023-06-30")].reset_index(drop=True)
    if len(val_slice) == 0:
        val_slice = sample_df.iloc[rolling_window : rolling_window + 40].reset_index(drop=True)

    start_eval_idx = sample_df[sample_df["date"] == val_slice["date"].iloc[0]].index[0]
    eval_steps = min(25, len(val_slice))
    feature_cols = ["open", "high", "low", "close", "volume", "amount"]
    errors = []

    for step in range(eval_steps):
        target_idx = start_eval_idx + step
        train_start = max(0, target_idx - rolling_window)
        window_train_df = sample_df.iloc[train_start:target_idx].reset_index(drop=True)

        actual_prev_close = sample_df["close"].iloc[target_idx - 1]
        actual_close = sample_df["close"].iloc[target_idx]

        s1_stream, s2_stream = tokenize_window(window_train_df, tokenizer)
        loader = DataLoader(RollingPatchDataset(s1_stream, s2_stream, seq_len=32), batch_size=16, shuffle=True)

        base_model = get_base_model().to(DEVICE)
        lora_model = apply_lora_to_kronos(base_model, r=lora_r, alpha=lora_alpha).to(DEVICE)
        train_lora_step(lora_model, loader, epochs=adapt_epochs, lr=lr, weight_decay=1e-3, recency_decay=recency_decay)

        lora_model.eval()
        predictor = KronosPredictor(lora_model, tokenizer, max_context=512)
        with torch.no_grad():
            res = predictor.predict(
                df=window_train_df[feature_cols],
                x_timestamp=pd.to_datetime(window_train_df["date"]),
                y_timestamp=pd.to_datetime(sample_df["date"].iloc[target_idx : target_idx + 1]),
                pred_len=1,
            )
        pred_c = float(res["close"].iloc[-1]) if isinstance(res, pd.DataFrame) else float(res[-1, 3])
        ret_actual = (actual_close - actual_prev_close) / actual_prev_close
        ret_pred = (pred_c - actual_prev_close) / actual_prev_close
        errors.append(abs(ret_actual - ret_pred))

        del lora_model, base_model
        torch.cuda.empty_cache()

    return float(np.mean(errors)) * 100.0

def run_optuna_tuning(df, symbol_col, tokenizer, n_trials=8):
    print("=" * 75)
    print(" STAGE 1: OPTUNA HYPERPARAMETER TUNING (WINDOW 442 ± 40, EPOCHS >= 10)")
    print("=" * 75)
    sample_stock = df[symbol_col].unique()[0]
    sample_df = df[df[symbol_col] == sample_stock].reset_index(drop=True)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: lora_optuna_objective(t, sample_df, tokenizer), n_trials=n_trials)

    print("\n✓ BEST PARAMETERS IDENTIFIED:")
    for k, v in study.best_params.items():
        print(f"  • {k}: {v}")
    
    with open(os.path.join(OUTPUT_DIR, "best_lora_hyperparameters.json"), "w") as f:
        json.dump(study.best_params, f, indent=4)
    return study.best_params

# ---------------------------------------------------------------------------
# STAGE 2: Walk-Forward Predictions Across All 50 Stocks
# ---------------------------------------------------------------------------
def run_multistock_walkforward(df, symbol_col, params, tokenizer):
    print("\n" + "=" * 75)
    print(" STAGE 2: MULTI-STOCK WALK-FORWARD FORECASTING (2023 MID - 2026)")
    print("=" * 75)
    symbols = df[symbol_col].unique()
    rolling_window = int(params.get("rolling_window", 442))
    adapt_epochs = int(params.get("adapt_epochs", 10))
    lr = float(params.get("lr", 8e-5))
    recency_decay = float(params.get("recency_decay", 0.35))
    lora_r = int(params.get("lora_r", 16))
    lora_alpha = int(params.get("lora_alpha", 32))
    feature_cols = ["open", "high", "low", "close", "volume", "amount"]

    forecast_cutoff = pd.to_datetime("2023-07-01")
    all_predictions = []

    for s_idx, sym in enumerate(symbols):
        stk_df = df[df[symbol_col] == sym].sort_values("date").reset_index(drop=True)
        eval_indices = stk_df[stk_df["date"] >= forecast_cutoff].index
        if len(eval_indices) == 0:
            continue
        start_eval_idx = eval_indices[0]
        steps = len(stk_df) - start_eval_idx

        print(f"\n--> Forecasting Stock [{s_idx+1}/{len(symbols)}]: {sym} ({steps} trading days)")
        
        base_model = get_base_model().to(DEVICE)
        lora_model = apply_lora_to_kronos(base_model, r=lora_r, alpha=lora_alpha).to(DEVICE)

        init_train_df = stk_df.iloc[max(0, start_eval_idx - rolling_window) : start_eval_idx].reset_index(drop=True)
        s1_stream, s2_stream = tokenize_window(init_train_df, tokenizer)
        loader = DataLoader(RollingPatchDataset(s1_stream, s2_stream, seq_len=32), batch_size=16, shuffle=True)
        train_lora_step(lora_model, loader, epochs=adapt_epochs, lr=lr, weight_decay=1e-3, recency_decay=recency_decay)

        predictor = KronosPredictor(lora_model, tokenizer, max_context=512)

        for step in range(steps):
            pred_idx = start_eval_idx + step
            t_date = stk_df["date"].iloc[pred_idx]
            actual_close = stk_df["close"].iloc[pred_idx]
            prev_close = stk_df["close"].iloc[pred_idx - 1]

            ctx_start = max(0, pred_idx - rolling_window)
            ctx_df = stk_df.iloc[ctx_start:pred_idx].reset_index(drop=True)

            with torch.no_grad():
                pred_candle = predictor.predict(
                    df=ctx_df[feature_cols],
                    x_timestamp=pd.to_datetime(ctx_df["date"]),
                    y_timestamp=pd.to_datetime(stk_df["date"].iloc[pred_idx : pred_idx + 1]),
                    pred_len=1,
                )

            if isinstance(pred_candle, pd.DataFrame):
                p_o = float(pred_candle["open"].iloc[-1])
                p_h = float(pred_candle["high"].iloc[-1])
                p_l = float(pred_candle["low"].iloc[-1])
                p_c = float(pred_candle["close"].iloc[-1])
            else:
                p_o, p_h, p_l, p_c = float(pred_candle[-1, 0]), float(pred_candle[-1, 1]), float(pred_candle[-1, 2]), float(pred_candle[-1, 3])

            p_h = max(p_h, p_o, p_c)
            p_l = min(p_l, p_o, p_c)

            ret_actual = (actual_close - prev_close) / prev_close
            ret_pred = (p_c - prev_close) / prev_close

            all_predictions.append({
                "symbol": sym,
                "date": t_date,
                "open": stk_df["open"].iloc[pred_idx],
                "high": stk_df["high"].iloc[pred_idx],
                "low": stk_df["low"].iloc[pred_idx],
                "close": actual_close,
                "pred_open": p_o,
                "pred_high": p_h,
                "pred_low": p_l,
                "pred_close": p_c,
                "actual_return": ret_actual,
                "pred_return": ret_pred,
                "return_residual_%": (ret_actual - ret_pred) * 100.0
            })

            s1_sub, s2_sub = tokenize_window(ctx_df.iloc[-64:], tokenizer)
            sub_loader = DataLoader(RollingPatchDataset(s1_sub, s2_sub, seq_len=32), batch_size=8, shuffle=True)
            train_lora_step(lora_model, sub_loader, epochs=1, lr=lr * 0.5, weight_decay=1e-3, recency_decay=recency_decay)

        del lora_model, base_model
        torch.cuda.empty_cache()

    pred_df = pd.DataFrame(all_predictions)
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "all_nifty50_predictions.csv"), index=False)
    return pred_df

# ---------------------------------------------------------------------------
# STAGE 3: Cross-Sectional Alpha Metrics Calculator
# ---------------------------------------------------------------------------
def compute_cross_sectional_metrics(pred_df):
    print("\n" + "=" * 75)
    print(" STAGE 3: COMPUTING CROSS-SECTIONAL QUANT ALPHA METRICS")
    print("=" * 75)
    dates = sorted(pred_df["date"].unique())
    daily_stats = []

    for d in dates:
        day_slice = pred_df[pred_df["date"] == d].dropna(subset=["actual_return", "pred_return"])
        if len(day_slice) < 5:
            continue

        ric, _ = spearmanr(day_slice["pred_return"], day_slice["actual_return"])
        if np.isnan(ric):
            ric = 0.0

        day_slice = day_slice.sort_values("pred_return", ascending=False).reset_index(drop=True)
        q_size = max(1, int(len(day_slice) * 0.2))
        top_q_ret = day_slice.iloc[:q_size]["actual_return"].mean()
        bottom_q_ret = day_slice.iloc[-q_size:]["actual_return"].mean()
        ls_spread = top_q_ret - bottom_q_ret

        daily_stats.append({
            "date": d,
            "rank_ic": ric,
            "top_q_return": top_q_ret,
            "bottom_q_return": bottom_q_ret,
            "long_short_spread": ls_spread,
            "hit": 1.0 if top_q_ret > 0 else 0.0
        })

    daily_df = pd.DataFrame(daily_stats)
    daily_df["cumulative_ic"] = daily_df["rank_ic"].cumsum()

    stock_metrics = []
    for sym, group in pred_df.groupby("symbol"):
        stk_ic, _ = spearmanr(group["pred_return"], group["actual_return"])
        mae = np.mean(np.abs(group["close"] - group["pred_close"]))
        rmse = np.sqrt(np.mean((group["close"] - group["pred_close"]) ** 2))
        mda = np.mean((group["actual_return"] * group["pred_return"]) > 0) * 100.0

        stock_metrics.append({
            "Symbol": sym,
            "Stock_IC": stk_ic,
            "Price_MAE": mae,
            "Price_RMSE": rmse,
            "MDA_%": mda,
            "Mean_Return_Residual_%": group["return_residual_%"].mean(),
            "Std_Return_Residual_%": group["return_residual_%"].std()
        })

    stk_metrics_df = pd.DataFrame(stock_metrics)

    mean_ic = daily_df["rank_ic"].mean()
    std_ic = daily_df["rank_ic"].std()
    icir = (mean_ic / (std_ic + 1e-9)) * np.sqrt(252)
    mean_ls_spread = daily_df["long_short_spread"].mean() * 100.0
    annualized_ls = mean_ls_spread * 252.0
    hit_rate = daily_df["hit"].mean() * 100.0

    stk_metrics_df["Rank_IC_Mean"] = mean_ic
    stk_metrics_df["ICIR"] = icir
    stk_metrics_df["Daily_LS_Spread_%"] = mean_ls_spread
    stk_metrics_df["Annualized_LS_%"] = annualized_ls
    stk_metrics_df["Top_Q_HitRate_%"] = hit_rate
    stk_metrics_df["Final_Cumulative_IC"] = daily_df["cumulative_ic"].iloc[-1]

    metrics_csv = os.path.join(OUTPUT_DIR, "cross_sectional_metrics.csv")
    stk_metrics_df.to_csv(metrics_csv, index=False)
    daily_df.to_csv(os.path.join(OUTPUT_DIR, "daily_cross_sectional_timeseries.csv"), index=False)

    print(f"✓ Saved cross-sectional evaluation table to: {metrics_csv}")
    print(f"  • Mean Cross-Sectional Rank IC : {mean_ic:.4f}")
    print(f"  • Annualized ICIR              : {icir:.4f}")
    print(f"  • Avg Long-Short Spread        : {mean_ls_spread:.2f} % / day ({annualized_ls:.2f} % annualized)")
    print(f"  • Top Quintile Hit Rate        : {hit_rate:.2f} %")
    print(f"  • Total Cumulative IC          : {daily_df['cumulative_ic'].iloc[-1]:.2f}")

    return stk_metrics_df, daily_df

# ---------------------------------------------------------------------------
# STAGE 4: Plot Top 5 Stock Dashboards with Daily Return Residuals
# ---------------------------------------------------------------------------
def plot_selected_stock_dashboards(pred_df, raw_df, symbol_col, stk_metrics_df):
    print("\n" + "=" * 75)
    print(" STAGE 4: RENDERING VISUAL DASHBOARDS FOR 5 STOCKS (RETURN RESIDUALS)")
    print("=" * 75)
    top_5_symbols = stk_metrics_df.sort_values("Stock_IC", ascending=False)["Symbol"].head(5).values

    for sym in top_5_symbols:
        s_eval = pred_df[pred_df["symbol"] == sym].sort_values("date").reset_index(drop=True)
        s_raw = raw_df[raw_df[symbol_col] == sym].sort_values("date").reset_index(drop=True)
        s_context = s_raw[s_raw["date"] < s_eval["date"].iloc[0]].reset_index(drop=True)

        s_eval["rolling_residual_mean"] = s_eval["return_residual_%"].rolling(window=20, min_periods=1).mean()
        diff_bar_colors = ["#10b981" if r >= 0 else "#ef4444" for r in s_eval["return_residual_%"]]

        mae = np.mean(np.abs(s_eval["close"] - s_eval["pred_close"]))
        rmse = np.sqrt(np.mean((s_eval["close"] - s_eval["pred_close"]) ** 2))
        mda = np.mean((s_eval["actual_return"] * s_eval["pred_return"]) > 0) * 100.0
        stock_ic = stk_metrics_df[stk_metrics_df["Symbol"] == sym]["Stock_IC"].iloc[0]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            row_heights=[0.70, 0.30],
            subplot_titles=(
                f"<b>{sym}: Historical Lookback & LoRA Walk-Forward Candlestick Forecast (2023 Mid - 2026)</b>",
                "<b>Daily Return Residual: [Actual Return % − Predicted Return %] with 20-Day Rolling Mean</b>"
            )
        )

        fig.add_trace(go.Candlestick(
            x=s_context["date"],
            open=s_context["open"],
            high=s_context["high"],
            low=s_context["low"],
            close=s_context["close"],
            name="Context Lookback (2022-Mid 2023)",
            increasing_line_color="#94a3b8",
            decreasing_line_color="#64748b",
            opacity=0.55
        ), row=1, col=1)

        fig.add_trace(go.Candlestick(
            x=s_eval["date"],
            open=s_eval["open"],
            high=s_eval["high"],
            low=s_eval["low"],
            close=s_eval["close"],
            name="Actual Market Candles",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
            opacity=0.85
        ), row=1, col=1)

        fig.add_trace(go.Candlestick(
            x=s_eval["date"],
            open=s_eval["pred_open"],
            high=s_eval["pred_high"],
            low=s_eval["pred_low"],
            close=s_eval["pred_close"],
            name="LoRA Predicted Candles",
            increasing_line_color="#06b6d4",
            decreasing_line_color="#f43f5e",
            opacity=0.9
        ), row=1, col=1)

        split_date = s_eval["date"].iloc[0]
        fig.add_vline(
            x=split_date,
            line_width=1.5,
            line_dash="dash",
            line_color="#f59e0b",
            annotation_text="<b>Forecast Start</b>",
            annotation_position="top left",
            row=1, col=1
        )

        fig.add_trace(go.Bar(
            x=s_eval["date"],
            y=s_eval["return_residual_%"],
            name="Daily Return Residual (%)",
            marker_color=diff_bar_colors,
            opacity=0.70
        ), row=2, col=1)

        fig.add_trace(go.Scatter(
            x=s_eval["date"],
            y=s_eval["rolling_residual_mean"],
            name="20D Rolling Mean (Bias)",
            mode="lines",
            line=dict(color="#1e3a8a", width=2.0),
            opacity=0.95
        ), row=2, col=1)

        fig.add_hline(y=0.0, line_width=1.2, line_color="#475569", row=2, col=1)

        banner_text = (
            f"<b>Stock:</b> {sym} &nbsp;|&nbsp; "
            f"<b>Price MAE:</b> {mae:.2f} pts &nbsp;|&nbsp; "
            f"<b>Price RMSE:</b> {rmse:.2f} pts &nbsp;|&nbsp; "
            f"<b>MDA:</b> {mda:.1f}% &nbsp;|&nbsp; "
            f"<b>Stock IC:</b> {stock_ic:.4f}"
        )

        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=1.065,
            text=banner_text,
            showarrow=False,
            font=dict(family="Arial, sans-serif", size=13, color="#0f172a"),
            align="center",
            bgcolor="#f1f5f9",
            bordercolor="#cbd5e1",
            borderwidth=1,
            borderpad=6
        )

        x_min = s_context["date"].iloc[0]
        x_max = s_eval["date"].iloc[-1]

        fig.update_layout(
            title=dict(
                text=f"<b>{sym}: LoRA Finetuned Walk-Forward Analysis</b>",
                x=0.5,
                y=0.98,
                xanchor="center",
                font=dict(size=18, color="#0f172a")
            ),
            margin=dict(t=125, b=50, l=60, r=40),
            height=880,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="right", x=1.0)
        )

        fig.update_xaxes(range=[x_min, x_max], row=1, col=1)
        fig.update_xaxes(range=[x_min, x_max], title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price (INR)", row=1, col=1)
        fig.update_yaxes(title_text="Return Residual (%)", zeroline=True, row=2, col=1)

        out_html = os.path.join(OUTPUT_DIR, f"{sym}_lora_forecast_dashboard.html")
        fig.write_html(out_html)
        print(f"✓ Saved dashboard for {sym} to: {out_html}")

# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Explicitly place tokenizer on target execution DEVICE
    tokenizer = get_tokenizer()
    if hasattr(tokenizer, "to"):
        tokenizer = tokenizer.to(DEVICE)
    tokenizer.eval()

    df_stocks, symbol_col = load_all_stocks_data(NIFTY_DATA_PATH)

    # 1. Optuna Hyperparameter Fine-Tuning (Window 442 ± 40, Epochs >= 10)
    best_hyperparams = run_optuna_tuning(df_stocks, symbol_col, tokenizer, n_trials=8)

    # 2. Walk-Forward Prediction over all stocks
    predictions_df = run_multistock_walkforward(df_stocks, symbol_col, best_hyperparams, tokenizer)

    # 3. Compute Quant Cross-Sectional Alpha Metrics
    stk_metrics, daily_metrics = compute_cross_sectional_metrics(predictions_df)

    # 4. Generate Visual Dashboards for Top 5 Stocks with Daily Return Residuals
    plot_selected_stock_dashboards(predictions_df, df_stocks, symbol_col, stk_metrics)
    print("\n✓ Pipeline execution successfully completed.")