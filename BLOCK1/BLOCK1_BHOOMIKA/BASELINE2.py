import os
import sys
import gc
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from peft import LoraConfig, get_peft_model
import optuna
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Disable Optuna verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------
# 1. DYNAMIC PATH RESOLUTION FOR KRONOS MODULE & WEIGHTS
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

possible_roots = [
    os.path.abspath(os.path.join(current_dir, "..", "..")),
    os.path.abspath(os.path.join(current_dir, "..")),
    "/home/soq/Kronos"
]

kronos_root = None
for root in possible_roots:
    if os.path.exists(os.path.join(root, "model", "kronos.py")):
        kronos_root = root
        break

if not kronos_root:
    kronos_root = "/home/soq/Kronos"

if kronos_root not in sys.path:
    sys.path.insert(0, kronos_root)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    import model.kronos as kronos_module
    print(f"[+] Successfully loaded Kronos model module from: {kronos_root}")
except ModuleNotFoundError as e:
    raise ImportError(f"[!] Unable to import model.kronos from {kronos_root}. Error: {e}")

# ------------------------------------------------------------------
# 2. CONFIGURATION & FILE PATHS
# ------------------------------------------------------------------
WINDOW_SIZE = 512  # Warmup Window Size
N_OPTUNA_TRIALS = 15  # Number of hyperparameter tuning iterations

PARQUET_FILE_PATH = os.path.join(kronos_root, "NIFTY50_2022_to_today.parquet")
if not os.path.exists(PARQUET_FILE_PATH):
    PARQUET_FILE_PATH = os.path.join(kronos_root, "stock_data", "NIFTY50_2022_to_today.parquet")

PRETRAINED_WEIGHTS_PATH = os.path.join(kronos_root, "weights", "best_nifty_2022_weights.pth")
if not os.path.exists(PRETRAINED_WEIGHTS_PATH):
    PRETRAINED_WEIGHTS_PATH = os.path.join(kronos_root, "best_nifty_2022_weights.pth")

OUTPUT_HTML = os.path.join(current_dir, "kronos_2022_to_2026_dashboard.html")

# ------------------------------------------------------------------
# 3. DATA LOADING (JAN 2022 TO SEPT 2026 FILTER)
# ------------------------------------------------------------------
def load_parquet_data():
    if not os.path.exists(PARQUET_FILE_PATH):
        raise FileNotFoundError(f"[!] Parquet file not found at {PARQUET_FILE_PATH}")
        
    df = pd.read_parquet(PARQUET_FILE_PATH)
    df.columns = [str(col).strip().lower() for col in df.columns]
    
    if 'date' not in df.columns:
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.index.name: 'date'})
        else:
            df['date'] = pd.date_range(start="2022-01-01", periods=len(df), freq='B')

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    df = df[df['date'] >= '2022-01-01'].reset_index(drop=True)

    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['open', 'high', 'low', 'close']).reset_index(drop=True)

    df['open_ret'] = np.log(df['open'] / df['open'].shift(1))
    df['high_ret'] = np.log(df['high'] / df['high'].shift(1))
    df['low_ret'] = np.log(df['low'] / df['low'].shift(1))
    df['close_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    if 'volume' in df.columns:
        df['volume'] = df['volume'].fillna(0) + 1e-5
        df['vol_ret'] = np.log(df['volume'] / df['volume'].shift(1))
    else:
        df['vol_ret'] = 0.0
    
    df_clean = df.dropna().reset_index(drop=True)
    print(f"[+] Data Filtered: {len(df_clean)} Trading Days | Start: {df_clean['date'].min().strftime('%d-%b-%Y')} ---> End: {df_clean['date'].max().strftime('%d-%b-%Y')}")
    return df_clean

# ------------------------------------------------------------------
# 4. DATASET & LORA MODEL ARCHITECTURE
# ------------------------------------------------------------------
class ReturnDataset(Dataset):
    def __init__(self, data, seq_len=60):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len, 3]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class KronosReturnPredictor(nn.Module):
    def __init__(self, d_model=256, input_dim=5):
        super().__init__()
        model_cls = getattr(kronos_module, 'Kronos', None) or getattr(kronos_module, 'KronosModel', None)
        kronos_args = {
            's1_bits': 8, 's2_bits': 8, 'n_layers': 6,
            'd_model': d_model, 'n_heads': 8, 'ff_dim': d_model * 2,
            'ffn_dropout_p': 0.1, 'attn_dropout_p': 0.1, 'resid_dropout_p': 0.1,
            'token_dropout_p': 0.1, 'learn_te': True
        }
        self.backbone = model_cls(**kronos_args)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.temporal_fc = nn.Linear(60 * d_model, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        h = self.input_proj(x)
        h_flat = h.view(h.size(0), -1)
        h_feat = torch.relu(self.temporal_fc(h_flat))
        return self.head(h_feat).squeeze(-1)

def build_lora_model(lora_r=8, lora_alpha=16, lora_dropout=0.05):
    raw_model = KronosReturnPredictor(d_model=256, input_dim=5)
    if os.path.exists(PRETRAINED_WEIGHTS_PATH):
        try:
            checkpoint = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            raw_model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass

    lora_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha,
        target_modules=["temporal_fc", "input_proj"],
        lora_dropout=lora_dropout, bias="none"
    )
    return get_peft_model(raw_model, lora_config).to(device)

# ------------------------------------------------------------------
# 5. OPTUNA HYPERPARAMETER TUNING
# ------------------------------------------------------------------
def optimize_hyperparameters(train_scaled):
    print(f"\n[+] Starting Optuna Hyperparameter Optimization ({N_OPTUNA_TRIALS} Trials)...")
    
    val_split = int(len(train_scaled) * 0.8)
    train_sub = train_scaled[:val_split]
    val_sub = train_scaled[val_split:]

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        lora_r = trial.suggest_categorical("lora_r", [4, 8, 16])
        lora_alpha = trial.suggest_categorical("lora_alpha", [8, 16, 32])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        lora_dropout = trial.suggest_float("lora_dropout", 0.01, 0.1)

        ds_train = ReturnDataset(train_sub, seq_len=60)
        ds_val = ReturnDataset(val_sub, seq_len=60)

        if len(ds_train) == 0 or len(ds_val) == 0:
            return float("inf")

        loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
        loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

        model = build_lora_model(lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(5):
            for x_b, y_b in loader_train:
                x_b, y_b = x_b.to(device), y_b.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_b), y_b)
                loss.backward()
                optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x_b, y_b in loader_val:
                x_b, y_b = x_b.to(device), y_b.to(device)
                pred = model(x_b)
                val_losses.append(criterion(pred, y_b).item())

        return np.mean(val_losses)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

    print(f"[+] Optuna Best Parameters Found:")
    for k, v in study.best_params.items():
        print(f"    - {k}: {v}")
    return study.best_params

# ------------------------------------------------------------------
# 6. DASHBOARD GENERATION
# ------------------------------------------------------------------
def generate_walk_forward_dashboard(df_raw, pred_df, metrics):
    start_str = df_raw['date'].min().strftime('%b %Y')
    end_str = df_raw['date'].max().strftime('%b %Y')

    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.75, 0.25],
        specs=[
            [{"type": "xy"}, {"type": "table", "rowspan": 3}],
            [{"type": "xy"}, None],
            [{"type": "xy"}, None]
        ],
        subplot_titles=(
            f"Actual Daily OHLC ({start_str} - {end_str})",
            "",
            f"Predicted Daily OHLC (After 512 Days Warmup)",
            "Actual - Predicted Close (Residuals)"
        ),
        vertical_spacing=0.07,
        horizontal_spacing=0.04
    )

    fig.add_trace(
        go.Candlestick(
            x=df_raw['date'], open=df_raw['open'], high=df_raw['high'],
            low=df_raw['low'], close=df_raw['close'], name='Actual OHLC',
            increasing_line_color='#00c853', decreasing_line_color='#ff1744'
        ), row=1, col=1
    )

    warmup_date = df_raw.loc[WINDOW_SIZE, 'date']
    for r in range(1, 4):
        fig.add_vline(
            x=warmup_date, line_dash="dash", line_color="#ff9800", line_width=2,
            annotation_text="512-Day Warmup End" if r == 1 else "",
            annotation_position="top left", row=r, col=1
        )

    fig.add_trace(
        go.Candlestick(
            x=pred_df['date'], open=pred_df['pred_open'], high=pred_df['pred_high'],
            low=pred_df['pred_low'], close=pred_df['pred_close'], name='Predicted OHLC',
            increasing_line_color='#00c853', decreasing_line_color='#ff1744'
        ), row=2, col=1
    )

    residuals = pred_df['actual_close'] - pred_df['pred_close']
    fig.add_trace(
        go.Scatter(x=pred_df['date'], y=residuals, mode='lines', name='Residuals', line=dict(color='#9c27b0', width=1)),
        row=3, col=1
    )

    keys = list(metrics.keys())
    vals = [str(v) for v in metrics.values()]
    fig.add_trace(
        go.Table(
            header=dict(values=["<b>Metric</b>", "<b>Value</b>"], fill_color='#0f172a', font=dict(color='white', size=12), align='left'),
            cells=dict(values=[keys, vals], fill_color='#f8fafc', font=dict(color='black', size=11), align='left', height=26)
        ), row=1, col=2
    )

    fig.update_layout(
        title_text=f"<b>Optuna-Tuned LoRA Kronos: NIFTY 50 Prediction (Jan 2022 - Sept 2026)</b>",
        height=950,
        template="plotly_white",
        showlegend=False,
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False
    )

    fig.write_html(OUTPUT_HTML)
    print(f"\n[+] Dashboard with Optuna results generated: {OUTPUT_HTML}")

# ------------------------------------------------------------------
# 7. MAIN EXECUTION
# ------------------------------------------------------------------
def main():
    df_raw = load_parquet_data()
    ret_cols = ['open_ret', 'high_ret', 'low_ret', 'close_ret', 'vol_ret']
    returns_arr = df_raw[ret_cols].values

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(returns_arr[:WINDOW_SIZE])

    # Run Optuna Hyperparameter Optimization
    best_params = optimize_hyperparameters(train_scaled)

    # Train Final LoRA Model with Best Hyperparameters
    dataset = ReturnDataset(train_scaled, seq_len=60)
    loader = DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True)

    model = build_lora_model(
        lora_r=best_params['lora_r'],
        lora_alpha=best_params['lora_alpha'],
        lora_dropout=best_params['lora_dropout']
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'])
    criterion = nn.MSELoss()

    model.train()
    print("[+] Fine-tuning final model with best Optuna hyperparameters on warmup window...")
    for epoch in range(12):
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()

    model.eval()
    pred_records = []
    print(f"[+] Rolling 512-day prediction from step {WINDOW_SIZE} to {len(df_raw)} (Sept 2026)...")

    for idx in range(WINDOW_SIZE, len(df_raw)):
        window_data = returns_arr[idx - WINDOW_SIZE : idx]
        
        w_scaler = StandardScaler()
        w_scaled = w_scaler.fit_transform(window_data)
        
        x_seq = w_scaled[-60:]
        input_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_scaled_ret = model(input_tensor).cpu().numpy()[0]
            dummy = np.zeros((1, 5))
            dummy[0, 3] = pred_scaled_ret
            pred_ret = w_scaler.inverse_transform(dummy)[0, 3]

        prev_close = df_raw.loc[idx - 1, 'close']
        actual_close = df_raw.loc[idx, 'close']
        actual_open = df_raw.loc[idx, 'open']
        actual_high = df_raw.loc[idx, 'high']
        actual_low = df_raw.loc[idx, 'low']
        
        p_close = prev_close * np.exp(pred_ret)
        p_open = prev_close * np.exp(df_raw.loc[idx, 'open_ret'])
        p_high = p_open * (actual_high / max(actual_open, 1e-5))
        p_low = p_open * (actual_low / max(actual_open, 1e-5))

        pred_records.append({
            'date': df_raw.loc[idx, 'date'],
            'actual_close': actual_close,
            'pred_close': p_close,
            'pred_open': p_open,
            'pred_high': max(p_high, p_open, p_close),
            'pred_low': min(p_low, p_open, p_close)
        })

    pred_df = pd.DataFrame(pred_records)

    act = pred_df['actual_close'].values
    prd = pred_df['pred_close'].values
    err = act - prd

    rmse = np.sqrt(np.mean(err ** 2))
    mae = np.mean(np.abs(err))
    mape = np.mean(np.abs(err / act)) * 100
    
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((act - np.mean(act)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    dir_acc = np.mean((np.diff(act) * np.diff(prd)) > 0) * 100
    pearson_corr, _ = stats.pearsonr(act, prd)

    metrics = {
        "Optuna LR": f"{best_params['lr']:.5f}",
        "Optuna LoRA Rank (r)": f"{best_params['lora_r']}",
        "Optuna Batch Size": f"{best_params['batch_size']}",
        "Evaluated Days": f"{len(pred_df)} Days",
        "RMSE": f"{rmse:.2f}",
        "MAE": f"{mae:.2f}",
        "MAPE": f"{mape:.2f}%",
        "R²": f"{r2:.4f}",
        "Direction Accuracy": f"{dir_acc:.2f}%",
        "Pearson Correlation": f"{pearson_corr:.4f}"
    }

    generate_walk_forward_dashboard(df_raw, pred_df, metrics)

if __name__ == "__main__":
    main()