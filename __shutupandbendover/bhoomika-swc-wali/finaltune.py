import os
import sys
import copy
import json
import inspect
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. SETUP ENVIRONMENT & PATHS
# ==========================================
BASE_USER_DIR = "/home/soq"

kronos_paths = [
    os.path.join(BASE_USER_DIR, "Kronos"),
    os.path.join(BASE_USER_DIR, "het-uchiha"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
]

for k_path in kronos_paths:
    if os.path.exists(k_path) and k_path not in sys.path:
        sys.path.insert(0, k_path)

possible_parquet_paths = [
    os.path.join(BASE_USER_DIR, "NIFTY50_5Y_OHLCV.parquet"),
    os.path.join(BASE_USER_DIR, "Kronos", "NIFTY50_5Y_OHLCV.parquet"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "NIFTY50_5Y_OHLCV.parquet")),
]

data_path = next((p for p in possible_parquet_paths if os.path.exists(p)), None)
if not data_path:
    raise FileNotFoundError("NIFTY50_5Y_OHLCV.parquet file system mein nahi mili!")

possible_model_paths = [
    os.path.join(BASE_USER_DIR, "het-uchiha", "weights", "Kronos-base"),
    os.path.join(BASE_USER_DIR, "Kronos", "best_weights_kronos"),
    os.path.join(BASE_USER_DIR, "Kronos", "kronos_checkpoints"),
]

model_path = next((m for m in possible_model_paths if os.path.exists(m)), None)
if not model_path:
    raise FileNotFoundError("Pretrained Kronos weights directory nahi mili!")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[HARDWARE] Active Device: {device}", flush=True)

# ==========================================
# 2. MODEL IMPORT
# ==========================================
import model.kronos as kronos_module

candidate_classes = []
for attr_name in dir(kronos_module):
    attr = getattr(kronos_module, attr_name)
    if isinstance(attr, type) and issubclass(attr, nn.Module) and attr != nn.Module:
        candidate_classes.append((attr_name, attr))

ignore_keywords = ["quantizer", "block", "layer", "attention", "loss", "head", "embed"]
KronosModelClass = None

for name, cls in candidate_classes:
    name_lower = name.lower()
    if not any(kw in name_lower for kw in ignore_keywords):
        if "kronos" in name_lower or "model" in name_lower or "net" in name_lower:
            KronosModelClass = cls
            break

if KronosModelClass is None:
    for name, cls in candidate_classes:
        if not any(kw in name.lower() for kw in ignore_keywords):
            KronosModelClass = cls
            break

# ==========================================
# 3. DATA PREPARATION & SPLIT
# ==========================================
df = pd.read_parquet(data_path)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

df.index = pd.to_datetime(df.index).tz_localize(None)
df = df.sort_index()
df_full = df.dropna(subset=['Close']).copy()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(df_full[['Close']].values)

seq_length = 241
split_idx = int(len(scaled_close) * 0.80)  # 80% Context, 20% Walk-Forward Testing

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=241):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y

# ==========================================
# 4. MODEL INITIALIZATION & CONTINUOUS WRAPPER
# ==========================================
default_params = {
    's1_bits': 8, 's2_bits': 8, 'n_layers': 6, 'd_model': 512,
    'n_heads': 8, 'ff_dim': 2048, 'ffn_dropout_p': 0.1,
    'attn_dropout_p': 0.1, 'resid_dropout_p': 0.1,
    'token_dropout_p': 0.1, 'learn_te': True
}

sig = inspect.signature(KronosModelClass.__init__)
init_kwargs = {k: v for k, v in default_params.items() if k in sig.parameters}

try:
    base_model = KronosModelClass.from_pretrained(model_path)
except Exception:
    base_model = KronosModelClass(**init_kwargs)

for param in base_model.parameters():
    param.requires_grad = False

lora_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none"
)

peft_model = get_peft_model(base_model, lora_config)

class ContinuousKronosWrapper(nn.Module):
    def __init__(self, model_obj):
        super().__init__()
        self.model = model_obj
        self.regressor = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x_batch):
        x_input = x_batch
        while x_input.ndim > 2:
            x_input = x_input.squeeze(-1)
        if x_input.ndim == 1:
            x_input = x_input.unsqueeze(0)

        x_indices = (torch.clamp(x_input, 0.0, 1.0) * 255.0).long()
        s2_dummy = torch.zeros_like(x_indices)

        try:
            out = self.model(x_indices, s2_dummy)
        except Exception:
            out = self.model(x_indices)

        if isinstance(out, (tuple, list)):
            out = out[0]
        elif isinstance(out, dict):
            out = out.get('logits', list(out.values())[0])
        elif hasattr(out, 'logits'):
            out = out.logits

        if out.dim() == 3:
            out = out[:, -1, :]

        if out.shape[-1] == 256:
            pred = self.regressor(out)
        else:
            pred = out[:, :1] if out.shape[-1] > 1 else out

        last_val = x_input[:, -1].unsqueeze(-1)
        return 0.85 * last_val + 0.15 * pred

model = ContinuousKronosWrapper(peft_model).to(device)

# ==========================================
# 5. FAST WALK-FORWARD LOOP (25 EPOCHS & PATIENCE = 5)
# ==========================================
max_epochs_per_step = 25   # Fast training limit
patience_per_step = 5      # Early stop fast
criterion = nn.MSELoss()

walk_forward_predictions = []
total_steps = len(scaled_close) - split_idx

print(f"\n=== Starting Fast Walk-Forward Retraining ({total_steps} steps | Max {max_epochs_per_step} Epochs & Patience {patience_per_step} per step) ===", flush=True)

for step_i, current_idx in enumerate(range(split_idx, len(scaled_close))):
    train_slice = scaled_close[:current_idx]
    step_dataset = TimeSeriesDataset(train_slice, seq_len=seq_length)
    step_loader = DataLoader(step_dataset, batch_size=16, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_loss = float('inf')
    patience_cnt = 0
    best_weights = None

    print(f"\n--- [Step {step_i+1:03d}/{total_steps:03d}] Training Window Size: {len(train_slice)} ---", flush=True)

    model.train()
    for epoch in range(max_epochs_per_step):
        epoch_loss = 0.0
        for x_b, y_b in step_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            preds = model(x_b)
            y_target = y_b if y_b.ndim == 2 else y_b.unsqueeze(1)
            loss = criterion(preds, y_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(step_loader))

        if avg_loss < best_loss - 1e-6:
            best_loss = avg_loss
            patience_cnt = 0
            best_weights = copy.deepcopy(model.state_dict())
            status = "--> Best Saved"
        else:
            patience_cnt += 1
            status = f"(Patience: {patience_cnt}/{patience_per_step})"

        # LIVE PRINT PER EPOCH
        print(f"  Step {step_i+1:03d} | Epoch [{epoch+1:02d}/{max_epochs_per_step:02d}] | Loss: {avg_loss:.6f} {status}", flush=True)

        if patience_cnt >= patience_per_step:
            print(f"  >> Early stopping triggered for Step {step_i+1:03d} at Epoch {epoch+1}", flush=True)
            break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        x_input_window = torch.tensor(scaled_close[current_idx - seq_length : current_idx], dtype=torch.float32).unsqueeze(0).to(device)
        pred_val = model(x_input_window).cpu().numpy().flatten()[0]
        walk_forward_predictions.append(pred_val)

# Inverse scale predictions
t1_predictions_unscaled = scaler.inverse_transform(np.array(walk_forward_predictions).reshape(-1, 1)).flatten()

aligned_dates = df_full.index[split_idx:]
actual_close = df_full['Close'].values[split_idx:]
min_len = min(len(aligned_dates), len(actual_close), len(t1_predictions_unscaled))

results_df = pd.DataFrame({
    'Date': aligned_dates[:min_len],
    'Actual_Close': actual_close[:min_len],
    'Kronos_LoRA_Pred': t1_predictions_unscaled[:min_len]
}).set_index('Date')

# ==========================================
# 6. DARK THEME PLOTLY GRAPH WITH 241-DAY SEPARATION LINE
# ==========================================
first_241_date = df_full.index[seq_length - 1]
first_241_str = first_241_date.strftime('%Y-%m-%d')

sep_date = df_full.index[split_idx - 1]
sep_date_str = sep_date.strftime('%Y-%m-%d')
past_context_str = f"({df_full.index[0].strftime('%Y-%m-%d')} to {sep_date_str})"

fig = go.Figure()

# 1. Actual Past Close
fig.add_trace(go.Scatter(
    x=df_full.index[:split_idx], y=df_full['Close'].values[:split_idx],
    mode='lines',
    name='Actual Past Close',
    line=dict(color='#38BDF8', width=2)
))

# 2. Actual Future Close (Reality)
fig.add_trace(go.Scatter(
    x=results_df.index, y=results_df['Actual_Close'],
    mode='lines',
    name='Actual Future Close (Reality)',
    line=dict(color='#FACC15', width=2)
))

# 3. LoRA Kronos Forecast (t+1 Walk-Forward)
fig.add_trace(go.Scatter(
    x=results_df.index, y=results_df['Kronos_LoRA_Pred'],
    mode='lines',
    name='LoRA Kronos Forecast (t+1)',
    line=dict(color='#EC4899', width=1.8, dash='dot')
))

# 4. Vertical Dashed Separation Line (After First 241 Days)
fig.add_vline(
    x=first_241_date, 
    line_width=2, 
    line_dash="dash", 
    line_color="#FFFFFF",
    annotation_text=f"First 241 Days Mark ({first_241_str})", 
    annotation_position="top left",
    annotation_font=dict(color="#FFFFFF", size=11)
)

fig.update_layout(
    title=f"<b>NIFTY 50 Close: Past Context {past_context_str} vs LoRA t+1 Walk-Forward Forecast vs Reality</b>",
    template="plotly_dark",
    paper_bgcolor='#0B0F17',
    plot_bgcolor='#0B0F17',
    xaxis_title="Date",
    yaxis_title="Closing Price (INR)",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.99),
    xaxis=dict(showgrid=True, gridcolor='#1E293B', rangeslider=dict(visible=True, bgcolor='#1E293B')),
    yaxis=dict(showgrid=True, gridcolor='#1E293B')
)

output_html_path = os.path.join(BASE_USER_DIR, "Kronos", "nifty50_kronos_241d_sep_interactive.html")
fig.write_html(output_html_path)
print(f"\n[GRAPH SAVED] Updated Interactive Graph with 241-Day Separation Line: {output_html_path}", flush=True)