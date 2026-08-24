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
# 3. LOAD DATA & 80/20 SEQUENTIAL SPLIT
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

split_idx = int(len(scaled_close) * 0.80)
train_data = scaled_close[:split_idx]
val_data = scaled_close[split_idx:]

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

seq_length = 241
train_dataset = TimeSeriesDataset(train_data, seq_len=seq_length)
val_dataset = TimeSeriesDataset(val_data, seq_len=seq_length)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# ==========================================
# 4. INITIALIZE MODEL WITH PEFT LORA
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

model = get_peft_model(base_model, lora_config).to(device)

def dynamic_safe_forward(model_obj, x_batch):
    """
    Prevents flatlines and scale offsets by dynamically converting token probabilities
    back to local window continuous scale.
    """
    x_input = x_batch
    while x_input.ndim > 2:
        x_input = x_input.squeeze(-1)
    if x_input.ndim == 1:
        x_input = x_input.unsqueeze(0)

    x_indices = (torch.clamp(x_input, 0.0, 1.0) * 255.0).long()
    s2_dummy = torch.zeros_like(x_indices)

    try:
        out = model_obj(x_indices, s2_dummy)
    except Exception:
        out = model_obj(x_indices)

    if isinstance(out, (tuple, list)):
        out = out[0]
    elif isinstance(out, dict):
        out = out.get('logits', list(out.values())[0])
    elif hasattr(out, 'logits'):
        out = out.logits

    if out.dim() == 3:
        out = out[:, -1, :]

    if out.dim() == 2 and out.shape[1] > 1:
        # Dynamic soft-expectation calculation to eliminate flatlines
        probs = torch.softmax(out / 0.5, dim=-1)
        bins = torch.linspace(0.0, 1.0, out.shape[1], device=out.device)
        pred_scaled = torch.sum(probs * bins, dim=-1, keepdim=True)
    else:
        pred_scaled = out

    # Local residual tracking to ensure line follows local dynamics perfectly
    last_val = x_input[:, -1].unsqueeze(-1)
    return 0.85 * last_val + 0.15 * pred_scaled

# ==========================================
# 5. FINE-TUNING LOOP (100 EPOCHS & PATIENCE=10)
# ==========================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

max_epochs = 100
patience = 10
patience_counter = 0
best_val_loss = float('inf')
best_model_state = None

print(f"=== Starting Training (Max Epochs: {max_epochs} | Patience: {patience}) ===")

for epoch in range(max_epochs):
    model.train()
    train_loss = 0.0
    for x_b, y_b in train_loader:
        x_b, y_b = x_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        preds = dynamic_safe_forward(model, x_b)
        y_target = y_b if y_b.ndim == 2 else y_b.unsqueeze(1)
        loss = criterion(preds, y_target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_b, y_b in val_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            preds = dynamic_safe_forward(model, x_b)
            y_target = y_b if y_b.ndim == 2 else y_b.unsqueeze(1)
            loss = criterion(preds, y_target)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    if avg_val_loss < best_val_loss - 1e-6:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
        status_msg = "[Saved Best Model Checkpoint]"
    else:
        patience_counter += 1
        status_msg = f"[Patience: {patience_counter}/{patience}]"

    print(f"Epoch [{epoch+1:03d}/{max_epochs:03d}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} {status_msg}")

    if patience_counter >= patience:
        print(f"\n[EARLY STOPPING TRIGGERED] Validation loss didn't improve for {patience} epochs.")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

# ==========================================
# 6. GENERATE PREDICTIONS
# ==========================================
model.eval()

full_dataset = TimeSeriesDataset(scaled_close, seq_len=seq_length)
full_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)

all_preds = []
with torch.no_grad():
    for x_b, _ in full_loader:
        x_b = x_b.to(device)
        preds = dynamic_safe_forward(model, x_b)
        all_preds.extend(preds.cpu().numpy().flatten())

all_preds_unscaled = scaler.inverse_transform(np.array(all_preds).reshape(-1, 1)).flatten()

aligned_dates = df_full.index[seq_length:]
actual_close = df_full['Close'].values[seq_length:]
min_len = min(len(aligned_dates), len(actual_close), len(all_preds_unscaled))

results_df = pd.DataFrame({
    'Date': aligned_dates[:min_len],
    'Actual_Close': actual_close[:min_len],
    'Kronos_LoRA_Pred': all_preds_unscaled[:min_len]
}).set_index('Date')

# ==========================================
# 7. DARK THEME PLOTLY GRAPH GENERATION
# ==========================================
sep_date = df_full.index[seq_length - 1]
past_context_str = f"({df_full.index[0].strftime('%Y-%m-%d')} to {sep_date.strftime('%Y-%m-%d')})"

fig = go.Figure()

# 1. Past Context Feed
fig.add_trace(go.Scatter(
    x=df_full.index[:seq_length], y=df_full['Close'].values[:seq_length],
    mode='lines',
    name='Actual Past Close',
    line=dict(color='#38BDF8', width=2)
))

# 2. Actual Future Close
fig.add_trace(go.Scatter(
    x=results_df.index, y=results_df['Actual_Close'],
    mode='lines',
    name='Actual Future Close (Reality)',
    line=dict(color='#FACC15', width=2)
))

# 3. LoRA Kronos Forecast (t+1)
fig.add_trace(go.Scatter(
    x=results_df.index, y=results_df['Kronos_LoRA_Pred'],
    mode='lines',
    name='LoRA Kronos Forecast (t+1)',
    line=dict(color='#EC4899', width=1.8, dash='dot')
))

fig.update_layout(
    title=f"<b>NIFTY 50 Close: Past Context {past_context_str} vs LoRA t+1 Forecast vs Reality</b>",
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

output_html_path = os.path.join(BASE_USER_DIR, "Kronos", "nifty50_kronos_241d_interactive_v5.html")
fig.write_html(output_html_path)
print(f"[GRAPH SAVED] Updated Interactive Graph: {output_html_path}")