import os
import sys
import glob
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =========================================================
# CONFIGURATION & LORA CONFIG DATACLASS
# =========================================================
@dataclass
class LoRAConfig:
    r: int = 8                    # Low-rank dimension
    alpha: float = 16.0           # LoRA scaling factor
    dropout: float = 0.05         # Dropout rate on adapter input
    learning_rate: float = 0.005  # Fine-tuning rate
    epochs: int = 15              # Fine-tuning epochs per window step

DATA_FILE = "NIFTY50_2022_to_today.parquet"
WINDOW_SIZE = 375                 # 375-day sliding window context
LOOKBACK = 30                     # Lagged log-return inputs
STEP_SIZE = 1                     # Step size = 1 day
TARGET_END_DATE = pd.to_datetime("2026-07-31")

PRETRAIN_EPOCHS = 150
DIVERGENCE_PATIENCE = 15
MIN_DELTA = 1e-4

# Interactive Plotly Line Chart Output
HTML_FILE = "nifty_375day_sliding_window_normalized.html"

lora_config = LoRAConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Active Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# =========================================================
# 1. DEFINE LORA MODULE & ARCHITECTURE
# =========================================================
class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, config: LoRAConfig):
        super().__init__()
        self.scaling = config.alpha / config.r
        
        # Base Linear Layer (W_0)
        self.linear = nn.Linear(in_features, out_features)
        
        # Adapter layers & Dropout
        self.dropout = nn.Dropout(config.dropout)
        self.lora_A = nn.Parameter(torch.randn(config.r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))

    def freeze_base_weights(self):
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.linear(x)
        lora_output = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output

class LoRAModel(nn.Module):
    def __init__(self, lookback: int, config: LoRAConfig):
        super().__init__()
        self.layer1 = LoRALinear(lookback, 64, config)
        self.relu = nn.ReLU()
        self.layer2 = LoRALinear(64, 1, config)

    def freeze_base(self):
        self.layer1.freeze_base_weights()
        self.layer2.freeze_base_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.layer1(x))
        return self.layer2(x)

# =========================================================
# 2. LOAD & CLEAN DATA
# =========================================================
possible_paths = [
    DATA_FILE,
    "/home/soq/NIFTY50_5Y_OHLCV.parquet",
    os.path.abspath("NIFTY50_5Y_OHLCV.parquet"),
    os.path.abspath("data/NIFTY50_5Y_OHLCV.parquet"),
    "/home/soq/Kronos/NIFTY50_5Y_OHLCV.parquet",
    "/home/soq/Kronos/data/NIFTY50_5Y_OHLCV.parquet"
]

data_path = None
for path in possible_paths:
    if os.path.exists(path):
        data_path = path
        break

if not data_path:
    matches = glob.glob("/home/soq/**/NIFTY*.parquet", recursive=True) + glob.glob("./**/NIFTY*.parquet", recursive=True)
    if matches:
        data_path = matches[0]

if not data_path:
    raise FileNotFoundError("Could not find NIFTY 50 parquet dataset.")

print(f"Loaded Dataset: {data_path}")
df = pd.read_parquet(data_path)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] if isinstance(col, tuple) else str(col) for col in df.columns]

date_col = next((c for c in df.columns if str(c).lower() in ["date", "timestamps", "timestamp", "datetime"]), df.columns[0])
close_col = next((c for c in df.columns if str(c).lower() in ["close", "adj close"]), None)

if close_col is None:
    raise ValueError(f"Close column missing. Found: {df.columns.tolist()}")

df["Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.tz_localize(None)
df["Close"] = pd.to_numeric(df[close_col].astype(str).str.replace(",", ""), errors="coerce")

df = df[["Date", "Close"]].dropna().sort_values("Date").drop_duplicates("Date").reset_index(drop=True)

# Context window anchor
start_idx_2022 = df[df["Date"] >= "2022-01-01"].index
start_idx = start_idx_2022[0] if len(start_idx_2022) > 0 else WINDOW_SIZE
if start_idx < WINDOW_SIZE:
    start_idx = WINDOW_SIZE

print(f"Total Records       : {len(df)}")
print(f"Window Size         : {WINDOW_SIZE} trading days")
print(f"Lookback Context    : {LOOKBACK} trading days")
print(f"Initial Context End : {df['Date'].iloc[start_idx - 1].date()}")
print(f"Prediction Start    : {df['Date'].iloc[start_idx].date()} (Day 376)")
print(f"Target Horizon End  : {TARGET_END_DATE.date()}")

# =========================================================
# 3. BASE MODEL PRE-TRAINING WITH DIVERGENCE EARLY STOPPING
# =========================================================
prices = df["Close"].values
dates = df["Date"].values

baseline_prices = prices[:start_idx]
baseline_log_returns = np.diff(np.log(baseline_prices))

init_samples = sliding_window_view(baseline_log_returns, window_shape=LOOKBACK + 1)
split_idx = max(int(len(init_samples) * 0.85), 2)

X_train_init = torch.tensor(init_samples[:split_idx, :-1], dtype=torch.float32).to(device)
y_train_init = torch.tensor(init_samples[:split_idx, -1], dtype=torch.float32).unsqueeze(1).to(device)

X_val_init = torch.tensor(init_samples[split_idx:, :-1], dtype=torch.float32).to(device)
y_val_init = torch.tensor(init_samples[split_idx:, -1], dtype=torch.float32).unsqueeze(1).to(device)

model = LoRAModel(lookback=LOOKBACK, config=lora_config).to(device)
pretrain_optimizer = optim.Adam(model.parameters(), lr=0.008, weight_decay=1e-4)
criterion = nn.MSELoss()

best_val_loss = float("inf")
divergence_counter = 0
prev_train_loss, prev_val_loss = None, None
best_model_state = None

print("\n" + "=" * 70)
print("PRE-TRAINING BASE WEIGHTS W_0 WITH DIVERGENCE EARLY STOPPING")
print("=" * 70)

for epoch in range(1, PRETRAIN_EPOCHS + 1):
    model.train()
    pretrain_optimizer.zero_grad()
    loss = criterion(model(X_train_init), y_train_init)
    loss.backward()
    pretrain_optimizer.step()
    curr_train_loss = loss.item()

    model.eval()
    with torch.no_grad():
        curr_val_loss = criterion(model(X_val_init), y_val_init).item() if len(X_val_init) > 0 else curr_train_loss

    if curr_val_loss < (best_val_loss - MIN_DELTA):
        best_val_loss = curr_val_loss
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if prev_train_loss is not None and prev_val_loss is not None:
        if curr_train_loss < prev_train_loss and curr_val_loss >= (prev_val_loss - MIN_DELTA):
            divergence_counter += 1
        else:
            divergence_counter = 0

    prev_train_loss, prev_val_loss = curr_train_loss, curr_val_loss

    if epoch % 10 == 0 or divergence_counter > 0:
        print(f"Epoch [{epoch:03d}/{PRETRAIN_EPOCHS:03d}] ── Train Loss: {curr_train_loss:.6f} | Val Loss: {curr_val_loss:.6f} | Divergence: {divergence_counter}/{DIVERGENCE_PATIENCE}")

    if divergence_counter >= DIVERGENCE_PATIENCE:
        print(f"Early stopping triggered at epoch {epoch} due to divergence.")
        break

if best_model_state:
    model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})

model.freeze_base()

# =========================================================
# 4. WALK-FORWARD 375-DAY SLIDING LORA INFERENCE
# =========================================================
print("\n" + "=" * 70)
print("EXECUTING 375-DAY SLIDING WINDOW FORECAST TO JULY 2026")
print("=" * 70)

available_post_start = df.iloc[start_idx:].copy().reset_index(drop=True)
last_available_date = df["Date"].iloc[-1]

if last_available_date < TARGET_END_DATE:
    extended_dates = pd.date_range(start=last_available_date + pd.offsets.BDay(1), end=TARGET_END_DATE, freq="B")
else:
    extended_dates = pd.DatetimeIndex([])

full_rollout_dates = list(available_post_start["Date"]) + list(extended_dates)
total_steps = len(full_rollout_dates)

current_prices = list(df["Close"].iloc[:start_idx].values)
actual_prices = []
predicted_prices = []
prediction_dates = []

for i in range(total_steps):
    target_date = full_rollout_dates[i]
    
    window_prices = np.array(current_prices[-WINDOW_SIZE:], dtype=float)
    window_log_returns = np.diff(np.log(window_prices))
    
    window_samples = sliding_window_view(window_log_returns, window_shape=LOOKBACK + 1)
    
    X_window = torch.tensor(window_samples[:, :-1], dtype=torch.float32).to(device)
    y_window = torch.tensor(window_samples[:, -1], dtype=torch.float32).unsqueeze(1).to(device)
    
    X_step_test = torch.tensor(window_log_returns[-LOOKBACK:], dtype=torch.float32).unsqueeze(0).to(device)
    
    adapter_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lora_config.learning_rate)
    
    model.train()
    for _ in range(lora_config.epochs):
        adapter_optimizer.zero_grad()
        loss = criterion(model(X_window), y_window)
        loss.backward()
        adapter_optimizer.step()
        
    model.eval()
    with torch.no_grad():
        pred_log_ret = model(X_step_test).item()
        
    last_price = current_prices[-1]
    pred_price = last_price * np.exp(pred_log_ret)
    
    predicted_prices.append(pred_price)
    prediction_dates.append(target_date)
    
    if i < len(available_post_start):
        real_price = available_post_start["Close"].iloc[i]
        actual_prices.append(real_price)
        current_prices.append(real_price)
    else:
        actual_prices.append(np.nan)
        current_prices.append(pred_price)
        
    if (i + 1) % 100 == 0 or (i + 1) == total_steps:
        print(f"Progress: [{i + 1}/{total_steps}] sessions | Current Date: {pd.Timestamp(target_date).date()}")

actual_prices = np.array(actual_prices)
predicted_prices = np.array(predicted_prices)
prediction_dates = pd.to_datetime(prediction_dates)

# =========================================================
# 5. EVALUATION METRICS
# =========================================================
valid_mask = ~np.isnan(actual_prices)
if np.sum(valid_mask) > 0:
    eval_act = actual_prices[valid_mask]
    eval_pred = predicted_prices[valid_mask]
    
    mae = mean_absolute_error(eval_act, eval_pred)
    rmse = np.sqrt(mean_squared_error(eval_act, eval_pred))
    mape = np.mean(np.abs((eval_act - eval_pred) / eval_act)) * 100
else:
    mae, rmse, mape = 0.0, 0.0, 0.0

print("\n" + "=" * 70)
print("EVALUATION METRICS")
print("=" * 70)
print(f"MAE  : {mae:.2f} points")
print(f"RMSE : {rmse:.2f} points")
print(f"MAPE : {mape:.2f}%")

# =========================================================
# 6. BASE-100 NORMALIZATION & DUAL-PANEL LINE GRAPH
# =========================================================
base_price = df["Close"].iloc[start_idx]

context_dates = df["Date"].iloc[:start_idx].values
context_norm = (df["Close"].iloc[:start_idx].values / base_price) * 100

actual_norm = np.full_like(actual_prices, np.nan)
actual_norm[valid_mask] = (actual_prices[valid_mask] / base_price) * 100

pred_norm = (predicted_prices / base_price) * 100
abs_error = np.full_like(actual_prices, np.nan)
abs_error[valid_mask] = np.abs(actual_prices[valid_mask] - predicted_prices[valid_mask])

# Create Subplots: Top = Normalized Price Lines, Bottom = Error Area Line
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.70, 0.30],
    subplot_titles=(
        "NIFTY 50: 375-Day Sliding Window LoRA Forecast (Base 100 = Jan 2022)",
        "Absolute Prediction Error (Original NIFTY Points)"
    )
)

# 1. Historical Context Line
fig.add_trace(
    go.Scattergl(
        x=context_dates,
        y=context_norm,
        mode="lines",
        name="375-Day Context Baseline",
        line=dict(width=1.8, color="#64748b"),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Context Norm:</b> %{y:.2f}<extra></extra>"
    ),
    row=1, col=1
)

# 2. Actual Market Outcome Line
if np.sum(valid_mask) > 0:
    fig.add_trace(
        go.Scattergl(
            x=prediction_dates[valid_mask],
            y=actual_norm[valid_mask],
            mode="lines",
            name="Actual Market Outcome (Base 100)",
            line=dict(width=2.2, color="#38bdf8"),
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual Norm:</b> %{y:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

# 3. LoRA Forecasted Line
fig.add_trace(
    go.Scattergl(
        x=prediction_dates,
        y=pred_norm,
        mode="lines",
        name="LoRA Sliding Forecast (Base 100)",
        line=dict(width=2.5, dash="dash", color="#c084fc"),
        hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Forecast Norm:</b> %{y:.2f}<extra></extra>"
    ),
    row=1, col=1
)

# 4. Reference Horizontal Baseline (Base 100 Anchor)
fig.add_hline(
    y=100,
    line_dash="dot",
    line_color="#94a3b8",
    line_width=1,
    row=1, col=1
)

# 5. Bottom Panel: Absolute Error Line Plot
if np.sum(valid_mask) > 0:
    fig.add_trace(
        go.Scattergl(
            x=prediction_dates[valid_mask],
            y=abs_error[valid_mask],
            mode="lines",
            name="Absolute Error (Pts)",
            line=dict(width=1.2, color="#f87171"),
            fill="tozeroy",
            hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Error:</b> %{y:.2f} pts<extra></extra>"
        ),
        row=2, col=1
    )

fig.update_layout(
    title=dict(
        text=(
            "<b>NIFTY50: 375-Day Sliding Window LoRA Forecast (Jan 2022 – Jul 2026)</b><br>"
            f"<sup>LoRA Rank: {lora_config.r} | Alpha: {lora_config.alpha} | Window: {WINDOW_SIZE} Days | "
            f"MAE: {mae:.2f} pts | RMSE: {rmse:.2f} pts | MAPE: {mape:.2f}%</sup>"
        ),
        x=0.02,
        xanchor="left",
        font=dict(size=18, color="#f8fafc")
    ),
    template="plotly_dark",
    height=860,
    autosize=True,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(l=70, r=40, t=110, b=60)
)

fig.update_yaxes(
    title_text="<b>Normalized Price (Base 100)</b>",
    showgrid=True,
    gridcolor="#334155",
    row=1, col=1
)

fig.update_yaxes(
    title_text="<b>Absolute Error (Pts)</b>",
    showgrid=True,
    gridcolor="#334155",
    row=2, col=1
)

fig.update_xaxes(
    title_text="Date",
    showgrid=True,
    gridcolor="#334155",
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=[
            dict(count=1, label="1M", step="month", stepmode="backward"),
            dict(count=3, label="3M", step="month", stepmode="backward"),
            dict(count=6, label="6M", step="month", stepmode="backward"),
            dict(count=1, label="1Y", step="year", stepmode="backward"),
            dict(step="all", label="All")
        ],
        bgcolor="#1e293b",
        activecolor="#2563eb",
        font=dict(color="#f8fafc", size=11)
    ),
    row=2, col=1
)

# Export interactive line chart directly to HTML
fig.write_html(HTML_FILE, include_plotlyjs=True)
print(f"\n[SUCCESS] Interactive line chart generated: '{HTML_FILE}'")
print("=" * 70)