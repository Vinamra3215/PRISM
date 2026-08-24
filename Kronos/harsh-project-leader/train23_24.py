import math
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. SETUP & LOCAL PARQUET DATA LOADING
# ==========================================
DATA_FILE = "NIFTY50_2022_to_today.parquet"
TRAIN_START = "2023-01-01"
TRAIN_END = "2024-12-31"
TEST_START = "2025-01-01"
TEST_END = "2026-07-31"
SAVE_PATH = "./best_weights"
GRAPH_FILE = "performancegraph.png"
SEQ_LEN = 30

if not os.path.exists(DATA_FILE):
    if os.path.exists("NIFTY50_2022_to_day.parquet"):
        DATA_FILE = "NIFTY50_2022_to_day.parquet"
    else:
        raise FileNotFoundError(f"Parquet file '{DATA_FILE}' not found in directory.")

df_raw = pd.read_parquet(DATA_FILE)

# Parse Date with format="mixed" to eliminate UserWarning
date_col = "Date" if "Date" in df_raw.columns else df_raw.columns[0]
df_raw["Date"] = pd.to_datetime(df_raw[date_col], format="mixed", errors="coerce")

# Locate Close price column
close_col = "Close" if "Close" in df_raw.columns else ("Adj Close" if "Adj Close" in df_raw.columns else None)
if not close_col:
    raise ValueError(f"Close price column not found in {df_raw.columns.tolist()}")

df_raw[close_col] = pd.to_numeric(df_raw[close_col], errors="coerce")

# Clean and structure dataset
df = df_raw[["Date", close_col]].dropna().sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
df.set_index("Date", inplace=True)
df.rename(columns={close_col: "Close"}, inplace=True)

# Stationarity & Momentum Technical Features
df["Returns"] = df["Close"].pct_change().fillna(0.0)
df["MA5_Ratio"] = (df["Close"] / df["Close"].rolling(5).mean()) - 1.0
df["MA20_Ratio"] = (df["Close"] / df["Close"].rolling(20).mean()) - 1.0
df.dropna(inplace=True)

feature_cols = ["Returns", "MA5_Ratio", "MA20_Ratio"]

# Split Dataset
train_df = df[(df.index >= TRAIN_START) & (df.index <= TRAIN_END)]
test_df = df[(df.index >= TEST_START) & (df.index <= TEST_END)]

if len(train_df) <= SEQ_LEN:
    raise ValueError(f"Insufficient training samples ({len(train_df)}). Need > {SEQ_LEN}.")

test_dates = test_df.index
test_series = test_df['Close'].values

# Standardize Features
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_df[feature_cols].values)
test_scaled = scaler.transform(test_df[feature_cols].values)

class MultiFeatureDataset(Dataset):
    def __init__(self, data, seq_len=SEQ_LEN):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        # x is (seq_len, num_features), target y is the next day's scaled Return (col 0)
        return self.data[idx : idx + self.seq_len], self.data[idx + self.seq_len, 0]

train_dataset = MultiFeatureDataset(train_scaled, seq_len=SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ==========================================
# 2. LORA ADAPTER DEFINITION
# ==========================================
class LoRALinear(nn.Module):
    """Low-Rank Adaptation (LoRA) module with PyTorch Linear interface parity."""
    def __init__(self, original_linear: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.original_linear = original_linear
        for param in self.original_linear.parameters():
            param.requires_grad = False
            
        in_dim, out_dim = original_linear.in_features, original_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros((in_dim, rank)))
        self.lora_B = nn.Parameter(torch.zeros((rank, out_dim)))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def weight(self): return self.original_linear.weight
    @property
    def bias(self): return self.original_linear.bias
    @property
    def in_features(self): return self.original_linear.in_features
    @property
    def out_features(self): return self.original_linear.out_features

    def forward(self, x):
        return self.original_linear(x) + ((x @ self.lora_A @ self.lora_B) * self.scaling)

# ==========================================
# 3. TRANSFORMER MODEL ARCHITECTURE
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim=len(feature_cols), hidden_dim=64, seq_len=SEQ_LEN):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=seq_len + 10)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        return self.regressor(x[:, -1, :]).squeeze(-1)

def apply_lora_to_model(model: nn.Module, rank: int = 4, alpha: float = 8.0):
    for param in model.parameters():
        param.requires_grad = False
    for layer in model.transformer.layers:
        layer.linear1 = LoRALinear(layer.linear1, rank=rank, alpha=alpha)
        layer.linear2 = LoRALinear(layer.linear2, rank=rank, alpha=alpha)
    model.regressor = LoRALinear(model.regressor, rank=rank, alpha=alpha)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TimeSeriesTransformer(input_dim=len(feature_cols))
model = apply_lora_to_model(model, rank=4, alpha=8.0).to(device)

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"Total Model Parameters    : {sum(p.numel() for p in model.parameters())}")
print(f"Trainable LoRA Parameters : {sum(p.numel() for p in trainable_params)}")

criterion = nn.HuberLoss()  # Huber Loss provides robustness against market anomaly spikes
optimizer = torch.optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

# Fine-Tuning Loop
epochs = 30
best_loss = float("inf")

print("\nStarting LoRA Fine-Tuning Loop...")
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x_batch.size(0)
        
    epoch_loss = total_loss / len(train_dataset)
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        os.makedirs(SAVE_PATH, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, "best_weights.pt"))
        
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch {epoch+1:02d}/{epochs} - Huber Loss: {epoch_loss:.6f}")

# ==========================================
# 4. ROLLING PREDICTION & RECONSTRUCTION
# ==========================================
model.load_state_dict(torch.load(os.path.join(SAVE_PATH, "best_weights.pt")))
model.eval()

full_features = np.vstack([train_scaled[-SEQ_LEN:], test_scaled])
predicted_returns_scaled = []

with torch.no_grad():
    for i in range(len(test_scaled)):
        input_seq = torch.tensor(full_features[i : i + SEQ_LEN], dtype=torch.float32).unsqueeze(0).to(device)
        pred_scaled = model(input_seq).item()
        predicted_returns_scaled.append(pred_scaled)

# Inverse scaling specifically for return column (Col 0)
pred_returns = (np.array(predicted_returns_scaled) * scaler.scale_[0]) + scaler.mean_[0]

# Reconstruct NIFTY price targets from predicted returns: P(t) = P(t-1) * (1 + return_pred)
predictions_actual = []
for i in range(len(test_series)):
    prev_price = train_df['Close'].iloc[-1] if i == 0 else test_series[i - 1]
    pred_price = prev_price * (1.0 + pred_returns[i])
    predictions_actual.append(pred_price)

predictions_actual = np.array(predictions_actual)

# ==========================================
# 5. METRIC EVALUATION
# ==========================================
mae = np.mean(np.abs(predictions_actual - test_series))
rmse = np.sqrt(np.mean((predictions_actual - test_series) ** 2))
mape = np.mean(np.abs((test_series - predictions_actual) / test_series)) * 100
dir_acc = np.mean(np.sign(test_df['Returns'].values[1:]) == np.sign(pred_returns[1:])) * 100

print("\n" + "="*45)
print("     UN-LAGGED MODEL EVALUATION (RETURNS)     ")
print("="*45)
print(f" Mean Absolute Error (MAE)     : {mae:.2f} points")
print(f" Root Mean Squared Error (RMSE): {rmse:.2f} points")
print(f" Mean Absolute Percentage Error: {mape:.2f}%")
print(f" Directional Accuracy          : {dir_acc:.2f}%")
print("="*45 + "\n")

residual_errors = predictions_actual - test_series

# ==========================================
# 6. PLOTLY VISUALIZATION
# ==========================================
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    subplot_titles=(
        "<b>NIFTY 50 - Actual Price vs Feature-Aware Prediction</b>", 
        "<b>Residual Error (Original Points)</b>"
    ),
    row_heights=[0.7, 0.3]
)

# Actual NIFTY
fig.add_trace(go.Scattergl(
    x=test_dates, y=test_series, mode='lines',
    name='Actual NIFTY', line=dict(color='#1f77b4', width=2),
    hovertemplate="<b>Date:</b> %{x|%d-%b-%Y}<br><b>Actual:</b> %{y:.2f}<extra></extra>"
), row=1, col=1)

# Predicted NIFTY
fig.add_trace(go.Scattergl(
    x=test_dates, y=predictions_actual, mode='lines',
    name='Predicted NIFTY', line=dict(color='#ff7f0e', width=1.8),
    hovertemplate="<b>Date:</b> %{x|%d-%b-%Y}<br><b>Predicted:</b> %{y:.2f}<extra></extra>"
), row=1, col=1)

# Residuals
fig.add_trace(go.Scattergl(
    x=test_dates, y=residual_errors, mode='lines',
    name='Residual Error', line=dict(color='#d62728', width=1.2), fill='tozeroy',
    hovertemplate="<b>Date:</b> %{x|%d-%b-%Y}<br><b>Error:</b> %{y:.2f} pts<extra></extra>"
), row=2, col=1)

fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5, row=2, col=1)

fig.update_layout(
    height=800, template="plotly_white", hovermode="x unified",
    legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.8)'),
    margin=dict(l=60, r=30, t=80, b=50)
)
fig.update_yaxes(title_text="NIFTY Index Points", row=1, col=1)
fig.update_yaxes(title_text="Error (Points)", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)

fig.show()

try:
    fig.write_image(GRAPH_FILE, width=1200, height=750)
    print(f"Performance graph saved to '{GRAPH_FILE}'.")
except Exception as e:
    print(f"Image export skipped: {e}")