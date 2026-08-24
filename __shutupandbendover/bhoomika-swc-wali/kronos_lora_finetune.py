import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 1. PATHS & CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_HTML_NAME = f"kronos_241d_rolling_eval_{timestamp}.html"
OUTPUT_HTML = os.path.join(BASE_DIR, OUTPUT_HTML_NAME)
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "kronos_rolling_lora_weights.pt")

WINDOW_SIZE = 241      # 241 Days Context Window
LORA_RANK = 8
LORA_ALPHA = 16
LEARNING_RATE = 2e-4   # Balanced online LR
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[*] Executing on Device: {DEVICE}")

# ==========================================
# 2. DATA LOADING & CLEANING
# ==========================================
def load_full_nifty_data():
    parquet_path = os.path.join(BASE_DIR, "Kronos", "data", "NIFTY50_5Y_OHLCV.parquet")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    
    possible_files = glob.glob("/home/soq/**/market data*.csv", recursive=True) + \
                     glob.glob("/home/soq/**/NIFTY50*.parquet", recursive=True)
    if possible_files:
        f = possible_files[0]
        return pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
    return None

df = load_full_nifty_data()
if df is None:
    raise FileNotFoundError("Full NIFTY Data File Nahi Mili!")

df = df.reset_index()
df.columns = [str(c).strip() for c in df.columns]

date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
close_col = next((c for c in df.columns if 'close' in c.lower()), None)

df = df.rename(columns={date_col: 'Date', close_col: 'Close'})
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
    found = next((c for c in df.columns if c.lower() == col.lower()), None)
    if found:
        df[found] = pd.to_numeric(df[found].astype(str).str.replace(',', '').str.strip(), errors='coerce')

df = df.dropna(subset=['Close']).reset_index(drop=True)

features = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df.columns]
raw_matrix = df[features].values
close_idx = features.index('Close')
total_rows = len(df)

# ==========================================
# 3. LORA ARCHITECTURE
# ==========================================
class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, r=8, lora_alpha=16):
        super().__init__()
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.weight = original_layer.weight
        self.bias = original_layer.bias

        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = nn.functional.linear(x, self.weight, self.bias)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out

class KronosBaseBackbone(nn.Module):
    def __init__(self, feature_dim=5, d_model=128, nhead=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(feature_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):
        embeds = self.input_proj(x)
        out = self.transformer_encoder(embeds)
        last_step = out[:, -1, :]
        pred = self.head(last_step)
        return pred.squeeze(-1)

def inject_lora_safely(model, r=8, alpha=16):
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Linear):
            setattr(model, name, LoRALinear(module, r=r, lora_alpha=alpha))
        else:
            inject_lora_safely(module, r=r, alpha=alpha)

model = KronosBaseBackbone(feature_dim=len(features))
inject_lora_safely(model, r=LORA_RANK, alpha=LORA_ALPHA)
model.to(DEVICE)

lora_trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(lora_trainable_params, lr=LEARNING_RATE, weight_decay=1e-4)
criterion = nn.MSELoss()

# ==========================================
# 4. WALK-FORWARD ROLLING WITH ONLINE WEIGHT UPDATES
# ==========================================
all_predictions = [np.nan] * WINDOW_SIZE

print("\n" + "="*60)
print("   RUNNING REAL-TIME WALK-FORWARD ROLLING FINE-TUNING")
print("="*60)

for i in range(WINDOW_SIZE, total_rows):
    # Dynamic Window Normalization (Window Mean/Std Scale)
    window_data = raw_matrix[i - WINDOW_SIZE : i].copy()
    mean_vec = np.mean(window_data, axis=0)
    std_vec = np.std(window_data, axis=0) + 1e-8
    
    norm_window = (window_data - mean_vec) / std_vec
    
    actual_target = raw_matrix[i, close_idx]
    norm_target = (actual_target - mean_vec[close_idx]) / std_vec[close_idx]
    
    x_tensor = torch.tensor(norm_window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    y_tensor = torch.tensor([norm_target], dtype=torch.float32).to(DEVICE)
    
    # 1. PREDICT Day 242
    model.eval()
    with torch.no_grad():
        pred_norm = model(x_tensor).item()
        # Un-scale to actual rupee price
        pred_actual = (pred_norm * std_vec[close_idx]) + mean_vec[close_idx]
        all_predictions.append(pred_actual)
        
    # 2. UPDATE WEIGHTS IMMEDIATELY AFTER PREDICTION (Online LoRA Training Step)
    model.train()
    optimizer.zero_grad()
    out = model(x_tensor)
    loss = criterion(out, y_tensor)
    loss.backward()
    optimizer.step()

    if i % 150 == 0 or i == total_rows - 1:
        print(f"[{i}/{total_rows}] Date: {df['Date'].iloc[i].strftime('%Y-%m-%d')} | Step Loss: {loss.item():.5f} | Pred: ₹{pred_actual:,.2f} | Actual: ₹{actual_target:,.2f}")

df['Rolling_LoRA_Pred'] = all_predictions

# ==========================================
# 5. VISUALIZATION
# ==========================================
df_feed = df.iloc[:WINDOW_SIZE].copy().reset_index(drop=True)
df_future = df.iloc[WINDOW_SIZE:].copy().reset_index(drop=True)

cutoff_date = df['Date'].iloc[WINDOW_SIZE - 1]

fig = go.Figure()

# 1. Initial 241 Days Feed Line
fig.add_trace(go.Scatter(
    x=df_feed['Date'], y=df_feed['Close'],
    name="Initial 241-Day Feed Context",
    mode='lines', line=dict(color='#38BDF8', width=2),
    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Feed Price:</b> ₹%{y:,.2f}<extra></extra>"
))

# 2. Actual Market Price (After 241st Day)
fig.add_trace(go.Scatter(
    x=df_future['Date'], y=df_future['Close'],
    name="Actual Market Price",
    mode='lines', line=dict(color='#F8FAFC', width=2),
    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Actual:</b> ₹%{y:,.2f}<extra></extra>"
))

# 3. LoRA Fine-Tuned Online Rolling Prediction
fig.add_trace(go.Scatter(
    x=df_future['Date'], y=df_future['Rolling_LoRA_Pred'],
    name="LoRA Fine-Tuned Prediction (Next Day)",
    mode='lines', line=dict(color='#F43F5E', width=2.2, dash='dash'),
    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>LoRA Pred:</b> ₹%{y:,.2f}<extra></extra>"
))

# Boundary Line
fig.add_vline(
    x=cutoff_date.timestamp() * 1000, 
    line_width=2, line_dash="dash", line_color="#F59E0B",
    annotation_text="End of 241 Days Feed (Prediction Boundary)", 
    annotation_position="top left",
    annotation_font=dict(color="#F59E0B", size=12)
)

fig.update_layout(
    title=dict(
        text="<b>Kronos: Walk-Forward Online Fine-Tuned Rolling Model</b>",
        font=dict(size=15, color="#FFFFFF")
    ),
    template="plotly_dark", paper_bgcolor="#0A0E14", plot_bgcolor="#0A0E14", hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#FFFFFF")),
    yaxis=dict(title=dict(text="<b>NIFTY Price (₹)</b>", font=dict(color="#FFFFFF", size=13)), gridcolor="#1E2638", autorange=True),
    xaxis=dict(title=dict(text="<b>Time / Date Axis</b>", font=dict(color="#FFFFFF")), type="date", gridcolor="#1E2638",
               rangeslider=dict(visible=True, thickness=0.12, bgcolor="#0F172A", bordercolor="#1E293B")),
    margin=dict(l=50, r=50, t=80, b=50)
)

fig.write_html(OUTPUT_HTML, include_plotlyjs=True, full_html=True, auto_open=False)

print("\n" + "="*60)
print(f"SUCCESS! Walk-Forward Online Learning Plot Generated.")
print(f"File -> {OUTPUT_HTML_NAME}")
print("="*60 + "\n")