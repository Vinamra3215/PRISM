import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from timesfm3 import TimesFM3Evaluator, ModelConfig

# ==========================================
# 1. CONFIGURATION & DATA LOADING
# ==========================================
PARQUET_FILE_PATH = "/home/soq/__shutupandbendover/nifty50 lora all/nifty50 lora 118/NIFTY50_OHLCV/RELIANCE_OHLCV.parquet" 
TARGET_COL = "Close"                      

# Load data
df = pd.read_parquet(PARQUET_FILE_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Split into context (Jan 2024 to Nov 2025) and evaluation (Dec 2025 to Jan 16, 2026)
train_df = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2025-11-30')].copy()
test_df = df[(df['Date'] >= '2025-12-01') & (df['Date'] <= '2026-01-16')].copy()

if train_df.empty or test_df.empty:
    raise ValueError("Check dataset date ranges: Train or Test split is empty.")

horizon = len(test_df)
context_prices = train_df[TARGET_COL].values.astype(np.float32)
y_true = test_df[TARGET_COL].values.astype(np.float32)

# ==========================================
# 2. LOCAL ZERO-SHOT FORECASTING
# ==========================================
# Using "." assumes you ran `huggingface-cli download ... --local-dir .` in this folder.
# This prevents requests to the HF Hub and uses the local model.safetensors and config.json.
config = ModelConfig(
    checkpoint_path="/home/soq/__shutupandbendover/_JohnMiller/TimesFM3", 
    per_core_batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu"
)
tfm = TimesFM3Evaluator(config)

# Run zero-shot inference
outputs = list(tfm.predict_batch(
    contexts=[context_prices],
    horizon=horizon,
    return_quantiles=True,
    use_symmetric_averaging=False
))

# Extract the median point forecast
y_pred = outputs[0].forecast

# ==========================================
# 3. METRICS CALCULATION
# ==========================================
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
mae = np.mean(np.abs(y_true - y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
r2 = r2_score(y_true, y_pred)

# Directional Accuracy (DA)
last_context_price = train_df[TARGET_COL].iloc[-1]
diff_true = np.diff(np.insert(y_true, 0, last_context_price))
diff_pred = np.diff(np.insert(y_pred, 0, last_context_price))
directional_accuracy = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100

# Return (% change)
actual_return = ((y_true[-1] - last_context_price) / last_context_price) * 100
pred_return = ((y_pred[-1] - last_context_price) / last_context_price) * 100

# Pearson Correlation
pearson_corr, _ = pearsonr(y_true, y_pred)

# ==========================================
# 4. ROBUST PLOTLY VISUALIZATION
# ==========================================
# Create a 2-row layout: Top for the chart, Bottom for the metrics table
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=False,
    vertical_spacing=0.12,
    row_heights=[0.75, 0.25], 
    specs=[[{"type": "scatter"}],
           [{"type": "table"}]]
)

# --- Add Graph Traces (Row 1) ---
recent_context = train_df.iloc[-60:] # Last 60 days of train for visual continuity
fig.add_trace(go.Scatter(
    x=recent_context['Date'], y=recent_context[TARGET_COL],
    mode='lines', name='Historical Context (Pre-Dec 2025)',
    line=dict(color='gray', width=1.5)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=test_df['Date'], y=y_true,
    mode='lines+markers', name='Actual Price',
    line=dict(color='#2E7D32', width=2.5), marker=dict(size=5)
), row=1, col=1)

fig.add_trace(go.Scatter(
    x=test_df['Date'], y=y_pred,
    mode='lines+markers', name='TimesFM 3.0 Forecast',
    line=dict(color='#D32F2F', width=2.5, dash='dash'), marker=dict(size=5)
), row=1, col=1)

# --- Add Metrics Table (Row 2) ---
fig.add_trace(go.Table(
    header=dict(
        values=["<b>Metric</b>", "<b>Value</b>", "<b>Metric</b>", "<b>Value</b>"],
        fill_color='paleturquoise',
        align='left',
        font=dict(size=12, color='black')
    ),
    cells=dict(
        values=[
            ["RMSE", "MAE", "MAPE", "R² Score"],
            [f"{rmse:.2f}", f"{mae:.2f}", f"{mape:.2f}%", f"{r2:.4f}"],
            ["Directional Accuracy", "Actual Return", "Predicted Return", "Pearson Corr"],
            [f"{directional_accuracy:.2f}%", f"{actual_return:+.2f}%", f"{pred_return:+.2f}%", f"{pearson_corr:.4f}"]
        ],
        fill_color='lavender',
        align='left',
        font=dict(size=12, color='black'),
        height=30
    )
), row=2, col=1)

# Update layout to look clean
fig.update_layout(
    title="<b>TimesFM 3.0 Zero-Shot: RELIANCE</b>",
    height=800,  # Ensure enough vertical space for both chart and table
    hovermode="x unified",
    template="plotly_white",
    legend=dict(
        orientation="h",
        yanchor="bottom", y=1.02,
        xanchor="right", x=1
    ),
    margin=dict(l=40, r=40, t=80, b=40)
)

fig.update_yaxes(title_text="Price", row=1, col=1)

fig.show()