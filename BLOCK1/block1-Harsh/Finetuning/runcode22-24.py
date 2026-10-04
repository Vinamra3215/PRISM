import os
import sys
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Add current repository root to Python path
sys.path.insert(0, os.path.abspath("."))

# ============================================================
# 1. LOAD KRONOS FOUNDATION MODEL & TOKENIZER
# ============================================================
print("\n========================================")
print("LOADING KRONOS FOUNDATION MODEL & TOKENIZER")
print("========================================")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Inference Device: {device}")

try:
    from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    
    # 1. Load Tokenizer from Hugging Face Hub
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    
    # 2. Load Base Model from Hugging Face Hub
    model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    
    # 3. Instantiate Predictor Wrapper
    predictor = KronosPredictor(model=model, tokenizer=tokenizer, device=device, max_context=512)
    print("Kronos-base and Tokenizer loaded successfully.")

except Exception as e:
    print(f"\n[CRITICAL ERROR] Failed to initialize Kronos model pipeline: {e}")
    sys.exit(1)

# ============================================================
# 2. FETCH & PREPROCESS LIVE 2022–2024 RELIANCE DATA
# ============================================================
symbol = "RELIANCE.NS"
start_date = "2022-01-01"
end_date = "2024-12-31"

print(f"\nFetching data for {symbol} ({start_date} to {end_date})...")
df = yf.download(symbol, start=start_date, end=end_date, progress=False)

if df.empty:
    print(f"[ERROR] No data returned for {symbol}.")
    sys.exit(1)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()
df.rename(columns={
    'Date': 'timestamps',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}, inplace=True)

df['timestamps'] = pd.to_datetime(df['timestamps'])
# Kronos expects open, high, low, close (and optional volume)
df = df.dropna(subset=['open', 'high', 'low', 'close']).sort_values('timestamps').reset_index(drop=True)

# Context and Forecast Settings
lookback_len = 400
pred_len = 120  # Forecast 30 trading days

if len(df) <= (lookback_len + pred_len):
    x_df = df.iloc[:-pred_len].copy().reset_index(drop=True)
    actual_df = df.iloc[-pred_len:].copy().reset_index(drop=True)
else:
    x_df = df.iloc[-(lookback_len + pred_len):-pred_len].copy().reset_index(drop=True)
    actual_df = df.iloc[-pred_len:].copy().reset_index(drop=True)

x_timestamp = x_df['timestamps']
y_timestamp = actual_df['timestamps']
y_true = actual_df['close'].values

print(f"Context Window (Lookback) : {len(x_df)} candles")
print(f"Forecast Window (Horizon) : {len(actual_df)} candles ({y_timestamp.iloc[0].strftime('%Y-%m-%d')} to {y_timestamp.iloc[-1].strftime('%Y-%m-%d')})")

# ============================================================
# 3. RUN REAL KRONOS NEURAL NETWORK INFERENCE
# ============================================================
print("\n========================================")
print("RUNNING REAL MODEL PREDICTION...")
print("========================================")

try:
    pred_df = predictor.predict(
        df=x_df, 
        x_timestamp=x_timestamp, 
        y_timestamp=y_timestamp, 
        pred_len=pred_len,
        T=1.0,           # Temperature (1.0 = standard sampling)
        top_p=0.9,       # Nucleus sampling probability
        sample_count=1   # Number of forecast paths
    )
    
    # Extract predicted close prices from the real model output
    y_pred = pred_df['close'].values
    print("Inference completed successfully.")

except Exception as e:
    print(f"\n[ERROR] Kronos forward pass failed: {e}")
    sys.exit(1)

y_pred = np.array(y_pred).flatten()[:pred_len]

# ============================================================
# 4. COMPUTE REAL PERFORMANCE & DIVERGENCE METRICS
# ============================================================
absolute_errors = np.abs(y_true - y_pred)
mape_daily = (absolute_errors / y_true) * 100

max_div_idx = np.argmax(absolute_errors)
max_div_date = y_timestamp.iloc[max_div_idx].strftime('%Y-%m-%d')
max_div_val_true = y_true[max_div_idx]
max_div_val_pred = y_pred[max_div_idx]
max_error_amount = absolute_errors[max_div_idx]

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
mape = np.mean(mape_daily)

print("\n========================================")
print("ACTUAL KRONOS MODEL PERFORMANCE METRICS")
print("========================================")
print(f"RMSE : {rmse:.2f} ₹")
print(f"MAE  : {mae:.2f} ₹")
print(f"MAPE : {mape:.2f}%")
print(f"Max Divergence Date: {max_div_date} (Error: ₹{max_error_amount:.2f})")

# ============================================================
# 5. RENDER INTERACTIVE DIAGNOSTIC PLOT IN PLOTLY
# ============================================================
fig = go.Figure()

# 1. Ground Truth Line
fig.add_trace(go.Scatter(
    x=y_timestamp,
    y=y_true,
    mode="lines+markers",
    name="Actual Ground Truth (₹)",
    line=dict(color="#38bdf8", width=2.5),
    marker=dict(size=5)
))

# 2. Real Kronos Prediction Line
fig.add_trace(go.Scatter(
    x=y_timestamp,
    y=y_pred,
    mode="lines+markers",
    name="Kronos Actual Prediction (₹)",
    line=dict(color="#a855f7", width=2.5, dash="dash"),
    marker=dict(size=5)
))

# 3. Highlight Maximum Divergence Point
fig.add_trace(go.Scatter(
    x=[y_timestamp.iloc[max_div_idx]],
    y=[max_div_val_pred],
    mode="markers+text",
    name=f"Max Divergence (Δ = ₹{max_error_amount:.2f})",
    marker=dict(color="#ef4444", size=12, symbol="x"),
    text=[f"  Max Divergence<br>  Δ ₹{max_error_amount:.2f}"],
    textposition="top right"
))

# Dotted connector line showing divergence gap
fig.add_shape(
    type="line",
    x0=y_timestamp.iloc[max_div_idx],
    y0=max_div_val_true,
    x1=y_timestamp.iloc[max_div_idx],
    y1=max_div_val_pred,
    line=dict(color="#ef4444", width=2, dash="dot")
)

fig.update_layout(
    title=dict(
        text=f"{symbol} — Real Kronos Prediction vs Ground Truth (2022–2024 Horizon)",
        x=0.5,
        xanchor="center",
        font=dict(size=18)
    ),
    template="plotly_dark",
    height=720,
    autosize=True,
    margin=dict(l=80, r=80, t=100, b=80),
    yaxis=dict(
        title="<b>Stock Close Price</b> (₹)",
        showgrid=True,
        gridcolor="#334155"
    ),
    xaxis=dict(
        title="Forecast Date",
        showgrid=True,
        gridcolor="#334155"
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),
    annotations=[
        dict(
            x=0.01,
            y=0.95,
            xref="paper",
            yref="paper",
            text=f"<b>Real Model Metrics:</b><br>• RMSE: ₹{rmse:.2f}<br>• MAE: ₹{mae:.2f}<br>• MAPE: {mape:.2f}%",
            showarrow=False,
            bgcolor="#1e293b",
            bordercolor="#475569",
            borderwidth=1,
            align="left"
        )
    ],
    hovermode="x unified"
)

# Display plot interactively in Plotly (browser / Jupyter notebook)
fig.show()