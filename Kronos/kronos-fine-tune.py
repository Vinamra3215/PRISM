import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from peft import LoraConfig, get_peft_model, PeftModel

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# =========================================================
# CONFIGURATION & HYPERPARAMETERS
# =========================================================
DATA_FILE = "NIFTY50_2022_to_today.parquet"
WINDOW_SIZE = 375          # 375-day lookback window context
STEP_SIZE = 1              # Step size 1 session per prediction
TARGET_END_DATE = pd.to_datetime("2026-07-31")

MAX_EPOCHS = 150
DIVERGENCE_PATIENCE = 15   # Early stopping if train drops while val stagnates/increases for 15 loops
MIN_DELTA = 1e-4

CSV_FILE = "nifty_375day_lora_predictions_normalized.csv"
HTML_FILE = "nifty_375day_sliding_window_normalized.html"

# =========================================================
# 1. SETUP GPU HARDWARE & LOAD FOUNDATION KRONOS MODEL
# =========================================================
print("=" * 70)
print("1. INITIALIZING GPU & KRONOS LORA MODEL")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Active Compute Device: {device} ({gpu_name})")

try:
    from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    if hasattr(tokenizer, 'to'):
        tokenizer = tokenizer.to(device)
        
    base_model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    
    # Identify linear projection modules for LoRA adapter attachment
    target_modules = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            layer_name = name.split(".")[-1]
            if layer_name in ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2", "proj", "linear"]:
                target_modules.append(layer_name)
    target_modules = list(set(target_modules))
    if not target_modules:
        target_modules = ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    print("\nLoRA Adapter Configuration:")
    model.print_trainable_parameters()
except Exception as e:
    print(f"[CRITICAL ERROR] Kronos setup failed: {e}")
    sys.exit(1)

# =========================================================
# 2. LOAD & CLEAN DATASET (WITH STRING-TO-FLOAT CASTING)
# =========================================================
print("\n" + "=" * 70)
print("2. LOADING DATASET")
print("=" * 70)

possible_paths = [
    DATA_FILE,
    "/home/soq/NIFTY50_5Y_OHLCV.parquet",
    os.path.abspath("NIFTY50_5Y_OHLCV.parquet"),
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
raw_df = pd.read_parquet(data_path)

if isinstance(raw_df.columns, pd.MultiIndex):
    raw_df.columns = [col[0] if isinstance(col, tuple) else str(col) for col in raw_df.columns]

raw_df.columns = [str(c).strip().lower() for c in raw_df.columns]

# Parse Timestamps
if isinstance(raw_df.index, pd.DatetimeIndex):
    raw_df = raw_df.reset_index()
    raw_df.rename(columns={raw_df.columns[0]: 'timestamps'}, inplace=True)
else:
    time_col = None
    for col in ['timestamps', 'timestamp', 'date', 'datetime', 'time', 'index']:
        if col in raw_df.columns:
            time_col = col
            break
    if time_col:
        raw_df.rename(columns={time_col: 'timestamps'}, inplace=True)
    else:
        raw_df['timestamps'] = pd.date_range(end=pd.Timestamp.today(), periods=len(raw_df), freq='B')

raw_df['timestamps'] = pd.to_datetime(raw_df['timestamps'], errors='coerce').dt.tz_localize(None)

# Standardize OHLC column names
for target in ['open', 'high', 'low', 'close']:
    if target not in raw_df.columns:
        matching = [c for c in raw_df.columns if target in c]
        if matching:
            raw_df.rename(columns={matching[0]: target}, inplace=True)

if 'volume' not in raw_df.columns:
    vol_match = [c for c in raw_df.columns if 'vol' in c]
    raw_df['volume'] = raw_df[vol_match[0]] if vol_match else 1.0

# -------------------------------------------------------------
# Explicit numeric type cast (Fixes string multiplication bug)
# -------------------------------------------------------------
for col in ['open', 'high', 'low', 'close', 'volume']:
    if col in raw_df.columns:
        raw_df[col] = pd.to_numeric(raw_df[col].astype(str).str.replace(',', ''), errors='coerce')

# Resample to Daily if high-frequency
if len(raw_df) > 5000:
    raw_df.set_index('timestamps', inplace=True)
    df = raw_df.resample('D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna().reset_index()
else:
    df = raw_df.reset_index(drop=True)

# Fill Missing/Zero Values & Generate Amount safely
df['volume'] = df['volume'].replace(0, 1.0).fillna(1.0).astype(float)
df['close'] = df['close'].ffill().bfill().astype(float)
df['open'] = df['open'].fillna(df['close']).astype(float)
df['high'] = df['high'].fillna(df['close']).astype(float)
df['low'] = df['low'].fillna(df['close']).astype(float)

if 'amount' not in df.columns and 'turnover' not in df.columns:
    df['amount'] = (df['close'] * df['volume']).astype(float)
elif 'turnover' in df.columns:
    df['amount'] = pd.to_numeric(df['turnover'], errors='coerce').fillna(df['close'] * df['volume']).astype(float)
else:
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(df['close'] * df['volume']).astype(float)

df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'amount']).sort_values('timestamps').drop_duplicates(subset=['timestamps']).reset_index(drop=True)

start_idx = WINDOW_SIZE

print(f"Total Records       : {len(df)}")
print(f"Window Size         : {WINDOW_SIZE} trading sessions")
print(f"Step Size           : {STEP_SIZE} session(s)")
print(f"Initial Context End : {df['timestamps'].iloc[start_idx - 1].date()}")
print(f"Prediction Start    : {df['timestamps'].iloc[start_idx].date()} (Day 376)")
print(f"Horizon End         : {TARGET_END_DATE.date()}")

# =========================================================
# 3. TOKEN EXTRACTION & FORWARD PASS HELPERS
# =========================================================
feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
inner_model = model.get_base_model() if hasattr(model, "get_base_model") else model
vocab_size_s1 = getattr(inner_model, 's1_vocab_size', 1024)
vocab_size_s2 = getattr(inner_model, 's2_vocab_size', 1024)
if hasattr(inner_model, 's1_embed'):
    vocab_size_s1 = inner_model.s1_embed.num_embeddings
if hasattr(inner_model, 's2_embed'):
    vocab_size_s2 = inner_model.s2_embed.num_embeddings

def get_tokens(data_df):
    raw = data_df[feature_cols].values.astype(np.float32)
    m = np.mean(raw, axis=0, keepdims=True)
    s = np.std(raw, axis=0, keepdims=True) + 1e-8
    norm = (raw - m) / s
    tensor_in = torch.tensor(norm, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        enc = tokenizer.encode(tensor_in)
    
    if isinstance(enc, tuple) and len(enc) == 2:
        s1, s2 = enc
    elif hasattr(enc, 's1_ids') and hasattr(enc, 's2_ids'):
        s1, s2 = enc.s1_ids, enc.s2_ids
    elif isinstance(enc, torch.Tensor) and enc.ndim == 3 and enc.shape[-1] == 2:
        s1, s2 = enc[..., 0], enc[..., 1]
    else:
        s1 = enc if isinstance(enc, torch.Tensor) else torch.tensor(enc, device=device)
        s2 = torch.zeros_like(s1)
        
    s1 = torch.clamp(s1.to(device=device, dtype=torch.long), 0, vocab_size_s1 - 1)
    s2 = torch.clamp(s2.to(device=device, dtype=torch.long), 0, vocab_size_s2 - 1)
    if s1.ndim == 1:
        s1 = s1.unsqueeze(0)
        s2 = s2.unsqueeze(0)
    return s1, s2

def forward_step(s1, s2):
    out = model(s1, s2)
    if isinstance(out, tuple):
        out_s1, out_s2 = out[0], out[1]
    elif hasattr(out, 'logits_s1') and hasattr(out, 'logits_s2'):
        out_s1, out_s2 = out.logits_s1, out.logits_s2
    elif hasattr(out, 'logits'):
        out_s1 = out.logits
        out_s2 = None
    else:
        out_s1 = out
        out_s2 = None
    return out_s1, out_s2

criterion = nn.CrossEntropyLoss()

# =========================================================
# 4. LORA FINE-TUNING ON BASELINE CONTEXT
# =========================================================
print("\n" + "=" * 70)
print("3. TRAINING LORA ADAPTERS ON GPU")
print("=" * 70)

initial_train_df = df.iloc[:start_idx].copy().reset_index(drop=True)
s1_tokens, s2_tokens = get_tokens(initial_train_df)
seq_len = s1_tokens.shape[1]
split_idx = max(int(seq_len * 0.85), 2)

s1_train_x, s1_train_y = s1_tokens[:, :split_idx-1], s1_tokens[:, 1:split_idx]
s2_train_x, s2_train_y = s2_tokens[:, :split_idx-1], s2_tokens[:, 1:split_idx]

s1_val_x, s1_val_y = s1_tokens[:, split_idx-1:-1], s1_tokens[:, split_idx:]
s2_val_x, s2_val_y = s2_tokens[:, split_idx-1:-1], s2_tokens[:, split_idx:]

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2)
checkpoint_dir = os.path.abspath("best_nifty_lora_sliding")
best_val_loss = float("inf")
best_epoch = 0
divergence_counter = 0

prev_train_loss = None
prev_val_loss = None

print(f"Starting LoRA fine-tuning (Max Epochs: {MAX_EPOCHS}, Divergence Patience: {DIVERGENCE_PATIENCE} loops)...")

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    optimizer.zero_grad()
    
    out_s1, out_s2 = forward_step(s1_train_x, s2_train_x)
    loss = criterion(out_s1.view(-1, out_s1.size(-1)), s1_train_y.contiguous().view(-1))
    if out_s2 is not None:
        loss += criterion(out_s2.view(-1, out_s2.size(-1)), s2_train_y.contiguous().view(-1))
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    current_train_loss = loss.item()
    
    # Validation step
    model.eval()
    with torch.no_grad():
        if s1_val_x.shape[1] > 0 and s1_val_y.shape[1] > 0:
            val_s1, val_s2 = forward_step(s1_val_x, s2_val_x)
            current_val_loss = criterion(val_s1.view(-1, val_s1.size(-1)), s1_val_y.contiguous().view(-1)).item()
            if val_s2 is not None:
                current_val_loss += criterion(val_s2.view(-1, val_s2.size(-1)), s2_val_y.contiguous().view(-1)).item()
        else:
            current_val_loss = current_train_loss

    if current_val_loss < (best_val_loss - MIN_DELTA):
        best_val_loss = current_val_loss
        best_epoch = epoch
        model.save_pretrained(checkpoint_dir)
        marker = " ◄ [BEST SAVED]"
    else:
        marker = ""

    # Divergence rule check
    if prev_train_loss is not None and prev_val_loss is not None:
        train_loss_decreased = (current_train_loss < prev_train_loss)
        val_loss_stagnant_or_increased = (current_val_loss >= prev_val_loss - MIN_DELTA)
        
        if train_loss_decreased and val_loss_stagnant_or_increased:
            divergence_counter += 1
            marker += f" | Divergence: {divergence_counter}/{DIVERGENCE_PATIENCE}"
        else:
            divergence_counter = 0

    prev_train_loss = current_train_loss
    prev_val_loss = current_val_loss

    if epoch % 5 == 0 or epoch == 1 or "BEST" in marker or divergence_counter > 0:
        print(f"Epoch [{epoch:03d}/{MAX_EPOCHS:03d}] ── Train Loss: {current_train_loss:.4f} | Val Loss: {current_val_loss:.4f}{marker}")

    if divergence_counter >= DIVERGENCE_PATIENCE:
        print(f"\n[EARLY STOPPING TRIGGERED] Divergence detected for {DIVERGENCE_PATIENCE} consecutive loops.")
        break

# Restore best checkpoint weights
model = PeftModel.from_pretrained(base_model, checkpoint_dir)
print(f"\nRestored Best LoRA Checkpoint from Epoch {best_epoch:03d} (Best Val Loss: {best_val_loss:.4f})")

# =========================================================
# 5. STEP-SIZE 1 SLIDING WINDOW ROLLOUT (JAN 2022 -> JULY 2026)
# =========================================================
print("\n" + "=" * 70)
print("4. RUNNING 375-DAY SLIDING WINDOW TRANSFORMER PREDICTIONS (STEP=1)")
print("=" * 70)

model.eval()
predictor = KronosPredictor(model=model, tokenizer=tokenizer, device=device, max_context=512)

available_post_start = df.iloc[start_idx:].copy().reset_index(drop=True)
last_available_date = df['timestamps'].iloc[-1]

if last_available_date < TARGET_END_DATE:
    extended_dates = pd.date_range(
        start=last_available_date + pd.offsets.BDay(1),
        end=TARGET_END_DATE,
        freq='B'
    )
else:
    extended_dates = pd.DatetimeIndex([])

full_rollout_dates = list(available_post_start['timestamps']) + list(extended_dates)
total_steps = len(full_rollout_dates)

current_full_df = df.iloc[:start_idx].copy().reset_index(drop=True)
predicted_rows = []

print(f"Executing step-wise rolling inference across {total_steps} trading sessions...")

with torch.no_grad():
    for step_i in range(0, total_steps, STEP_SIZE):
        target_timestamp = full_rollout_dates[step_i]
        
        # 375-day lookback slice
        sliding_context = current_full_df.iloc[-WINDOW_SIZE:].copy().reset_index(drop=True)
        
        step_pred_df = predictor.predict(
            df=sliding_context,
            x_timestamp=sliding_context['timestamps'],
            y_timestamp=pd.Series([target_timestamp]),
            pred_len=1,
            T=1.0,
            top_p=0.9,
            sample_count=1
        )
        
        step_pred_df['timestamps'] = target_timestamp
        predicted_rows.append(step_pred_df)
        
        # Roll window forward by 1 step
        current_full_df = pd.concat([current_full_df, step_pred_df], ignore_index=True)
        
        if (step_i + 1) % 50 == 0 or (step_i + 1) == total_steps:
            print(f"Progress: [{step_i + 1}/{total_steps}] sessions -> Evaluated Date: {target_timestamp.strftime('%Y-%m-%d')}")

forecast_df = pd.concat(predicted_rows, ignore_index=True)

# Metrics calculation on actual data
has_ground_truth = len(available_post_start) > 0
actual_eval_prices = []
pred_eval_prices = []

if has_ground_truth:
    eval_overlap = pd.merge(
        available_post_start[['timestamps', 'close']].rename(columns={'close': 'actual'}),
        forecast_df[['timestamps', 'close']].rename(columns={'close': 'pred'}),
        on='timestamps',
        how='inner'
    )
    actual_eval_prices = eval_overlap['actual'].values
    pred_eval_prices = eval_overlap['pred'].values
    
    mae = mean_absolute_error(actual_eval_prices, pred_eval_prices)
    rmse = np.sqrt(mean_squared_error(actual_eval_prices, pred_eval_prices))
    mape = np.mean(np.abs((actual_eval_prices - pred_eval_prices) / actual_eval_prices)) * 100
else:
    mae, rmse, mape = 0.0, 0.0, 0.0

print("\n" + "=" * 70)
print("EVALUATION METRICS (ORIGINAL INR POINTS)")
print("=" * 70)
print(f"MAE  : {mae:.2f} points")
print(f"RMSE : {rmse:.2f} points")
print(f"MAPE : {mape:.2f}%")

# =========================================================
# 6. GLOBAL 0-1 NORMALIZATION FOR TWO-PANEL PLOT
# =========================================================
plot_scaler = MinMaxScaler(feature_range=(0, 1))

all_prices = np.concatenate([df['close'].iloc[:start_idx].values, forecast_df['close'].values])
plot_scaler.fit(all_prices.reshape(-1, 1))

context_dates = df['timestamps'].iloc[:start_idx].values
context_normalized = plot_scaler.transform(df['close'].iloc[:start_idx].values.reshape(-1, 1)).flatten()

actual_normalized = plot_scaler.transform(available_post_start['close'].values.reshape(-1, 1)).flatten() if has_ground_truth else np.array([])
predicted_normalized = plot_scaler.transform(forecast_df['close'].values.reshape(-1, 1)).flatten()

absolute_error = np.abs(actual_eval_prices - pred_eval_prices) if has_ground_truth else np.array([])

# =========================================================
# 7. SAVE RESULTS TO CSV
# =========================================================
results_df = pd.DataFrame({
    "Date": forecast_df['timestamps'],
    "Predicted_Price": forecast_df['close'],
    "Predicted_Normalized": predicted_normalized
})

if has_ground_truth:
    results_df = pd.merge(
        results_df,
        available_post_start[['timestamps', 'close']].rename(columns={'timestamps': 'Date', 'close': 'Actual_Price'}),
        on='Date',
        how='left'
    )
    results_df['Actual_Normalized'] = plot_scaler.transform(results_df['Actual_Price'].fillna(0).values.reshape(-1, 1)).flatten()
    results_df['Absolute_Error'] = np.abs(results_df['Actual_Price'] - results_df['Predicted_Price'])

results_df.to_csv(CSV_FILE, index=False)
print(f"\nPredictions saved to: {CSV_FILE}")

# =========================================================
# 8. CREATE TWO-PANEL INTERACTIVE PLOTLY DASHBOARD
# =========================================================
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.10,
    row_heights=[0.68, 0.32],
    subplot_titles=(
        "NIFTY 50: Kronos Transformer LoRA 375-Day Sliding Window (Normalized 0–1)",
        "Absolute Prediction Error (Original NIFTY Points)"
    )
)

# Panel 1: Context Baseline
fig.add_trace(
    go.Scatter(
        x=context_dates,
        y=context_normalized,
        mode="lines",
        name="375-Day Context Baseline",
        line=dict(color="#94a3b8", width=1.8),
        hovertemplate="<b>Date:</b> %{x|%d-%b-%Y}<br><b>Context Norm:</b> %{y:.4f}<extra></extra>"
    ),
    row=1, col=1
)

# Panel 1: Actual Ground Truth
if has_ground_truth:
    fig.add_trace(
        go.Scatter(
            x=available_post_start['timestamps'],
            y=actual_normalized,
            mode="lines",
            name="Actual NIFTY50 (Normalized)",
            line=dict(color="#38bdf8", width=2),
            hovertemplate="<b>Date:</b> %{x|%d-%b-%Y}<br><b>Actual Norm:</b> %{y:.4f}<extra></extra>"
        ),
        row=1, col=1
    )

# Panel 1: LoRA Forecast
fig.add_trace(
    go.Scatter(
        x=forecast_df['timestamps'],
        y=predicted_normalized,
        mode="lines",
        name="Kronos LoRA Sliding Forecast",
        line=dict(color="#c084fc", width=2.2, dash="dash"),
        hovertemplate="<b>Date:</b> %{x|%d-%b-%Y}<br><b>Predicted Norm:</b> %{y:.4f}<extra></extra>"
    ),
    row=1, col=1
)

# Panel 2: Error Curve
if has_ground_truth:
    fig.add_trace(
        go.Scatter(
            x=available_post_start['timestamps'][:len(absolute_error)],
            y=absolute_error,
            mode="lines",
            name="Absolute Error",
            line=dict(color="#f87171", width=1.2),
            fill="tozeroy",
            hovertemplate="<b>Date:</b> %{x|%d-%b-%Y}<br><b>Error:</b> %{y:.2f} pts<extra></extra>"
        ),
        row=2, col=1
    )

fig.update_layout(
    title=dict(
        text=(
            "<b>NIFTY50: 375-Day Sliding Window LoRA Forecast (Jan 2022 – Jul 2026)</b><br>"
            f"<sup>Model: Kronos-base + LoRA (r=8, alpha=16) | Window: {WINDOW_SIZE} Days | "
            f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.2f}%</sup>"
        ),
        x=0.02,
        xanchor="left"
    ),
    template="plotly_dark",
    height=860,
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

fig.update_yaxes(title_text="Normalized Price (0 to 1)", range=[-0.05, 1.05], tickformat=".2f", row=1, col=1)
fig.update_yaxes(title_text="Absolute Error (Pts)", row=2, col=1)

fig.update_xaxes(
    title_text="Date",
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
        activecolor="#2563eb"
    ),
    row=2, col=1
)

fig.write_html(HTML_FILE, include_plotlyjs=True)
print(f"Interactive graph saved to: {HTML_FILE}")
print("\n" + "=" * 70)
print("EXECUTION COMPLETE")
print("=" * 70)