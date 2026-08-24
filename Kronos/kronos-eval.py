import os
import sys
import glob
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from peft import LoraConfig, get_peft_model, TaskType

script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
os.chdir(script_dir)
sys.path.insert(0, script_dir)

# ============================================================
# 1. SETUP HARDWARE & INJECT LORA ADAPTERS
# ============================================================
print("\n========================================")
print("1. LOADING FOUNDATION KRONOS & APPLYING LORA")
print("========================================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Active Device: {device}")

try:
    from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    if hasattr(tokenizer, 'to'):
        tokenizer = tokenizer.to(device)
        
    base_model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
    
    # Identify linear layers for LoRA injection
    target_modules = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Linear):
            layer_name = name.split(".")[-1]
            if layer_name in ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2", "proj", "linear"]:
                target_modules.append(layer_name)
    target_modules = list(set(target_modules))
    if not target_modules:
        target_modules = ["q_proj", "v_proj"]  # Fallback default

    # LoRA Configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none"
    )
    
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    model.print_trainable_parameters()
except Exception as e:
    print(f"[CRITICAL ERROR] Failed to configure LoRA model: {e}")
    sys.exit(1)

# ============================================================
# 2. LOAD DATASET & SLICE: 2022 TRAIN, 2023-JULY 2026 FORECAST
# ============================================================
print("\n========================================")
print("2. PREPARING DATASET: 2022 CONTEXT & HORIZON TO 2026")
print("========================================")

possible_paths = [
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
    print("[ERROR] Could not find NIFTY 50 parquet file.")
    sys.exit(1)

print(f"Loaded Dataset: {data_path}")
raw_df = pd.read_parquet(data_path)

if isinstance(raw_df.columns, pd.MultiIndex):
    raw_df.columns = [col[0] if isinstance(col, tuple) else str(col) for col in raw_df.columns]

raw_df.columns = [str(c).strip().lower() for c in raw_df.columns]

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

raw_df['timestamps'] = pd.to_datetime(raw_df['timestamps']).dt.tz_localize(None)

for target in ['open', 'high', 'low', 'close']:
    if target not in raw_df.columns:
        matching = [c for c in raw_df.columns if target in c]
        if matching:
            raw_df.rename(columns={matching[0]: target}, inplace=True)

if 'volume' not in raw_df.columns:
    vol_match = [c for c in raw_df.columns if 'vol' in c]
    raw_df['volume'] = raw_df[vol_match[0]] if vol_match else 1.0

# Resample to Daily
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

df['volume'] = df['volume'].replace(0, 1.0).fillna(1.0)
if 'amount' not in df.columns and 'turnover' not in df.columns:
    df['amount'] = df['close'] * df['volume']
elif 'turnover' in df.columns:
    df['amount'] = df['turnover'].fillna(df['close'] * df['volume'])

df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume', 'amount']).sort_values('timestamps').reset_index(drop=True)

# Slicing: 2022 for Training
train_mask = (df['timestamps'] >= '2022-01-01') & (df['timestamps'] <= '2022-12-31')
train_df = df[train_mask].copy().reset_index(drop=True)

if len(train_df) == 0:
    print("[WARNING] Strict 2022 filter empty, taking first 250 available sessions.")
    train_df = df.iloc[:250].copy().reset_index(drop=True)

last_train_date = pd.to_datetime(train_df['timestamps'].iloc[-1])
target_end_date = pd.to_datetime("2026-07-31")

full_pred_dates = pd.date_range(
    start=last_train_date + pd.offsets.BDay(1),
    end=target_end_date,
    freq='B'
)
pred_len = len(full_pred_dates)

test_ground_truth_df = df[df['timestamps'] > last_train_date].copy().reset_index(drop=True)
has_ground_truth = len(test_ground_truth_df) > 0

print(f"Training Window : {len(train_df)} sessions ({train_df['timestamps'].iloc[0].strftime('%Y-%m-%d')} to {last_train_date.strftime('%Y-%m-%d')})")
print(f"Forecast Horizon: {pred_len} sessions ({full_pred_dates[0].strftime('%Y-%m-%d')} to {full_pred_dates[-1].strftime('%Y-%m-%d')})")

# ============================================================
# 3. TOKEN EXTRACTION & FORWARD HELPERS
# ============================================================
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

# ============================================================
# 4. LORA FINE-TUNING WITH DIVERGENCE EARLY STOPPING
# ============================================================
print("\n========================================")
print("3. TRAINING LORA ADAPTERS (EARLY STOPPING: 15 DIVERGENCE LOOPS)")
print("========================================")

s1_tokens, s2_tokens = get_tokens(train_df)
seq_len = s1_tokens.shape[1]
split_idx = max(int(seq_len * 0.85), 2)

s1_train_x, s1_train_y = s1_tokens[:, :split_idx-1], s1_tokens[:, 1:split_idx]
s2_train_x, s2_train_y = s2_tokens[:, :split_idx-1], s2_tokens[:, 1:split_idx]

s1_val_x, s1_val_y = s1_tokens[:, split_idx-1:-1], s1_tokens[:, split_idx:]
s2_val_x, s2_val_y = s2_tokens[:, split_idx-1:-1], s2_tokens[:, split_idx:]

max_epochs = 200
divergence_patience = 15
divergence_counter = 0
min_delta = 1e-4

# Optimize ONLY the LoRA trainable adapter parameters
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-2)
checkpoint_dir = os.path.abspath("best_nifty_lora_adapter")
best_val_loss = float("inf")
best_epoch = 0

prev_train_loss = None
prev_val_loss = None

print(f"Starting LoRA fine-tuning: max_epochs={max_epochs}, divergence_patience={divergence_patience} continuous loops...")

for epoch in range(1, max_epochs + 1):
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

    if current_val_loss < (best_val_loss - min_delta):
        best_val_loss = current_val_loss
        best_epoch = epoch
        model.save_pretrained(checkpoint_dir)
        marker = " ◄ [BEST SAVED]"
    else:
        marker = ""

    # Divergence Early Stopping Rule
    if prev_train_loss is not None and prev_val_loss is not None:
        train_loss_decreased = (current_train_loss < prev_train_loss)
        val_loss_stagnant_or_increased = (current_val_loss >= prev_val_loss - min_delta)
        
        if train_loss_decreased and val_loss_stagnant_or_increased:
            divergence_counter += 1
            marker += f" | Divergence: {divergence_counter}/{divergence_patience}"
        else:
            divergence_counter = 0

    prev_train_loss = current_train_loss
    prev_val_loss = current_val_loss

    if epoch % 5 == 0 or epoch == 1 or "BEST" in marker or divergence_counter > 0:
        print(f"Epoch [{epoch:03d}/{max_epochs:03d}] ── Train Loss: {current_train_loss:.4f} | Val Loss: {current_val_loss:.4f}{marker}")

    if divergence_counter >= divergence_patience:
        print(f"\n[EARLY STOPPING TRIGGERED] Divergence detected: Train loss decreased while Val loss stagnated/increased for {divergence_patience} consecutive loops.")
        break

# Restore Best LoRA Adapter Checkpoint
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, checkpoint_dir)
print(f"\nRestored Best LoRA Checkpoint from Epoch {best_epoch:03d} (Best Val Loss: {best_val_loss:.4f})")

# ============================================================
# 5. MULTI-CHUNK AUTOREGRESSIVE ROLLOUT TO JULY 2026
# ============================================================
print("\n========================================")
print("4. GENERATING FORECAST (JAN 2023 TO JULY 2026)")
print("========================================")

model.eval()
predictor = KronosPredictor(model=model, tokenizer=tokenizer, device=device, max_context=512)

max_chunk_size = 350
current_context_df = train_df.copy()
all_predictions = []
remaining_len = pred_len
current_start_idx = 0

with torch.no_grad():
    while remaining_len > 0:
        chunk_len = min(remaining_len, max_chunk_size)
        chunk_dates = full_pred_dates[current_start_idx : current_start_idx + chunk_len]
        context_slice = current_context_df.iloc[-400:].reset_index(drop=True)
        
        chunk_pred_df = predictor.predict(
            df=context_slice,
            x_timestamp=context_slice['timestamps'],
            y_timestamp=pd.Series(chunk_dates),
            pred_len=chunk_len,
            T=1.0,
            top_p=0.9,
            sample_count=1
        )
        
        chunk_pred_df['timestamps'] = chunk_dates
        all_predictions.append(chunk_pred_df)
        current_context_df = pd.concat([current_context_df, chunk_pred_df], ignore_index=True)
        
        current_start_idx += chunk_len
        remaining_len -= chunk_len
        print(f"Generated {current_start_idx}/{pred_len} trading sessions...")

full_pred_df = pd.concat(all_predictions, ignore_index=True)
y_pred = full_pred_df['close'].values[:pred_len]

rmse_text, mae_text, mape_text = "N/A", "N/A", "N/A"
if has_ground_truth:
    overlap_df = pd.merge(
        test_ground_truth_df[['timestamps', 'close']].rename(columns={'close': 'actual'}),
        pd.DataFrame({'timestamps': full_pred_dates, 'pred': y_pred}),
        on='timestamps',
        how='inner'
    )
    if len(overlap_df) > 0:
        mse = mean_squared_error(overlap_df['actual'], overlap_df['pred'])
        rmse_text = f"{np.sqrt(mse):.2f}"
        mae_text = f"{mean_absolute_error(overlap_df['actual'], overlap_df['pred']):.2f}"
        mape_text = f"{np.mean(np.abs((overlap_df['actual'] - overlap_df['pred']) / overlap_df['actual'])) * 100:.2f}%"

# ============================================================
# 6. RENDER FULL ZOOMABLE CHART
# ============================================================
fig = go.Figure()

# 1. 2022 Training History
fig.add_trace(go.Scatter(
    x=train_df['timestamps'], y=train_df['close'],
    mode="lines", name="2022 Training Context", line=dict(color="#64748b", width=2)
))

# 2. Actual Market Ground Truth (2023 onward)
if has_ground_truth:
    fig.add_trace(go.Scatter(
        x=test_ground_truth_df['timestamps'], y=test_ground_truth_df['close'],
        mode="lines", name="Actual Market Ground Truth (2023+)", line=dict(color="#38bdf8", width=2)
    ))

# 3. Model Prediction (Jan 2023 to July 2026)
fig.add_trace(go.Scatter(
    x=full_pred_dates, y=y_pred,
    mode="lines", name="LoRA Forecast (Jan 2023 – Jul 2026)", line=dict(color="#c084fc", width=2.5)
))

fig.update_layout(
    title=dict(
        text="NIFTY 50: LoRA Fine-Tuned on 2022 → Forecasted to July 2026",
        x=0.5, xanchor="center", font=dict(size=19, color="#f8fafc")
    ),
    template="plotly_dark",
    height=780,
    autosize=True,
    dragmode="zoom",
    hovermode="x unified",
    margin=dict(l=60, r=60, t=110, b=80),
    yaxis=dict(title="<b>NIFTY 50 Index Level</b>", showgrid=True, gridcolor="#334155"),
    xaxis=dict(
        title="Date", showgrid=True, gridcolor="#334155", type="date",
        rangeslider=dict(visible=True, thickness=0.08),
        rangeselector=dict(
            buttons=[
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="All")
            ],
            bgcolor="#1e293b", activecolor="#2563eb", font=dict(color="#f8fafc", size=11)
        )
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    annotations=[dict(
        x=0.01, y=0.96, xref="paper", yref="paper",
        text=f"<b>LoRA Training Summary:</b><br>• Method: LoRA (r=8, alpha=16)<br>• Training Range: 2022-01 to 2022-12<br>• Forecast: 2023-01 to 2026-07<br>• Best Epoch: {best_epoch}<br>• Test RMSE: {rmse_text}<br>• Test MAE: {mae_text}<br>• Test MAPE: {mape_text}",
        showarrow=False, bgcolor="#0f172a", bordercolor="#334155", borderwidth=1, align="left", font=dict(size=12, color="#e2e8f0")
    )]
)

out_file = os.path.join(script_dir, "nifty_lora_2022_forecast.html")
fig.write_html(out_file, auto_open=False, include_plotlyjs=True)
print(f"\n[SUCCESS] Chart written to: {out_file}")