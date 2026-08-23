import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from peft import LoraConfig, get_peft_model, PeftModel

import config

repo_path = getattr(config, "KRONOS_REPO_PATH", "/home/soq/Kronos")
for p in [config.PROJECT_ROOT, repo_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_utils import get_base_model, get_tokenizer
from model.kronos import KronosPredictor
from walk_forward_dataset import normalize_dates

# Adapter checkpoint directory for this specific experiment
NIFTY_2024_ADAPTER_DIR = os.path.join(config.PROJECT_ROOT, "lora_adapters/nifty_2024_lora")
os.makedirs(NIFTY_2024_ADAPTER_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Dataset & Tokenization Utilities
# ---------------------------------------------------------
class KronosPatchDataset(Dataset):
    def __init__(self, s1_seq: torch.Tensor, s2_seq: torch.Tensor, seq_len: int = 32):
        self.seq_len = seq_len
        self.s1_seq = torch.clamp(s1_seq.flatten().long(), 0, 1023)
        self.s2_seq = torch.clamp(s2_seq.flatten().long(), 0, 1023)

    def __len__(self) -> int:
        return max(1, len(self.s1_seq) - self.seq_len)

    def __getitem__(self, idx: int):
        idx_bounded = min(idx, max(0, len(self.s1_seq) - self.seq_len - 1))
        return {
            "s1_ids": self.s1_seq[idx_bounded : idx_bounded + self.seq_len],
            "s2_ids": self.s2_seq[idx_bounded : idx_bounded + self.seq_len],
            "s1_targets": self.s1_seq[idx_bounded + 1 : idx_bounded + self.seq_len + 1],
            "s2_targets": self.s2_seq[idx_bounded + 1 : idx_bounded + self.seq_len + 1]
        }

def load_and_filter_nifty_data():
    """Loads Nifty parquet and splits strictly into 2024 train and 2025 eval slices."""
    df = pd.read_parquet(config.NIFTY_DATA_PATH)
    df.columns = df.columns.str.lower()
    df['date'] = normalize_dates(df['date'])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'amount' not in df.columns or df['amount'].isnull().all():
        df['amount'] = df['close'] * df['volume']
    else:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    df = df.ffill().bfill().sort_values('date').reset_index(drop=True)

    # Filter train (2024) and eval (2025) datasets
    df_train_2024 = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2024-12-31')].reset_index(drop=True)
    df_eval_2025 = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-12-31')].reset_index(drop=True)

    print(f"✓ Loaded {len(df_train_2024)} candles for 2024 Training ({df_train_2024['date'].min().date()} to {df_train_2024['date'].max().date()})")
    print(f"✓ Loaded {len(df_eval_2025)} candles for 2025 Evaluation ({df_eval_2025['date'].min().date()} to {df_eval_2025['date'].max().date()})")

    return df, df_train_2024, df_eval_2025

def prepare_token_loaders(df_train: pd.DataFrame, tokenizer, val_ratio: float = 0.15):
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    patch_size = 16
    
    windows = []
    for i in range(len(df_train) - patch_size + 1):
        windows.append(df_train[feature_cols].iloc[i : i + patch_size].values)
        
    raw_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(config.DEVICE)
    
    all_s1, all_s2 = [], []
    batch_size = 32
    
    with torch.no_grad():
        for b in range(0, len(raw_tensor), batch_size):
            slice_tensor = raw_tensor[b : b + batch_size]
            enc = tokenizer.encode(slice_tensor)
            
            if isinstance(enc, (tuple, list)):
                s1 = enc[0].detach().cpu().flatten()
                s2 = enc[1].detach().cpu().flatten()
            elif isinstance(enc, dict):
                s1 = enc['s1_ids'].detach().cpu().flatten()
                s2 = enc['s2_ids'].detach().cpu().flatten()
            elif isinstance(enc, torch.Tensor) and enc.shape[-1] >= 2:
                s1 = enc[..., 0].detach().cpu().flatten()
                s2 = enc[..., 1].detach().cpu().flatten()
            else:
                s1 = enc.detach().cpu().flatten()
                s2 = enc.detach().cpu().flatten()
                
            all_s1.append(s1)
            all_s2.append(s2)

    s1_all = torch.clamp(torch.cat(all_s1, dim=0), 0, 1023)
    s2_all = torch.clamp(torch.cat(all_s2, dim=0), 0, 1023)

    split_idx = int(len(s1_all) * (1 - val_ratio))
    seq_len = getattr(config, "SEQ_LEN", 32)

    train_ds = KronosPatchDataset(s1_all[:split_idx], s2_all[:split_idx], seq_len=seq_len)
    val_ds = KronosPatchDataset(s1_all[split_idx:], s2_all[split_idx:], seq_len=seq_len)

    if len(val_ds) == 0:
        val_ds = train_ds

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    return train_loader, val_loader

# ---------------------------------------------------------
# 2. LoRA Training Setup
# ---------------------------------------------------------
def build_peft_lora(base_model):
    available_modules = set()
    for name, _ in base_model.named_modules():
        for target in config.LORA_TARGET_MODULES:
            if name.endswith(target) or target in name:
                available_modules.add(name.split(".")[-1])

    target_mods = list(available_modules) if available_modules else ["q_proj", "v_proj", "k_proj", "out_proj"]

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=target_mods,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=None
    )
    return get_peft_model(base_model, lora_config)

def train_lora_2024(base_model, train_loader, val_loader):
    model = build_peft_lora(base_model)
    model.train()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = config.PATIENCE

    print("\n--> Starting 2024 NIFTY 50 LoRA Fine-Tuning...")

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            s1_in = batch['s1_ids'].to(config.DEVICE)
            s2_in = batch['s2_ids'].to(config.DEVICE)
            s1_tgt = batch['s1_targets'].to(config.DEVICE)

            optimizer.zero_grad()
            outputs = model(s1_ids=s1_in, s2_ids=s2_in)

            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('s1_logits', list(outputs.values())[0]))
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif hasattr(outputs, 's1_logits'):
                logits = outputs.s1_logits
            elif isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs

            loss = criterion(logits.reshape(-1, logits.size(-1)), s1_tgt.reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / max(1, len(train_loader))

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                s1_in = batch['s1_ids'].to(config.DEVICE)
                s2_in = batch['s2_ids'].to(config.DEVICE)
                s1_tgt = batch['s1_targets'].to(config.DEVICE)

                outputs = model(s1_ids=s1_in, s2_ids=s2_in)
                if isinstance(outputs, dict):
                    logits = outputs.get('logits', outputs.get('s1_logits', list(outputs.values())[0]))
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                elif hasattr(outputs, 's1_logits'):
                    logits = outputs.s1_logits
                elif isinstance(outputs, (tuple, list)):
                    logits = outputs[0]
                else:
                    logits = outputs

                loss = criterion(logits.reshape(-1, logits.size(-1)), s1_tgt.reshape(-1))
                val_loss += loss.item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"Epoch [{epoch:02d}/{config.EPOCHS:02d}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Checkpoint Saving
        if avg_val_loss < best_val_loss - config.MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            model.save_pretrained(NIFTY_2024_ADAPTER_DIR)
            print(f"  --> Best 2024 LoRA weights saved to {NIFTY_2024_ADAPTER_DIR}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"--> Early stopping triggered at Epoch {epoch}.")
                break

    return NIFTY_2024_ADAPTER_DIR

# ---------------------------------------------------------
# 3. 2025 Ground-Truth Rolling Prediction Loop
# ---------------------------------------------------------
def predict_2025_year(model, tokenizer, df_full, df_eval_2025, horizon=30):
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    max_context = getattr(config, "LOOKBACK_WINDOW", 400)
    
    predictor = KronosPredictor(model, tokenizer, max_context=max_context)
    
    eval_start_idx = df_full[df_full['date'] == df_eval_2025['date'].iloc[0]].index[0]
    total_eval_steps = len(df_eval_2025)
    pred_close = []

    print(f"\n--> Forecasting {total_eval_steps} trading days across 2025 (horizon={horizon} days)...")

    for step in range(0, total_eval_steps, horizon):
        curr_idx = eval_start_idx + step
        chunk_len = min(horizon, total_eval_steps - step)

        history_window = df_full.iloc[curr_idx - max_context : curr_idx].reset_index(drop=True)
        eval_window = df_full.iloc[curr_idx : curr_idx + chunk_len].reset_index(drop=True)

        x_features = history_window[feature_cols]
        x_timestamp = pd.to_datetime(history_window['date'])
        y_timestamp = pd.to_datetime(eval_window['date'])

        with torch.no_grad():
            pred_chunk = predictor.predict(
                df=x_features,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=chunk_len
            )

        if isinstance(pred_chunk, pd.DataFrame):
            pred_close.extend(pred_chunk['close'].values.tolist())
        else:
            pred_close.extend(pred_chunk[:, 3].tolist())

    return np.array(pred_close[:total_eval_steps])

# ---------------------------------------------------------
# 4. Main Pipeline Execution & Visualization
# ---------------------------------------------------------
def main():
    # 1. Load Data
    df_full, df_train_2024, df_eval_2025 = load_and_filter_nifty_data()
    tokenizer = get_tokenizer()
    if hasattr(tokenizer, 'to'):
        tokenizer = tokenizer.to(config.DEVICE)
    tokenizer.eval()

    # 2. Tokenize & Train LoRA on 2024
    train_loader, val_loader = prepare_token_loaders(df_train_2024, tokenizer)
    base_model = get_base_model().to(config.DEVICE)
    adapter_path = train_lora_2024(base_model, train_loader, val_loader)

    # 3. Load Trained Model for 2025 Predictions
    print(f"\n--> Loading fine-tuned LoRA model for 2025 inference from {adapter_path}...")
    base_model_eval = get_base_model().to(config.DEVICE)
    ft_model = PeftModel.from_pretrained(base_model_eval, adapter_path).to(config.DEVICE)
    ft_model.eval()

    # 4. Generate 2025 Predictions
    predicted_2025_close = predict_2025_year(ft_model, tokenizer, df_full, df_eval_2025, horizon=30)
    actual_2025_close = df_eval_2025['close'].values

    # 5. Evaluate Metrics
    rmse = np.sqrt(mean_squared_error(actual_2025_close, predicted_2025_close))
    mae = mean_absolute_error(actual_2025_close, predicted_2025_close)
    mape = np.mean(np.abs((actual_2025_close - predicted_2025_close) / actual_2025_close)) * 100

    print(f"\n================ NIFTY 50 (YEAR 2025) EVALUATION ================")
    print(f"Total Predicted Trading Days : {len(actual_2025_close)}")
    print(f"2025 Root Mean Squared Error : {rmse:.2f} pts")
    print(f"2025 Mean Absolute Error     : {mae:.2f} pts")
    print(f"2025 Mean Abs Percentage Err : {mape:.2f}%")
    print(f"=================================================================")

    # 6. Interactive Plotly Graph
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_eval_2025['date'],
        y=actual_2025_close,
        mode='lines',
        name='NIFTY 50 Actual 2025 (Ground Truth)',
        line=dict(color='#0f172a', width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=df_eval_2025['date'],
        y=predicted_2025_close,
        mode='lines',
        name='2024 LoRA Fine-Tuned Kronos Prediction',
        line=dict(color='#dc2626', width=2.2, dash='solid')
    ))

    fig.update_layout(
        title="NIFTY 50 (Year 2025): Actual vs 2024 LoRA Fine-Tuned Kronos Forecast",
        xaxis_title="Date (Year 2025)",
        yaxis_title="Index Level (Points)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    output_html = os.path.join(config.PROJECT_ROOT, "outputs/nifty50_2025_prediction.html")
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    fig.write_html(output_html)
    print(f"✓ Saved 2025 prediction comparison chart to: {output_html}")

if __name__ == "__main__":
    main()