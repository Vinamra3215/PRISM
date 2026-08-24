import os
import sys
import copy
import json
import inspect
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. SETUP ENVIRONMENT & DYNAMIC PATHS
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

# 1.1 Locate Parquet Data File
possible_parquet_paths = [
    os.path.join(BASE_USER_DIR, "NIFTY50_5Y_OHLCV.parquet"),
    os.path.join(BASE_USER_DIR, "Kronos", "NIFTY50_5Y_OHLCV.parquet"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "NIFTY50_5Y_OHLCV.parquet")),
]

data_path = next((p for p in possible_parquet_paths if os.path.exists(p)), None)
if not data_path:
    raise FileNotFoundError("NIFTY50_5Y_OHLCV.parquet file system mein nahi mili!")

print(f"[DATA PATH] Loaded From: {data_path}")

# 1.2 Locate Pretrained Model Weights
possible_model_paths = [
    os.path.join(BASE_USER_DIR, "het-uchiha", "weights", "Kronos-base"),
    os.path.join(BASE_USER_DIR, "Kronos", "best_weights_kronos"),
    os.path.join(BASE_USER_DIR, "Kronos", "kronos_checkpoints"),
]

model_path = next((m for m in possible_model_paths if os.path.exists(m)), None)
if not model_path:
    raise FileNotFoundError("Pretrained Kronos weights directory nahi mili!")

print(f"[MODEL PATH] Pretrained Weights From: {model_path}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"[HARDWARE] Active Device: {device}")


# ==========================================
# 2. ACCURATE KRONOS MODEL CLASS FILTER
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
            print(f"[MODEL IMPORT] Detected Main Model Class: '{name}'")
            break

if KronosModelClass is None:
    for name, cls in candidate_classes:
        if not any(kw in name.lower() for kw in ignore_keywords):
            KronosModelClass = cls
            print(f"[MODEL IMPORT] Fallback Model Class: '{name}'")
            break

if KronosModelClass is None:
    raise ImportError("model.kronos mein koi main model class nahi mili!")


# ==========================================
# 3. LOAD FULL NIFTY DATASET & 80/20 SPLIT
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
print(f"[DATA LOADED] Total Records: {len(df_full)} ({df_full.index.min().date()} to {df_full.index.max().date()})")

# Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_close = scaler.fit_transform(df_full[['Close']].values)

# Sequential Split (80% Train, 20% Validation)
split_idx = int(len(scaled_close) * 0.80)
train_data = scaled_close[:split_idx]
val_data = scaled_close[split_idx:]


# ==========================================
# 4. 100-DAY ROLLING WINDOW DATASETS
# ==========================================
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len=100):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return x, y

seq_length = 100
train_dataset = TimeSeriesDataset(train_data, seq_len=seq_length)
val_dataset = TimeSeriesDataset(val_data, seq_len=seq_length)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"[WINDOW CONFIG] Window: {seq_length} Days | Train Samples (80%): {len(train_dataset)} | Val Samples (20%): {len(val_dataset)}")


# ==========================================
# 5. ROBUST MODEL INITIALIZATION & WEIGHTS
# ==========================================
# Default fallback parameters for Kronos model
default_params = {
    's1_bits': 8,
    's2_bits': 8,
    'n_layers': 6,
    'd_model': 512,
    'n_heads': 8,
    'ff_dim': 2048,
    'ffn_dropout_p': 0.1,
    'attn_dropout_p': 0.1,
    'resid_dropout_p': 0.1,
    'token_dropout_p': 0.1,
    'learn_te': True
}

# Search for config file across directory structure
search_dirs = [model_path, os.path.dirname(model_path), os.path.join(BASE_USER_DIR, "Kronos")]
config_file = next((os.path.join(d, "config.json") for d in search_dirs if os.path.exists(os.path.join(d, "config.json"))), None)

if config_file:
    print(f"[CONFIG] Loaded configuration from: {config_file}")
    with open(config_file, "r") as f:
        loaded_cfg = json.load(f)
        default_params.update(loaded_cfg)

# Filter parameters matching constructor signature
sig = inspect.signature(KronosModelClass.__init__)
init_kwargs = {k: v for k, v in default_params.items() if k in sig.parameters}

try:
    base_model = KronosModelClass.from_pretrained(model_path)
    print("[MODEL] Instantiated via from_pretrained")
except Exception:
    base_model = KronosModelClass(**init_kwargs)
    print(f"[MODEL] Instantiated with auto-resolved args: {list(init_kwargs.keys())}")

# Load Pretrained Weights
weight_files = ["pytorch_model.bin", "model.safetensors", "model.bin"]
weight_path = next((os.path.join(model_path, wf) for wf in weight_files if os.path.exists(os.path.join(model_path, wf))), None)

if weight_path:
    print(f"[WEIGHTS] Loading pretrained weights from: {weight_path}")
    if weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(weight_path)
    else:
        state_dict = torch.load(weight_path, map_location="cpu")
    
    base_model.load_state_dict(state_dict, strict=False)

# Freeze Base Parameters
for param in base_model.parameters():
    param.requires_grad = False

# Apply LoRA Adapter
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)

model = get_peft_model(base_model, lora_config)
model.to(device)
model.print_trainable_parameters()


# ==========================================
# 6. TRAINING LOOP WITH EARLY STOPPING
# ==========================================
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

max_epochs = 100
patience = 10
patience_counter = 0
best_val_loss = float('inf')
best_model_state = None

print(f"\n=== Starting Fine-Tuning (80/20 Split | Max Epochs: {max_epochs} | Patience: {patience}) ===")

for epoch in range(max_epochs):
    # --- TRAIN PHASE ---
    model.train()
    train_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # --- VALIDATION PHASE ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # --- EARLY STOPPING CHECK ---
    if avg_val_loss < best_val_loss - 1e-6:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = copy.deepcopy(model.state_dict())
        status_msg = f"[Saved Best Model Checkpoint]"
    else:
        patience_counter += 1
        status_msg = f"[Patience: {patience_counter}/{patience}]"

    print(f"Epoch [{epoch+1:03d}/{max_epochs:03d}] | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} {status_msg}")

    if patience_counter >= patience:
        print(f"\n[EARLY STOPPING TRIGGERED] Validation loss didn't improve for {patience} consecutive epochs.")
        break


# ==========================================
# 7. SAVE BEST LORA WEIGHTS
# ==========================================
if best_model_state is not None:
    model.load_state_dict(best_model_state)

save_dir = os.path.join(BASE_USER_DIR, "Kronos", "best_weights", "kronos_lora_nifty_fulldata_100d_80_20")
os.makedirs(save_dir, exist_ok=True)

model.save_pretrained(save_dir)
print(f"\n[SUCCESS] Best LoRA Adapter Saved Successfully at:\n{save_dir}")