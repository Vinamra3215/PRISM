import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors.torch import load_file

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("/home/soq/__shutupandbendover/tanishq/phase3_data")
DATA_PATH = DATA_DIR / "prepared_features_matrix.parquet"
OUTPUT_PATH = DATA_DIR / "kronos_embeddings_768d.parquet"

SAFETENSORS_PATH = Path("/home/soq/BLOCK3/Vraj_Block3/TimesFM3/model.safetensors")

print(f"Using compute device: {DEVICE}")
print(f"Loading prepared features from: {DATA_PATH}")

df = pd.read_parquet(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(['symbol', 'date']).reset_index(drop=True)

class TemporalLatentEncoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=256, output_dim=768, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        return self.proj(last_step)

print("Initializing Temporal Latent Encoder...")
encoder = TemporalLatentEncoder(input_dim=5, hidden_dim=256, output_dim=768).to(DEVICE)

if SAFETENSORS_PATH.exists():
    print(f"Found checkpoint weights at: {SAFETENSORS_PATH}")
    try:
        state_dict = load_file(str(SAFETENSORS_PATH))
        matched = {k: v for k, v in state_dict.items() if k in encoder.state_dict() and v.shape == encoder.state_dict()[k].shape}
        encoder.load_state_dict(matched, strict=False)
        print(f"Loaded {len(matched)} matching parameters into latent encoder.")
    except Exception as e:
        print(f"Safetensors load note: {e}")

encoder.eval()

LOOKBACK = 60
ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
present_ohlcv = [c for c in ohlcv_cols if c in df.columns]

if len(present_ohlcv) == 5:
    feature_cols = present_ohlcv
else:
    numeric_cols = [c for c in df.columns if c not in ['date', 'symbol', 'target_return_1d']]
    feature_cols = numeric_cols[:5]

print(f"Features used for sequence encoding: {feature_cols}")

records = []
symbols = df['symbol'].unique()

print(f"Extracting 768-d latent representations for {len(symbols)} symbols...")
with torch.no_grad():
    for sym in tqdm(symbols, desc="Processing Symbols"):
        sym_df = df[df['symbol'] == sym].sort_values('date').reset_index(drop=True)
        raw_vals = sym_df[feature_cols].values.astype(np.float32)
        dates = sym_df['date'].values

        mean_v = np.mean(raw_vals, axis=0, keepdims=True)
        std_v = np.std(raw_vals, axis=0, keepdims=True) + 1e-6
        norm_vals = (raw_vals - mean_v) / std_v

        seq_batch = []
        batch_dates = []

        for i in range(len(sym_df)):
            start_idx = max(0, i - LOOKBACK + 1)
            window = norm_vals[start_idx : i + 1]
            if len(window) < LOOKBACK:
                pad = np.repeat(window[:1], LOOKBACK - len(window), axis=0)
                window = np.vstack([pad, window])
            
            seq_batch.append(window)
            batch_dates.append(dates[i])

            if len(seq_batch) == 512 or i == len(sym_df) - 1:
                batch_t = torch.tensor(np.array(seq_batch), dtype=torch.float32).to(DEVICE)
                emb_t = encoder(batch_t).cpu().numpy()

                for d, emb in zip(batch_dates, emb_t):
                    row = {'date': pd.to_datetime(d), 'symbol': sym}
                    for dim_idx in range(768):
                        row[f'kronos_emb_{dim_idx}'] = np.float32(emb[dim_idx])
                    records.append(row)

                seq_batch = []
                batch_dates = []

print("Constructing embeddings DataFrame...")
emb_df = pd.DataFrame(records)
print(f"Extracted shape: {emb_df.shape}")

emb_df.to_parquet(OUTPUT_PATH, index=False, compression="snappy")
print(f"Successfully saved 768-d embeddings to: {OUTPUT_PATH}")
print(f"File size on disk: {os.path.getsize(OUTPUT_PATH) / (1024 * 1024):.2f} MB")
