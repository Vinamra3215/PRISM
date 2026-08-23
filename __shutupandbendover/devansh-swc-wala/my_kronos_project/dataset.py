import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import config

class KronosTokenDataset(Dataset):
    def __init__(self, s1_seq: torch.Tensor, s2_seq: torch.Tensor, seq_len: int = 32):
        self.seq_len = seq_len
        self.s1_seq = torch.clamp(s1_seq.flatten().long(), 0, 1023)
        self.s2_seq = torch.clamp(s2_seq.flatten().long(), 0, 1023)

    def __len__(self) -> int:
        return max(0, len(self.s1_seq) - self.seq_len)

    def __getitem__(self, idx: int):
        return {
            "s1_ids": self.s1_seq[idx : idx + self.seq_len],
            "s2_ids": self.s2_seq[idx : idx + self.seq_len],
            "s1_targets": self.s1_seq[idx + 1 : idx + self.seq_len + 1],
            "s2_targets": self.s2_seq[idx + 1 : idx + self.seq_len + 1]
        }

def normalize_dates(series) -> pd.Series:
    dt_series = pd.to_datetime(series)
    if hasattr(dt_series, "dt") and dt_series.dt.tz is not None:
        dt_series = dt_series.dt.tz_localize(None)
    return pd.Series(pd.to_datetime(dt_series.values)).reset_index(drop=True)

def compute_log_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converts absolute OHLCV into stationary log-returns / relative changes."""
    df = df.copy()
    df.columns = df.columns.str.lower()
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Ensure amount exists
    if 'amount' not in df.columns or df['amount'].isnull().all():
        df['amount'] = df['close'] * df['volume']
    else:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    df = df.ffill().bfill()
    df_feat = pd.DataFrame(index=df.index)
    
    # 1. Close-to-Close Log Return
    df_feat['close_ret'] = np.log(df['close'] / df['close'].shift(1))
    
    # 2. Intra-candle Relative Moves (normalized by previous close)
    df_feat['open_ret'] = np.log(df['open'] / df['close'].shift(1))
    df_feat['high_ret'] = np.log(df['high'] / df['close'].shift(1))
    df_feat['low_ret'] = np.log(df['low'] / df['close'].shift(1))
    
    # 3. Volume and Amount Log Growth
    df_feat['vol_ret'] = np.log((df['volume'] + 1) / (df['volume'].shift(1) + 1))
    df_feat['amt_ret'] = np.log((df['amount'] + 1) / (df['amount'].shift(1) + 1))
    
    # Drop first NaN row resulting from shift(1)
    df_feat = df_feat.dropna().reset_index(drop=True)
    return df_feat

def load_training_tokens(tokenizer, val_ratio: float = 0.15):
    parquet_path = config.DATA_TRAIN_PATH
    df = pd.read_parquet(parquet_path)
    df.columns = df.columns.str.lower()
    df['date'] = normalize_dates(df['date'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    if 'amount' not in df.columns:
        df['amount'] = df['close'] * df['volume']
    else:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
    df = df.ffill().bfill().sort_values('date').reset_index(drop=True)
    
    # Extract stationary relative features
    df_returns = compute_log_return_features(df)
    feature_cols = ['open_ret', 'high_ret', 'low_ret', 'close_ret', 'vol_ret', 'amt_ret']
    
    total_candles = len(df_returns)
    window_size = 16
    
    windows = []
    for i in range(total_candles - window_size + 1):
        windows.append(df_returns[feature_cols].iloc[i : i + window_size].values)
    
    raw_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(config.DEVICE)
    print(f"--> Extracted {len(raw_tensor)} log-return sequence patches of length {window_size}...")

    if hasattr(tokenizer, 'to'):
        tokenizer = tokenizer.to(config.DEVICE)
    tokenizer.eval()

    all_s1 = []
    all_s2 = []
    
    batch_enc_size = 32
    with torch.no_grad():
        for b in range(0, len(raw_tensor), batch_enc_size):
            batch_slice = raw_tensor[b : b + batch_enc_size]
            enc = tokenizer.encode(batch_slice)
            
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

    s1_all = torch.cat(all_s1, dim=0)
    s2_all = torch.cat(all_s2, dim=0)

    # Strictly clamp to valid vocabulary boundary
    s1_all = torch.clamp(s1_all, 0, 1023)
    s2_all = torch.clamp(s2_all, 0, 1023)

    print(f"✓ Token bounds verified: s1 [{s1_all.min().item()}, {s1_all.max().item()}], s2 [{s2_all.min().item()}, {s2_all.max().item()}] (vocab: 1024)")

    split_idx = int(len(s1_all) * (1 - val_ratio))
    seq_len = 32

    train_dataset = KronosTokenDataset(s1_all[:split_idx], s2_all[:split_idx], seq_len=seq_len)
    val_dataset = KronosTokenDataset(s1_all[split_idx:], s2_all[split_idx:], seq_len=seq_len)

    if len(val_dataset) == 0:
        val_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    print(f"✓ Ready: {len(train_dataset)} train samples, {len(val_dataset)} val samples.")
    return df, train_loader, val_loader