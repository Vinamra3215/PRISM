import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import config

class RollingPatchDataset(Dataset):
    """
    Slices a 1D tokenized stream into overlapping sub-sequences of length seq_len
    for next-token prediction training.
    """
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

def normalize_dates(series) -> pd.Series:
    """Normalizes pandas datetime objects to timezone-naive datetimes."""
    dt_series = pd.to_datetime(series)
    if hasattr(dt_series, "dt") and dt_series.dt.tz is not None:
        dt_series = dt_series.dt.tz_localize(None)
    return pd.Series(pd.to_datetime(dt_series.values)).reset_index(drop=True)

def load_nifty_data(file_path: str = config.NIFTY_DATA_PATH) -> pd.DataFrame:
    """Loads and standardizes Nifty50 2022-2026 data."""
    df = pd.read_parquet(file_path)
    df.columns = df.columns.str.lower()
    df['date'] = normalize_dates(df['date'])

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'amount' not in df.columns or df['amount'].isnull().all():
        df['amount'] = df['close'] * df['volume']
    else:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')

    # Filter up to August 2, 2026
    cutoff_date = pd.to_datetime("2026-08-02")
    df = df[df['date'] <= cutoff_date]

    return df.ffill().bfill().sort_values('date').reset_index(drop=True)

def tokenize_window(window_df: pd.DataFrame, tokenizer) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts 16-candle patches from a 442-day slice and encodes them into Kronos s1 and s2 token streams.
    """
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    patch_size = 16
    
    windows = []
    for i in range(len(window_df) - patch_size + 1):
        windows.append(window_df[feature_cols].iloc[i : i + patch_size].values)
        
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
    return s1_all, s2_all