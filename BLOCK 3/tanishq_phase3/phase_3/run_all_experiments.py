import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from tqdm import tqdm

from models_3 import ConcatMLP, CrossAttentionGating, FiLMConditioning

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("/home/soq/__shutupandbendover/tanishq/phase3_data")
RESULTS_DIR = Path("/home/soq/__shutupandbendover/tanishq/phase3_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--arch", type=str, choices=["A", "B", "C", "all"], default="B")
parser.add_argument("--smoke_test", action="store_true", help="Run 5-day smoke test")
args = parser.parse_args()

print(f"Device: {DEVICE}")
print(f"Loading matrices from {DATA_DIR}...")

feat_df = pd.read_parquet(DATA_DIR / "prepared_features_matrix.parquet")
emb_df = pd.read_parquet(DATA_DIR / "kronos_embeddings_768d.parquet")

feat_df['date'] = pd.to_datetime(feat_df['date'])
emb_df['date'] = pd.to_datetime(emb_df['date'])

df = pd.merge(feat_df, emb_df, on=['date', 'symbol'], how='inner').sort_values(['date', 'symbol']).reset_index(drop=True)
print(f"Merged dataset shape: {df.shape}")

market_cols = ['nifty_daily_return', 'nifty_5d_return', 'nifty_vol_20d', 'pharma_return']
stock_cols = [
    'ma_ratio_50_200', 'rolling_volatility_10d', 'intraday_range',
    'keltner_channel_position', 'trend_strength', 'cs_return_z',
    'volume_price_trend', 'volatility_ratio_5d_20d', 'sector_relative_return',
    'rolling_skewness_20d'
]
kronos_cols = [f'kronos_emb_{i}' for i in range(768)]

df[stock_cols] = df[stock_cols].fillna(0.0)
df[market_cols] = df[market_cols].fillna(0.0)

# Differentiable Rank Correlation Loss
class CrossSectionalCorrelationLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, preds, targets):
        p_norm = (preds - preds.mean()) / (preds.std() + self.eps)
        t_norm = (targets - targets.mean()) / (targets.std() + self.eps)
        return -(p_norm * t_norm).mean()

class FastDataset(Dataset):
    def __init__(self, k, s, m, y):
        self.k = torch.tensor(k, dtype=torch.float32)
        self.s = torch.tensor(s, dtype=torch.float32)
        self.m = torch.tensor(m, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.k[idx], self.s[idx], self.m[idx], self.y[idx]

def compute_rank_ic(preds, targets):
    if len(preds) < 10 or np.all(preds == preds[0]):
        return 0.0
    rho, _ = spearmanr(preds, targets)
    return float(rho) if not np.isnan(rho) else 0.0

def build_model(arch_type):
    if arch_type == "A":
        return ConcatMLP(kronos_dim=768, stock_dim=10, market_dim=4, h1_dim=256, h2_dim=64, dropout_rate=0.2)
    elif arch_type == "B":
        return CrossAttentionGating(kronos_dim=768, stock_dim=10, num_market_tokens=4, d_model=256, n_heads=4, dropout_rate=0.15)
    elif arch_type == "C":
        return FiLMConditioning(kronos_dim=768, stock_dim=10, market_dim=4, film_hidden_dim=64, h1_dim=256, h2_dim=64, dropout_rate=0.2)

def run_experiment(arch_type):
    print(f"\n===================================================================")
    print(f"STARTING EVALUATION FOR ARCHITECTURE {arch_type}")
    print(f"===================================================================")

    WINDOW_SIZE = 250
    unique_dates = sorted(df['date'].unique())
    test_dates = [d for d in unique_dates if d >= pd.Timestamp("2024-01-01")]

    if args.smoke_test:
        test_dates = test_dates[:5]
        print(f"[SMOKE TEST MODE] Running first 5 test dates: {[d.strftime('%Y-%m-%d') for d in test_dates]}")

    daily_results = []
    prediction_records = []

    for test_date in tqdm(test_dates, desc=f"Walk-Forward Arch {arch_type}"):
        curr_pos = unique_dates.index(test_date)
        if curr_pos < WINDOW_SIZE:
            continue

        window_dates = unique_dates[curr_pos - WINDOW_SIZE : curr_pos]
        train_cutoff = int(len(window_dates) * 0.8)
        train_dates = window_dates[:train_cutoff]
        val_dates = window_dates[train_cutoff:]

        train_mask = df['date'].isin(train_dates)
        val_mask = df['date'].isin(val_dates)
        test_mask = (df['date'] == test_date)

        train_k = df.loc[train_mask, kronos_cols].values
        train_s = df.loc[train_mask, stock_cols].values
        train_m = df.loc[train_mask, market_cols].values
        train_y = df.loc[train_mask, 'target_return_1d'].values

        val_k = df.loc[val_mask, kronos_cols].values
        val_s = df.loc[val_mask, stock_cols].values
        val_m = df.loc[val_mask, market_cols].values
        val_y = df.loc[val_mask, 'target_return_1d'].values

        test_k = df.loc[test_mask, kronos_cols].values
        test_s = df.loc[test_mask, stock_cols].values
        test_m = df.loc[test_mask, market_cols].values
        test_y = df.loc[test_mask, 'target_return_1d'].values
        test_symbols = df.loc[test_mask, 'symbol'].values

        train_loader = DataLoader(FastDataset(train_k, train_s, train_m, train_y), batch_size=256, shuffle=True)
        val_k_t = torch.tensor(val_k, dtype=torch.float32).to(DEVICE)
        val_s_t = torch.tensor(val_s, dtype=torch.float32).to(DEVICE)
        val_m_t = torch.tensor(val_m, dtype=torch.float32).to(DEVICE)

        model = build_model(arch_type).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-4, weight_decay=1e-4)
        criterion = CrossSectionalCorrelationLoss()

        best_val_ic = -999.0
        best_weights = None
        no_improve = 0

        for epoch in range(35):
            model.train()
            for b_k, b_s, b_m, b_y in train_loader:
                b_k, b_s, b_m, b_y = b_k.to(DEVICE), b_s.to(DEVICE), b_m.to(DEVICE), b_y.to(DEVICE)
                optimizer.zero_grad()
                out = model(b_k, b_s, b_m)
                loss = criterion(out, b_y)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                v_pred = model(val_k_t, val_s_t, val_m_t).cpu().numpy()
                v_ic = compute_rank_ic(v_pred, val_y)

            if v_ic > best_val_ic:
                best_val_ic = v_ic
                best_weights = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 10:
                    break

        if best_weights is not None:
            model.load_state_dict(best_weights)

        model.eval()
        with torch.no_grad():
            t_k = torch.tensor(test_k, dtype=torch.float32).to(DEVICE)
            t_s = torch.tensor(test_s, dtype=torch.float32).to(DEVICE)
            t_m = torch.tensor(test_m, dtype=torch.float32).to(DEVICE)
            test_preds = model(t_k, t_s, t_m).cpu().numpy()

        day_ic = compute_rank_ic(test_preds, test_y)

        day_df = pd.DataFrame({'symbol': test_symbols, 'pred': test_preds, 'actual': test_y}).sort_values('pred', ascending=False).reset_index(drop=True)
        top10_ret = day_df.iloc[:10]['actual'].mean()
        bottom10_ret = day_df.iloc[-10:]['actual'].mean()
        ls_spread = top10_ret - bottom10_ret

        daily_results.append({
            'date': test_date,
            'rank_ic': day_ic,
            'ls_spread': ls_spread,
            'top10_ret': top10_ret,
            'bottom10_ret': bottom10_ret
        })

        for _, row in day_df.iterrows():
            prediction_records.append({
                'date': test_date,
                'symbol': row['symbol'],
                'pred_return': float(row['pred']),
                'actual_return': float(row['actual'])
            })

    results_df = pd.DataFrame(daily_results)
    preds_df = pd.DataFrame(prediction_records)

    file_suffix = f"arch_{arch_type.lower()}" + ("_smoke" if args.smoke_test else "")
    results_df.to_parquet(RESULTS_DIR / f"walk_forward_daily_metrics_{file_suffix}.parquet", index=False, compression="snappy")
    preds_df.to_parquet(RESULTS_DIR / f"predictions_{file_suffix}.parquet", index=False, compression="snappy")

    mean_ic = results_df['rank_ic'].mean()
    std_ic = results_df['rank_ic'].std()
    icir = mean_ic / (std_ic + 1e-8)
    mean_ls = results_df['ls_spread'].mean()
    std_ls = results_df['ls_spread'].std()
    sharpe = (mean_ls / (std_ls + 1e-8)) * np.sqrt(252.0)
    downside_returns = results_df['ls_spread'][results_df['ls_spread'] < 0]
    sortino = (mean_ls / (downside_returns.std() + 1e-8)) * np.sqrt(252.0)
    cum_returns = (1.0 + results_df['ls_spread']).cumprod()
    max_dd = ((cum_returns - cum_returns.cummax()) / cum_returns.cummax()).min()

    summary_text = f"""
===========================================================================
ARCH {arch_type} RESULTS {'[SMOKE TEST]' if args.smoke_test else ''}
===========================================================================
Days Evaluated:            {len(results_df)}
Mean Daily Rank IC:        {mean_ic:.4f}
ICIR (Rank IC / Std IC):   {icir:.4f}
Annualized L/S Return:     {mean_ls * 252.0 * 100:.2f}%
Sharpe Ratio:              {sharpe:.2f}
Sortino Ratio:             {sortino:.2f}
Max Drawdown:              {max_dd * 100:.2f}%
===========================================================================
"""
    print(summary_text)

targets = ["A", "B", "C"] if args.arch == "all" else [args.arch]
for target in targets:
    run_experiment(target)
