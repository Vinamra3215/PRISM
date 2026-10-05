import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("/home/soq/__shutupandbendover/tanishq/phase3_data")
RESULTS_DIR = Path("/home/soq/__shutupandbendover/tanishq/phase3_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("Loading feature matrix and Kronos embeddings...")
feat_df = pd.read_parquet(DATA_DIR / "prepared_features_matrix.parquet")
emb_df = pd.read_parquet(DATA_DIR / "kronos_embeddings_768d.parquet")

feat_df['date'] = pd.to_datetime(feat_df['date'])
emb_df['date'] = pd.to_datetime(emb_df['date'])

df = pd.merge(feat_df, emb_df, on=['date', 'symbol'], how='inner')
df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

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

# =========================================================================
# Architecture C: FiLM Conditioning (Feature-wise Linear Modulation)
# =========================================================================
class FiLMConditioning(nn.Module):
    def __init__(self, kronos_dim=768, stock_dim=10, market_dim=4, 
                 film_hidden_dim=64, h1_dim=256, h2_dim=64, dropout_rate=0.2):
        super().__init__()
        self.layer_norm = nn.LayerNorm(kronos_dim)
        
        # FiLM Generator: produces gamma (scale) and beta (shift) from market context
        self.film_backbone = nn.Sequential(
            nn.Linear(market_dim, film_hidden_dim),
            nn.ReLU()
        )
        self.to_gamma = nn.Linear(film_hidden_dim, kronos_dim)
        self.to_beta = nn.Linear(film_hidden_dim, kronos_dim)
        
        # Prediction Network
        in_dim = kronos_dim + stock_dim
        self.pred_net = nn.Sequential(
            nn.Linear(in_dim, h1_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h2_dim, 1)
        )
        
    def forward(self, kronos_emb, stock_feat, market_feat):
        k_norm = self.layer_norm(kronos_emb)
        
        # Affine scale (gamma) and shift (beta)
        film_h = self.film_backbone(market_feat)
        gamma = self.to_gamma(film_h)
        beta = self.to_beta(film_h)
        
        # Modulate Kronos representation directly
        modulated_kronos = gamma * k_norm + beta
        
        combined = torch.cat([modulated_kronos, stock_feat], dim=-1)
        return self.pred_net(combined).squeeze(-1)

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

# -------------------------------------------------------------------------
# Daily Walk-Forward Rolling Retraining Loop (2024 - Present)
# -------------------------------------------------------------------------
WINDOW_SIZE = 250
unique_dates = sorted(df['date'].unique())
test_dates = [d for d in unique_dates if d >= pd.Timestamp("2024-01-01")]

print(f"Total Unique Trading Dates: {len(unique_dates)}")
print(f"Total Test Dates to Walk-Forward: {len(test_dates)}")

daily_results = []
prediction_records = []

for t_idx, test_date in enumerate(tqdm(test_dates, desc="Walk-Forward Arch C")):
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
    
    train_k, train_s, train_m, train_y = df.loc[train_mask, kronos_cols].values, df.loc[train_mask, stock_cols].values, df.loc[train_mask, market_cols].values, df.loc[train_mask, 'target_return_1d'].values
    val_k, val_s, val_m, val_y = df.loc[val_mask, kronos_cols].values, df.loc[val_mask, stock_cols].values, df.loc[val_mask, market_cols].values, df.loc[val_mask, 'target_return_1d'].values
    test_k, test_s, test_m, test_y = df.loc[test_mask, kronos_cols].values, df.loc[test_mask, stock_cols].values, df.loc[test_mask, market_cols].values, df.loc[test_mask, 'target_return_1d'].values
    test_symbols = df.loc[test_mask, 'symbol'].values
    
    train_loader = DataLoader(FastDataset(train_k, train_s, train_m, train_y), batch_size=256, shuffle=True)
    val_k_t, val_s_t, val_m_t = torch.tensor(val_k, dtype=torch.float32).to(DEVICE), torch.tensor(val_s, dtype=torch.float32).to(DEVICE), torch.tensor(val_m, dtype=torch.float32).to(DEVICE)
    
    model = FiLMConditioning(kronos_dim=768, stock_dim=len(stock_cols), market_dim=len(market_cols), film_hidden_dim=64, h1_dim=256, h2_dim=64, dropout_rate=0.2).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    best_val_ic = -999.0
    best_weights = None
    no_improve = 0
    
    for epoch in range(40):
        model.train()
        for b_k, b_s, b_m, b_y in train_loader:
            b_k, b_s, b_m, b_y = b_k.to(DEVICE), b_s.to(DEVICE), b_m.to(DEVICE), b_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(b_k, b_s, b_m)
            loss = criterion(pred, b_y)
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
            if no_improve >= 15:
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
            'pred_return': row['pred'],
            'actual_return': row['actual']
        })

results_df = pd.DataFrame(daily_results)
preds_df = pd.DataFrame(prediction_records)

results_df.to_parquet(RESULTS_DIR / "walk_forward_daily_metrics_arch_c.parquet", index=False)
preds_df.to_parquet(RESULTS_DIR / "predictions_arch_c.parquet", index=False)

mean_ic = results_df['rank_ic'].mean()
std_ic = results_df['rank_ic'].std()
icir = mean_ic / (std_ic + 1e-8)

mean_ls = results_df['ls_spread'].mean()
std_ls = results_df['ls_spread'].std()
annualized_ls = mean_ls * 252.0
sharpe = (mean_ls / (std_ls + 1e-8)) * np.sqrt(252.0)

downside_returns = results_df['ls_spread'][results_df['ls_spread'] < 0]
sortino = (mean_ls / (downside_returns.std() + 1e-8)) * np.sqrt(252.0)

cum_returns = (1.0 + results_df['ls_spread']).cumprod()
max_dd = ((cum_returns - cum_returns.cummax()) / cum_returns.cummax()).min()
win_rate = (results_df['ls_spread'] > 0).mean() * 100.0

summary_report = f"""
===========================================================================
PRISM PHASE 3: ARCHITECTURE C (FiLM CONDITIONING) WALK-FORWARD SUMMARY
===========================================================================
Test Window:               {results_df['date'].min().strftime('%Y-%m-%d')} to {results_df['date'].max().strftime('%Y-%m-%d')}
Total Test Days:           {len(results_df)}
Rolling Training Window:   {WINDOW_SIZE} trading days (Retrained from Scratch Daily)
---------------------------------------------------------------------------
TIER 1: RANKING METRICS
Mean Daily Rank IC:        {mean_ic:.4f}
ICIR (Rank IC / Std IC):   {icir:.4f}
Win Rate (LS > 0):         {win_rate:.2f}%
---------------------------------------------------------------------------
TIER 3: STRATEGY PERFORMANCE (Top 10 Long / Bottom 10 Short)
Mean Daily Long-Short:     {mean_ls * 100:.3f}%
Annualized Return:         {annualized_ls * 100:.2f}%
Sharpe Ratio:              {sharpe:.2f}
Sortino Ratio:             {sortino:.2f}
Maximum Drawdown:          {max_dd * 100:.2f}%
===========================================================================
"""

print(summary_report)

with open(RESULTS_DIR / "arch_c_performance_summary.txt", "w") as f:
    f.write(summary_report)