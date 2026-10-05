import os
import json
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

# -------------------------------------------------------------------------
# 1. Load Pre-Extracted Datasets & Locked Hyperparameters
# -------------------------------------------------------------------------
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

# Load locked Optuna parameters
with open(RESULTS_DIR / "best_optuna_params_arch_b.json", "r") as f:
    optuna_params = json.load(f)

d_model = int(optuna_params['d_model'])
n_heads = int(optuna_params['n_heads'])
lr = float(optuna_params['lr'])
dropout = float(optuna_params['dropout'])

print(f"Locked Params: d_model={d_model}, n_heads={n_heads}, lr={lr:.6f}, dropout={dropout:.4f}")

# -------------------------------------------------------------------------
# 2. PyTorch Architecture B & Dataset
# -------------------------------------------------------------------------
class CrossAttentionGating(nn.Module):
    def __init__(self, kronos_dim=768, stock_dim=10, num_market_tokens=4, 
                 d_model=128, n_heads=4, dropout_rate=0.15):
        super().__init__()
        self.layer_norm = nn.LayerNorm(kronos_dim)
        self.proj_kronos = nn.Linear(kronos_dim, d_model)
        self.proj_stock = nn.Linear(stock_dim, d_model)
        self.proj_query = nn.Linear(2 * d_model, d_model)
        self.proj_market = nn.Linear(1, d_model)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout_rate, batch_first=True
        )
        self.gate_dense = nn.Linear(2 * d_model, d_model)
        self.sigmoid = nn.Sigmoid()
        
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
    def forward(self, kronos_emb, stock_feat, market_feat):
        k_norm = self.layer_norm(kronos_emb)
        k_proj = self.proj_kronos(k_norm)
        s_proj = self.proj_stock(stock_feat)
        
        stock_repr = torch.cat([k_proj, s_proj], dim=-1)
        Q = self.proj_query(stock_repr)
        
        market_tokens = market_feat.unsqueeze(-1)
        KV = self.proj_market(market_tokens)
        
        attn_out, _ = self.cross_attn(query=Q.unsqueeze(1), key=KV, value=KV)
        attn_out = attn_out.squeeze(1)
        
        gate_input = torch.cat([Q, attn_out], dim=-1)
        gate = self.sigmoid(self.gate_dense(gate_input))
        gated_repr = gate * attn_out + (1.0 - gate) * Q
        
        return self.head(gated_repr).squeeze(-1)

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
# 3. Daily Walk-Forward Rolling Retraining Loop (2024 - Present)
# -------------------------------------------------------------------------
WINDOW_SIZE = 250   # 250 trading days lookback (~1 trading year)
unique_dates = sorted(df['date'].unique())
test_dates = [d for d in unique_dates if d >= pd.Timestamp("2024-01-01")]

print(f"Total Unique Trading Dates: {len(unique_dates)}")
print(f"Total Test Dates to Walk-Forward: {len(test_dates)}")

daily_results = []
prediction_records = []

for t_idx, test_date in enumerate(tqdm(test_dates, desc="Walk-Forward Daily Retraining")):
    curr_pos = unique_dates.index(test_date)
    if curr_pos < WINDOW_SIZE:
        continue
    
    # Define Lookback Window [t - WINDOW_SIZE, t - 1]
    window_dates = unique_dates[curr_pos - WINDOW_SIZE : curr_pos]
    train_cutoff = int(len(window_dates) * 0.8)
    train_window_dates = window_dates[:train_cutoff]
    val_window_dates = window_dates[train_cutoff:]
    
    train_mask = df['date'].isin(train_window_dates)
    val_mask = df['date'].isin(val_window_dates)
    test_mask = (df['date'] == test_date)
    
    # Prepare Arrays
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
    
    # DataLoaders
    train_loader = DataLoader(FastDataset(train_k, train_s, train_m, train_y), batch_size=256, shuffle=True)
    val_k_t = torch.tensor(val_k, dtype=torch.float32).to(DEVICE)
    val_s_t = torch.tensor(val_s, dtype=torch.float32).to(DEVICE)
    val_m_t = torch.tensor(val_m, dtype=torch.float32).to(DEVICE)
    
    # Initialize fresh weights from scratch
    model = CrossAttentionGating(
        kronos_dim=768, stock_dim=len(stock_cols), num_market_tokens=len(market_cols),
        d_model=d_model, n_heads=n_heads, dropout_rate=dropout
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Early stopping tracking
    best_val_ic = -999.0
    best_weights = None
    no_improve = 0
    max_epochs = 40
    
    for epoch in range(max_epochs):
        model.train()
        for b_k, b_s, b_m, b_y in train_loader:
            b_k, b_s, b_m, b_y = b_k.to(DEVICE), b_s.to(DEVICE), b_m.to(DEVICE), b_y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(b_k, b_s, b_m)
            loss = criterion(pred, b_y)
            loss.backward()
            optimizer.step()
            
        # Validation Rank IC
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
                
    # Load best checkpoint
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    # Predict on Day t
    model.eval()
    with torch.no_grad():
        t_k = torch.tensor(test_k, dtype=torch.float32).to(DEVICE)
        t_s = torch.tensor(test_s, dtype=torch.float32).to(DEVICE)
        t_m = torch.tensor(test_m, dtype=torch.float32).to(DEVICE)
        test_preds = model(t_k, t_s, t_m).cpu().numpy()
        
    # Evaluate Day t
    day_ic = compute_rank_ic(test_preds, test_y)
    
    # Long-Short Quintile Portfolio: Top 10 Long, Bottom 10 Short
    day_df = pd.DataFrame({'symbol': test_symbols, 'pred': test_preds, 'actual': test_y})
    day_df = day_df.sort_values('pred', ascending=False).reset_index(drop=True)
    
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

# -------------------------------------------------------------------------
# 4. Generate Performance Metrics Table
# -------------------------------------------------------------------------
results_df = pd.DataFrame(daily_results)
preds_df = pd.DataFrame(prediction_records)

# Save Parquet outputs
results_df.to_parquet(RESULTS_DIR / "walk_forward_daily_metrics_arch_b.parquet", index=False)
preds_df.to_parquet(RESULTS_DIR / "walk_forward_predictions_arch_b.parquet", index=False)

# Metric Calculations
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
peak = cum_returns.cummax()
drawdown = (cum_returns - peak) / peak
max_dd = drawdown.min()

win_rate = (results_df['ls_spread'] > 0).mean() * 100.0

summary_report = f"""
===========================================================================
PRISM PHASE 3: ARCHITECTURE B (CROSS-ATTENTION GATING) WALK-FORWARD SUMMARY
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

with open(RESULTS_DIR / "arch_b_performance_summary.txt", "w") as f:
    f.write(summary_report)