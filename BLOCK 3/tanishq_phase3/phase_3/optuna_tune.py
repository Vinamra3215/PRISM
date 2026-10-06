import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr
import optuna

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("/home/soq/__shutupandbendover/tanishq/phase3_data")
OUTPUT_DIR = Path("/home/soq/__shutupandbendover/tanishq/phase3_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Using device: {DEVICE}")

# -------------------------------------------------------------------------
# 1. Load Pre-Extracted Datasets
# -------------------------------------------------------------------------
print("Loading feature matrix and Kronos embeddings...")
feat_df = pd.read_parquet(DATA_DIR / "prepared_features_matrix.parquet")
emb_df = pd.read_parquet(DATA_DIR / "kronos_embeddings_768d.parquet")

feat_df['date'] = pd.to_datetime(feat_df['date'])
emb_df['date'] = pd.to_datetime(emb_df['date'])

# Merge features and embeddings on (date, symbol)
df = pd.merge(feat_df, emb_df, on=['date', 'symbol'], how='inner')
df = df.sort_values(['date', 'symbol']).reset_index(drop=True)

# Define feature subsets
market_cols = ['nifty_daily_return', 'nifty_5d_return', 'nifty_vol_20d', 'pharma_return']
stock_cols = [
    'ma_ratio_50_200', 'rolling_volatility_10d', 'intraday_range',
    'keltner_channel_position', 'trend_strength', 'cs_return_z',
    'volume_price_trend', 'volatility_ratio_5d_20d', 'sector_relative_return',
    'rolling_skewness_20d'
]
kronos_cols = [f'kronos_emb_{i}' for i in range(768)]

# Fill any remaining NaNs
df[stock_cols] = df[stock_cols].fillna(0.0)
df[market_cols] = df[market_cols].fillna(0.0)

# -------------------------------------------------------------------------
# 2. Date Slices for Optuna (2022 Train, Jan-Jul 2023 Val)
# -------------------------------------------------------------------------
train_df = df[(df['date'] >= '2022-01-01') & (df['date'] <= '2022-12-31')].copy()
val_df = df[(df['date'] >= '2023-01-01') & (df['date'] <= '2023-07-31')].copy()

print(f"Optuna Train Samples (2022): {len(train_df)}")
print(f"Optuna Val Samples (Jan-Jul 2023): {len(val_df)}")

# -------------------------------------------------------------------------
# 3. PyTorch Dataset Definition
# -------------------------------------------------------------------------
class StockDataset(Dataset):
    def __init__(self, data):
        self.k_emb = torch.tensor(data[kronos_cols].values, dtype=torch.float32)
        self.s_feat = torch.tensor(data[stock_cols].values, dtype=torch.float32)
        self.m_feat = torch.tensor(data[market_cols].values, dtype=torch.float32)
        self.target = torch.tensor(data['target_return_1d'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        return self.k_emb[idx], self.s_feat[idx], self.m_feat[idx], self.target[idx]

train_dataset = StockDataset(train_df)
val_dataset = StockDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# -------------------------------------------------------------------------
# 4. Architecture B: Master Cross-Attention Gating Head
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
        # Gate dimension: concatenating Q (d_model) + attn_out (d_model) = 2 * d_model
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
        
        stock_repr = torch.cat([k_proj, s_proj], dim=-1) # [B, 2 * d_model]
        Q = self.proj_query(stock_repr)                  # [B, d_model]
        
        market_tokens = market_feat.unsqueeze(-1)        # [B, num_tokens, 1]
        KV = self.proj_market(market_tokens)            # [B, num_tokens, d_model]
        
        # Cross attention expects Q shape: [B, 1, d_model]
        attn_out, _ = self.cross_attn(query=Q.unsqueeze(1), key=KV, value=KV)
        attn_out = attn_out.squeeze(1)                  # [B, d_model]
        
        # Gating between stock representation (Q) and market-attended output (attn_out)
        gate_input = torch.cat([Q, attn_out], dim=-1)   # [B, 2 * d_model]
        gate = self.sigmoid(self.gate_dense(gate_input))
        gated_repr = gate * attn_out + (1.0 - gate) * Q
        
        return self.head(gated_repr).squeeze(-1)

# -------------------------------------------------------------------------
# 5. Daily Cross-Sectional Rank IC Evaluation
# -------------------------------------------------------------------------
def eval_rank_ic(model, val_loader, val_df):
    model.eval()
    preds = []
    with torch.no_grad():
        for k_emb, s_feat, m_feat, _ in val_loader:
            k_emb, s_feat, m_feat = k_emb.to(DEVICE), s_feat.to(DEVICE), m_feat.to(DEVICE)
            out = model(k_emb, s_feat, m_feat)
            preds.extend(out.cpu().numpy())
    
    val_eval_df = val_df[['date', 'target_return_1d']].copy()
    val_eval_df['pred'] = preds
    
    daily_ics = []
    for d, group in val_eval_df.groupby('date'):
        if len(group) >= 10 and group['pred'].nunique() > 1:
            rho, _ = spearmanr(group['pred'], group['target_return_1d'])
            if not np.isnan(rho):
                daily_ics.append(rho)
                
    return float(np.mean(daily_ics)) if len(daily_ics) > 0 else -1.0

# -------------------------------------------------------------------------
# 6. Optuna Search Objective
# -------------------------------------------------------------------------
def objective(trial):
    d_model = trial.suggest_categorical("d_model", [64, 128, 256])
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    
    model = CrossAttentionGating(
        kronos_dim=768, stock_dim=len(stock_cols), num_market_tokens=len(market_cols),
        d_model=d_model, n_heads=n_heads, dropout_rate=dropout
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    # Train for 5 epochs per trial to quickly compare hyperparameters
    for epoch in range(5):
        model.train()
        for k_emb, s_feat, m_feat, target in train_loader:
            k_emb, s_feat, m_feat, target = k_emb.to(DEVICE), s_feat.to(DEVICE), m_feat.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            out = model(k_emb, s_feat, m_feat)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            
    val_ic = eval_rank_ic(model, val_loader, val_df)
    return val_ic

if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="maximize")
    print("\nStarting Optuna Hyperparameter Optimization (20 trials)...")
    study.optimize(objective, n_trials=20)
    
    print("\n" + "=" * 60)
    print("OPTUNA BEST HYPERPARAMETERS:")
    print("=" * 60)
    for k, v in study.best_params.items():
        print(f"{k}: {v}")
    print(f"Best Validation Rank IC: {study.best_value:.4f}")
    print("=" * 60)
    
    import json
    with open(OUTPUT_DIR / "best_optuna_params_arch_b.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    print(f"Locked parameters saved to: {OUTPUT_DIR / 'best_optuna_params_arch_b.json'}")