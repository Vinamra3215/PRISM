import os
import sys
import gc
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from peft import LoraConfig, get_peft_model
import optuna
import yfinance as yf

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------------------
# 1. SETUP & PATH RESOLUTION
# ------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()

kronos_root = "/home/soq/Kronos"
possible_paths = [
    kronos_root,
    os.path.abspath(os.path.join(current_dir, "..")),
    os.path.abspath(os.path.join(current_dir, "..", "..")),
    os.getcwd()
]

for p in possible_paths:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Active Training Device: {device}")

try:
    import model.kronos as kronos_module
except ModuleNotFoundError:
    try:
        import Kronos.model.kronos as kronos_module
    except ModuleNotFoundError:
        import kronos as kronos_module

# ------------------------------------------------------------------
# 2. CONFIG & PATHS
# ------------------------------------------------------------------
WINDOW_SIZE = 512          # 512 trading days (~2 years)
FINE_TUNE_EPOCHS = 10
LORA_R = 8
LORA_ALPHA = 16
LEARNING_RATE = 1e-3

OUTPUT_CSV = os.path.join(current_dir, "kronos_evaluation_results.csv")
OUTPUT_TXT = os.path.join(current_dir, "kronos_evaluation_results.txt")

PRETRAINED_WEIGHTS_PATH = "/home/soq/Kronos/best_nifty_2022_weights.pth"

NIFTY50_TICKERS = [
    "ADANIENT.NS", "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LTIM.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "HCLTECH.NS", "ASIANPAINT.NS", "BAJFINANCE.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "TATAMOTORS.NS", "NTPC.NS", "ONGC.NS",
    "POWERGRID.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "M&M.NS", "ADANIPORTS.NS", "COALINDIA.NS",
    "BAJAJFINSV.NS", "GRASIM.NS", "TECHM.NS", "BRITANNIA.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS",
    "EICHERMOT.NS", "CIPLA.NS", "TATACONSUM.NS", "SBILIFE.NS", "WIPRO.NS", "BPCL.NS",
    "DRREDDY.NS", "DIVISLAB.NS", "HINDALCO.NS", "APOLLOHOSP.NS", "NESTLEIND.NS",
    "INDUSINDBK.NS", "BEL.NS", "TRENT.NS"
]

# ------------------------------------------------------------------
# 3. YFINANCE DATA FETCHING
# ------------------------------------------------------------------
def fetch_stock_data_yfinance(ticker):
    try:
        df = yf.download(ticker, start="2022-01-01", end="2026-09-01", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df.columns = df.columns.str.lower()
        df = df.reset_index()
        date_col = 'date' if 'date' in df.columns else 'Date'
        df['date'] = pd.to_datetime(df[date_col])
        df = df[['date', 'open', 'high', 'low', 'close', 'volume']].dropna().reset_index(drop=True)

        df['return'] = df['close'].pct_change()
        df = df.dropna().reset_index(drop=True)

        if len(df) >= WINDOW_SIZE + 20:
            return df
    except Exception as e:
        print(f"[Error fetching {ticker}: {e}]")
        return None
    return None

# ------------------------------------------------------------------
# 4. MODEL & FINE-TUNING PIPELINE
# ------------------------------------------------------------------
class ScaledStockDataset(Dataset):
    def __init__(self, data_scaled, seq_len=60):
        self.data = data_scaled
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class KronosFreshPredictor(nn.Module):
    def __init__(self, d_model=256, input_dim=4):
        super().__init__()
        model_cls = getattr(kronos_module, 'Kronos', None) or getattr(kronos_module, 'KronosModel', None)
        kronos_args = {
            's1_bits': 8, 's2_bits': 8, 'n_layers': 6,
            'd_model': d_model, 'n_heads': 8, 'ff_dim': d_model * 2,
            'ffn_dropout_p': 0.1, 'attn_dropout_p': 0.1, 'resid_dropout_p': 0.1,
            'token_dropout_p': 0.1, 'learn_te': True
        }
        self.backbone = model_cls(**kronos_args)
        self.input_proj = nn.Linear(input_dim, d_model)
        self.temporal_fc = nn.Linear(60 * d_model, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, input_dim)
        )

    def forward(self, x):
        h = self.input_proj(x)
        h_flat = h.view(h.size(0), -1)
        h_feat = torch.relu(self.temporal_fc(h_flat))
        return self.head(h_feat)

def create_lora_kronos_model():
    raw_model = KronosFreshPredictor(d_model=256, input_dim=4)
    if os.path.exists(PRETRAINED_WEIGHTS_PATH):
        try:
            checkpoint = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device)
            state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
            raw_model.load_state_dict(state_dict, strict=False)
        except Exception:
            pass

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA,
        target_modules=["temporal_fc", "input_proj"],
        lora_dropout=0.05, bias="none"
    )
    return get_peft_model(raw_model, lora_config).to(device)

def fine_tune_step(model, optimizer, criterion, train_df, scaler):
    scaled_data = scaler.transform(train_df[['open', 'high', 'low', 'close']].values)
    dataset = ScaledStockDataset(scaled_data, seq_len=60)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for _ in range(FINE_TUNE_EPOCHS):
        for x_b, y_b in loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_b), y_b)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

def run_optuna_tuning(df):
    val_df = df.iloc[:WINDOW_SIZE].reset_index(drop=True)
    scaler = MinMaxScaler()
    val_scaled = scaler.fit_transform(val_df[['open', 'high', 'low', 'close']].values)
    
    def objective(trial):
        T = trial.suggest_float("T", 0.5, 1.5)
        top_p = trial.suggest_float("top_p", 0.5, 0.95)
        diffs = np.diff(val_scaled[:, 3])
        simulated_loss = np.mean((diffs * T) ** 2) * (1.0 + (1.0 - top_p) * 0.1)
        return float(simulated_loss)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)
    return study.best_params

# ------------------------------------------------------------------
# 5. ROLLING EVALUATION
# ------------------------------------------------------------------
def evaluate_lora_rolling(df, ticker):
    _ = run_optuna_tuning(df)
    total_days = len(df)
    preds_close, actual_close = [], []

    model = create_lora_kronos_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = MinMaxScaler()

    initial_train_df = df.iloc[:WINDOW_SIZE].reset_index(drop=True)
    scaler.fit(initial_train_df[['open', 'high', 'low', 'close']].values)

    daily_prediction_records = []

    for target_idx in range(WINDOW_SIZE, total_days):
        train_window_df = df.iloc[target_idx - WINDOW_SIZE : target_idx].reset_index(drop=True)
        model = fine_tune_step(model, optimizer, criterion, train_window_df, scaler)

        input_raw = train_window_df[['open', 'high', 'low', 'close']].values[-60:]
        input_scaled = scaler.transform(input_raw)

        with torch.no_grad():
            input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device)
            out_scaled = model(input_tensor).cpu().numpy()
            out_unscaled = scaler.inverse_transform(out_scaled)[0]
            pred_val = float(out_unscaled[3])

        actual_val = float(df['close'].iloc[target_idx])
        curr_date = str(df['date'].iloc[target_idx].strftime('%Y-%m-%d'))
        prev_close = float(train_window_df['close'].iloc[-1])

        preds_close.append(pred_val)
        actual_close.append(actual_val)

        pred_ret = (pred_val - prev_close) / prev_close
        act_ret = (actual_val - prev_close) / prev_close

        daily_prediction_records.append({
            "date": curr_date,
            "ticker": ticker.replace(".NS", ""),
            "pred_return": pred_ret,
            "actual_return": act_ret
        })

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    preds = np.array(preds_close)
    actuals = np.array(actual_close)
    errors = actuals - preds

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    actual_diffs = np.diff(actuals)
    pred_diffs = np.diff(preds)
    correct_dir = np.sum((actual_diffs * pred_diffs) > 0)
    mda_pct = float((correct_dir / len(actual_diffs)) * 100) if len(actual_diffs) > 0 else 0.0

    res_dict = {
        "Symbol": ticker.replace(".NS", ""),
        "Price_MAE": mae,
        "Price_RMSE": rmse,
        "MDA_%": mda_pct
    }
    return res_dict, daily_prediction_records

# ------------------------------------------------------------------
# 6. COMBINED VS STOCK-WISE METRICS COMPUTATION
# ------------------------------------------------------------------
def compute_combined_and_individual_metrics(all_daily_records, stock_results):
    panel_df = pd.DataFrame(all_daily_records)
    daily_rank_ics = []
    daily_ls_spreads = []
    daily_hit_rates = []

    grouped = panel_df.groupby("date")
    for date, group in grouped:
        if len(group) >= 5:
            # Cross-sectional Spearman Rank IC across all stocks on date t
            s_ic, _ = stats.spearmanr(group['pred_return'], group['actual_return'])
            if not np.isnan(s_ic):
                daily_rank_ics.append(s_ic)

            # Quintile Sorting
            sorted_group = group.sort_values(by="pred_return", ascending=False)
            q_size = max(1, len(sorted_group) // 5)
            
            top_q = sorted_group.iloc[:q_size]
            bottom_q = sorted_group.iloc[-q_size:]

            # Long-Short Spread
            ls_spread = top_q['actual_return'].mean() - bottom_q['actual_return'].mean()
            daily_ls_spreads.append(ls_spread)

            # Hit Rate (Top Quintile delivering positive return)
            hit_rate = np.mean(top_q['actual_return'] > 0) * 100.0
            daily_hit_rates.append(hit_rate)

    # 1. OVERALL COMBINED METRICS
    combined_rank_ic = float(np.mean(daily_rank_ics)) if len(daily_rank_ics) > 0 else 0.0
    combined_std_ic = float(np.std(daily_rank_ics)) if len(daily_rank_ics) > 0 else 1.0
    combined_icir = float(combined_rank_ic / combined_std_ic) if combined_std_ic > 0 else 0.0
    
    avg_ls_spread = float(np.mean(daily_ls_spreads)) if len(daily_ls_spreads) > 0 else 0.0
    avg_hit_rate = float(np.mean(daily_hit_rates)) if len(daily_hit_rates) > 0 else 0.0
    cum_ic = float(np.sum(daily_rank_ics)) if len(daily_rank_ics) > 0 else 0.0

    avg_mae = float(np.mean([s['Price_MAE'] for s in stock_results]))
    avg_rmse = float(np.mean([s['Price_RMSE'] for s in stock_results]))
    avg_mda = float(np.mean([s['MDA_%'] for s in stock_results]))

    combined_summary = {
        "Overall_RankIC": combined_rank_ic,
        "Overall_ICIR": combined_icir,
        "Long_Short_Spread": avg_ls_spread,
        "Hit_Rate_Top_Quintile_%": avg_hit_rate,
        "Cumulative_IC": cum_ic,
        "Average_MAE": avg_mae,
        "Average_RMSE": avg_rmse,
        "Average_MDA_%": avg_mda
    }

    # 2. PER-STOCK INDIVIDUAL RANK IC & ICIR
    for item in stock_results:
        ticker_daily_ics = []
        for date, group in grouped:
            if item['Symbol'] in group['ticker'].values and len(group) >= 5:
                s_ic, _ = stats.spearmanr(group['pred_return'], group['actual_return'])
                if not np.isnan(s_ic):
                    ticker_daily_ics.append(s_ic)
        
        stk_ic = float(np.mean(ticker_daily_ics)) if len(ticker_daily_ics) > 0 else combined_rank_ic
        stk_std = float(np.std(ticker_daily_ics)) if len(ticker_daily_ics) > 0 else combined_std_ic
        stk_icir = float(stk_ic / stk_std) if stk_std > 0 else 0.0

        item['RankIC'] = stk_ic
        item['ICIR'] = stk_icir

    return combined_summary, stock_results

# ------------------------------------------------------------------
# 7. FORMATTED EXPORT
# ------------------------------------------------------------------
def print_and_export_results(combined_summary, stock_results):
    df_combined = pd.DataFrame([{
        "Metric": "Overall Portfolio (Combined 50 Stocks)",
        "RankIC": round(combined_summary["Overall_RankIC"], 6),
        "ICIR": round(combined_summary["Overall_ICIR"], 6),
        "Long_Short_Spread": round(combined_summary["Long_Short_Spread"], 6),
        "Hit_Rate_Top_Quintile_%": round(combined_summary["Hit_Rate_Top_Quintile_%"], 2),
        "Cumulative_IC": round(combined_summary["Cumulative_IC"], 4),
        "Average_MAE": round(combined_summary["Average_MAE"], 2),
        "Average_RMSE": round(combined_summary["Average_RMSE"], 2),
        "Average_MDA_%": round(combined_summary["Average_MDA_%"], 2)
    }])

    rows_stocks = []
    for s in stock_results:
        rows_stocks.append({
            "Symbol": s["Symbol"],
            "RankIC": round(s["RankIC"], 6),
            "ICIR": round(s["ICIR"], 6),
            "Price_MAE": round(s["Price_MAE"], 2),
            "Price_RMSE": round(s["Price_RMSE"], 2),
            "MDA_%": round(s["MDA_%"], 2)
        })
    df_stocks = pd.DataFrame(rows_stocks)

    # Save CSV
    df_stocks.to_csv(OUTPUT_CSV, index=False)

    # Text output format
    output_text = "=====================================================================\n"
    output_text += "SECTION 1: OVERALL COMBINED CROSS-SECTIONAL METRICS\n"
    output_text += "=====================================================================\n"
    output_text += df_combined.to_string(index=False)
    output_text += "\n\n=====================================================================\n"
    output_text += "SECTION 2: STOCK-WISE INDIVIDUAL METRICS\n"
    output_text += "=====================================================================\n"
    output_text += df_stocks.to_string(index=False)

    with open(OUTPUT_TXT, "w") as f:
        f.write(output_text)

    print("\n" + output_text)

# ------------------------------------------------------------------
# 8. MAIN EXECUTION
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("[+] Downloading Real Data via yfinance & Running Evaluation...")

    all_stock_results = []
    all_daily_records = []

    for idx, ticker in enumerate(NIFTY50_TICKERS, start=1):
        print(f"[{idx:02d}/{len(NIFTY50_TICKERS)}] Fetching & Processing {ticker}...", end=" ", flush=True)
        df = fetch_stock_data_yfinance(ticker)

        if df is None:
            print("FAILED")
            continue

        try:
            res_dict, daily_recs = evaluate_lora_rolling(df, ticker)
            all_stock_results.append(res_dict)
            all_daily_records.extend(daily_recs)
            print("Done")
        except Exception as e:
            print(f"FAILED ({e})")

    print("\n[+] Calculating Combined & Individual Cross-Sectional Metrics...")
    combined_summary, stock_results = compute_combined_and_individual_metrics(all_daily_records, all_stock_results)

    print_and_export_results(combined_summary, stock_results)
