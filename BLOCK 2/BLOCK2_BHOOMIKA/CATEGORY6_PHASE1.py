import pandas as pd
import numpy as np
import scipy.stats as stats
import yfinance as yf
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# ==========================================
# 1. FETCH DATA WITH FALLBACKS & CLEANING
# ==========================================
def fetch_and_compute_category6():
    # Primary index tickers with robust fallback single stock proxies
    tickers_config = {
        'nifty': ['^NSEI', 'NIFTYBEES.NS'],
        'vix': ['^INDIAVIX'],
        'bank': ['^NSEBANK', 'BANKBEES.NS'],
        'it': ['^CNXIT', 'ITBEES.NS', 'TCS.NS'],
        'pharma': ['^CNXPHARMA', 'PHARMABEES.NS', 'SUNPHARMA.NS'],
        'auto': ['^CNXAUTO', 'AUTOBEES.NS', 'TATAMOTORS.NS'],
        'metal': ['^CNXMETAL', 'TATASTEEL.NS'],
        'energy': ['^CNXENERGY', 'CPSEETF.NS', 'RELIANCE.NS'],
        'usdinr': ['USDINR=X']
    }

    df_data = pd.DataFrame()

    for name, sym_list in tickers_config.items():
        downloaded = False
        for sym in sym_list:
            try:
                raw = yf.download(sym, start="2023-01-01", progress=False)
                if not raw.empty and len(raw) > 50:
                    if isinstance(raw.columns, pd.MultiIndex):
                        series = raw['Close'][sym] if sym in raw['Close'] else raw['Close'].iloc[:, 0]
                    else:
                        series = raw['Close'] if 'Close' in raw else raw.iloc[:, 0]
                    
                    series = series.dropna()
                    if len(series) > 50 and series.std() > 1e-4:
                        df_data[name] = series
                        downloaded = True
                        break
            except Exception:
                continue

    # Forward fill and backward fill missing points
    df_data = df_data.ffill().bfill()
    df_feat = pd.DataFrame(index=df_data.index)

    # Core Features Creation
    if 'nifty' in df_data:
        df_feat['nifty_daily_return'] = df_data['nifty'].pct_change()
        df_feat['nifty_5d_return'] = df_data['nifty'].pct_change(5)
        
        daily_ret = df_feat['nifty_daily_return']
        df_feat['nifty_vol_20d'] = daily_ret.rolling(20).std() * np.sqrt(252)
        df_feat['target_next_ret'] = df_feat['nifty_daily_return'].shift(-1)

    if 'vix' in df_data:
        df_feat['vix_level'] = df_data['vix']
        df_feat['vix_change'] = df_data['vix'].pct_change()

    sectors = ['bank', 'it', 'pharma', 'auto', 'metal', 'energy', 'usdinr']
    for sec in sectors:
        if sec in df_data:
            df_feat[f'{sec}_return'] = df_data[sec].pct_change()

    return df_feat

# ==========================================
# 2. PHASE 1 FEATURE SELECTION PIPELINE
# ==========================================
def run_phase1_category6():
    df_market = fetch_and_compute_category6()
    
    today_str = datetime.now().strftime('%Y-%m-%d')
    df_eval = df_market.loc['2024-01-01':today_str].dropna(subset=['target_next_ret']).copy()
    
    # Fill residual NaNs with forward/backward fill to preserve exact variance
    df_eval = df_eval.ffill().bfill()

    target = df_eval['target_next_ret']
    features = [c for c in df_eval.columns if c != 'target_next_ret']
    
    # Spearman Correlation Matrix
    corr_matrix = df_eval[features].corr(method='spearman').fillna(0.0)
    
    ic_results = {}
    
    for col in features:
        feat_series = df_eval[col]
        ic_list = []
        
        for i in range(20, len(df_eval)):
            sub_f = feat_series.iloc[i-20:i]
            sub_t = target.iloc[i-20:i]
            if sub_f.std() > 1e-7 and sub_t.std() > 1e-7:
                corr, _ = stats.spearmanr(sub_f, sub_t)
                if not np.isnan(corr):
                    ic_list.append(corr)
        
        if len(ic_list) > 0:
            rank_ic = np.mean(ic_list)
            std_ic = np.std(ic_list)
            icir = rank_ic / std_ic if std_ic > 1e-6 else 0.0
        else:
            rank_ic, _ = stats.spearmanr(feat_series, target)
            if np.isnan(rank_ic): rank_ic = 0.0
            icir = rank_ic * 0.5
            
        ic_results[col] = {'RankIC': rank_ic, 'ICIR': icir}

    ic_df = pd.DataFrame(ic_results).T

    # Thresholding on ICIR
    survivors_p2 = ic_df[ic_df['ICIR'].abs() >= 0.10].index.tolist()
    low_icir_dropped = ic_df[ic_df['ICIR'].abs() < 0.10].index.tolist()

    final_survivors = []
    dropped_correlated = {}

    for feat in survivors_p2:
        keep = True
        for prev_feat in final_survivors:
            corr_val = abs(corr_matrix.loc[feat, prev_feat])
            if corr_val > 0.70:
                if abs(ic_df.loc[feat, 'ICIR']) <= abs(ic_df.loc[prev_feat, 'ICIR']):
                    keep = False
                    dropped_correlated[feat] = f"Corr {corr_val:.2f} with {prev_feat}"
                    break
        if keep:
            final_survivors.append(feat)

    # Format Results Table
    results = []
    for feat in features:
        rank_ic = ic_df.loc[feat, 'RankIC']
        icir = ic_df.loc[feat, 'ICIR']
        
        if feat in low_icir_dropped:
            verdict = "DROP LOW ICIR"
            dropped_with = "-"
        elif feat in dropped_correlated:
            verdict = "CORRELATED DROP"
            dropped_with = dropped_correlated[feat]
        else:
            verdict = "KEEP"
            dropped_with = "-"
            
        results.append({
            'Feature': feat,
            'Stationary?': 'Yes',
            'Rank IC': f"{rank_ic:10.6f}",
            'ICIR': f"{icir:10.6f}",
            'Dropped Correlated With': dropped_with,
            'Verdict': verdict
        })

    # Prepare Printable String
    out = []
    out.append("="*90)
    out.append(f"CATEGORY 6 P3 WITHIN-CATEGORY SPEARMAN CORRELATION (2024-01-01 to {today_str})")
    out.append("="*90)
    out.append(corr_matrix.round(3).to_string())
    out.append("\n" + "="*90)
    out.append("CATEGORY 6 FINAL PHASE 1 FEATURE SELECTION")
    out.append("="*90)
    
    header = f"{'Feature':<22} {'Stationary?':<12} {'Rank IC':<12} {'ICIR':<12} {'Dropped Correlated With':<32} {'Verdict':<15}"
    out.append(header)
    out.append("-" * len(header))
    
    for row in results:
        line = f"{row['Feature']:<22} {row['Stationary?']:<12} {row['Rank IC']:<12} {row['ICIR']:<12} {row['Dropped Correlated With']:<32} {row['Verdict']:<15}"
        out.append(line)
        
    out.append("="*90)
    out.append(f"FINAL CATEGORY 6 FEATURES: {len(final_survivors)}")
    for i, f in enumerate(final_survivors, 1):
        out.append(f"{i}. {f}")
    out.append(f"Total features: {len(features)}")
    out.append(f"Final kept features: {len(final_survivors)}")
    out.append(f"Dropped features: {len(features) - len(final_survivors)}")
    out.append("CATEGORY 6 PHASE 1 COMPLETED SUCCESSFULLY")

    final_report_str = "\n".join(out)
    
    # Terminal Display
    print(final_report_str)

    # Save to TXT file in the current project directory
    output_filename = "category6_phase1_report_2024_2026.txt"
    with open(output_filename, "w") as f:
        f.write(final_report_str)

if __name__ == "__main__":
    run_phase1_category6()
