import pandas as pd
import numpy as np

def run_step_c3(xgb_rankings_file):
    print("--- STEP C3: Final Feature Set Selection View ---")
    xgb_df = pd.read_csv(xgb_rankings_file)
    
    # Normalize column names to lowercase to prevent capitalization mismatches
    xgb_df.columns = xgb_df.columns.str.lower()
    
    # Check if 'feature' column exists
    if 'feature' not in xgb_df.columns:
        raise KeyError(f"Expected 'feature' column in CSV. Found columns: {xgb_df.columns.tolist()}")

    # ICIR dictionary with absolute values (for ranking predictive power regardless of direction)
    icir_dict = {
        'nifty_daily_return': abs(0.4272), 'pharma_return': abs(0.2659), 'nifty_5d_return': abs(-0.4056), 
        'nifty_vol_20d': abs(0.1943), 'vix_level': abs(0.1802), 'bank_return': abs(0.3777), 
        'ma_ratio_50_200': abs(0.0033), '52w_high_dist': abs(0.1100), 'ret_10d': abs(-0.2100), 
        'rolling_volatility_10d': abs(-0.0486), 'volume_price_trend': abs(0.0406), 'intraday_range': abs(-0.0900), 
        'cs_return_z': abs(-0.1726), 'ret_20d': abs(-0.0900), 'ret_5d': abs(-0.1900), 'keltner_channel_position': abs(-0.2045), 
        'vol_weight_mom': abs(0.0988), 'volatility_ratio_20d_60d': abs(-0.0712), 'upper_shadow_ratio': abs(0.1233), 
        'mfi': abs(-0.1247), 'macd_hist': abs(-0.1335), 'volatility_ratio_5d_20d': abs(0.0182), 'atr_14d': abs(-0.0422), 
        'vwap_deviation': abs(-0.1810), 'rolling_kurtosis_20d': abs(-0.0494), 'open_close_spread': abs(-0.1800), 
        'volume_momentum': abs(-0.0556), 'sector_relative_return': abs(-0.0767), 'trend_strength': abs(0.0943), 
        'mean_reversion': abs(0.1635), 'rsi_14': abs(-0.1435), 'rolling_skewness_20d': abs(-0.0422), 
        'volume_ratio_5d': abs(-0.0120), 'relative_strength_vs_nifty': abs(-0.0643), 'market_relative_return': abs(-0.0643), 
        'volume_ratio_20d': abs(-0.0865), 'macd_signal': abs(-0.0390), 'rsi_vol_diverg': abs(0.1108), 
        'return_rank_pct_20d': abs(-0.0643), 'consecutive_down_days': abs(0.1833), 'candle_direction': abs(-0.1727), 
        'breakout_signal': abs(0.1005), 'auto_return': 0.0
    }
    
    # Map absolute ICIR values onto the dataframe
    xgb_df['abs_icir'] = xgb_df['feature'].map(icir_dict).fillna(0.0)
    
    # Rank by ICIR natively (highest absolute ICIR = Rank 1)
    icir_rank_df = xgb_df[['feature', 'abs_icir']].sort_values('abs_icir', ascending=False).reset_index(drop=True)
    icir_rank_df['icir_rank'] = icir_rank_df.index + 1
    
    # Merge back
    final_df = pd.merge(xgb_df, icir_rank_df[['feature', 'icir_rank']], on='feature')
    
    # Reorder columns for readability (supporting both case variants)
    gain_col = 'xgb_gain' if 'xgb_gain' in final_df.columns else 'XGB_Gain'
    rank_col = 'xgb_rank' if 'xgb_rank' in final_df.columns else 'XGB_Rank'
    
    final_df = final_df[['feature', 'abs_icir', 'icir_rank', rank_col, gain_col]]
    final_df = final_df.sort_values(rank_col)
    
    print("\n=======================================================")
    print("FINAL CANDIDATES FOR MANUAL SELECTION:")
    print("=======================================================")
    print(final_df.to_string(index=False))
    
    output_csv = '/home/soq/__shutupandbendover/devansh-swc-wala/Block2/Combined_features_XG_Boost/final_combined_feature_matrix.csv'
    final_df.to_csv(output_csv, index=False)
    print(f"\nSaved output matrix to '{output_csv}'")

if __name__ == '__main__':
    run_step_c3('/home/soq/__shutupandbendover/devansh-swc-wala/Block2/Combined_features_XG_Boost/xgboost_rankings_c2.csv')