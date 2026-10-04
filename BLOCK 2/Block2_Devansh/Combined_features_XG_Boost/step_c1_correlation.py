import pandas as pd
import numpy as np

def run_step_c1(filepath):
    print("--- STEP C1: Cross-Category Correlation Check ---")
    df = pd.read_parquet(filepath)
    features = [col for col in df.columns if col not in ['date', 'ticker', 'sector', 'return_1d', 'next_day_return']]
    
    # ICIR dictionary extracted from the provided table
    # Note: auto_return ICIR was blank (NaN) in the table, setting it to 0.0 for comparison purposes
    icir_dict = {
        'nifty_daily_return': 0.4272, 'pharma_return': 0.2659, 'nifty_5d_return': -0.4056, 'nifty_vol_20d': 0.1943, 
        'vix_level': 0.1802, 'bank_return': 0.3777, 'ma_ratio_50_200': 0.0033, '52w_high_dist': 0.1100, 
        'ret_10d': -0.2100, 'rolling_volatility_10d': -0.0486, 'volume_price_trend': 0.0406, 'intraday_range': -0.0900, 
        'cs_return_z': -0.1726, 'ret_20d': -0.0900, 'ret_5d': -0.1900, 'keltner_channel_position': -0.2045, 
        'vol_weight_mom': 0.0988, 'volatility_ratio_20d_60d': -0.0712, 'upper_shadow_ratio': 0.1233, 'mfi': -0.1247, 
        'macd_hist': -0.1335, 'volatility_ratio_5d_20d': 0.0182, 'atr_14d': -0.0422, 'vwap_deviation': -0.1810, 
        'rolling_kurtosis_20d': -0.0494, 'open_close_spread': -0.1800, 'volume_momentum': -0.0556, 
        'sector_relative_return': -0.0767, 'trend_strength': 0.0943, 'mean_reversion': 0.1635, 'rsi_14': -0.1435, 
        'rolling_skewness_20d': -0.0422, 'volume_ratio_5d': -0.0120, 'relative_strength_vs_nifty': -0.0643, 
        'market_relative_return': -0.0643, 'volume_ratio_20d': -0.0865, 'macd_signal': -0.0390, 'rsi_vol_diverg': 0.1108, 
        'return_rank_pct_20d': -0.0643, 'consecutive_down_days': 0.1833, 'candle_direction': -0.1727, 'breakout_signal': 0.1005, 
        'auto_return': 0.0 # ICIR was blank in table
    }

    corr_matrix = df[features].dropna().corr(method='spearman').abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    features_to_drop = set()
    for feature1 in upper_tri.columns:
        for feature2 in upper_tri.index:
            if upper_tri.loc[feature2, feature1] > 0.7:
                icir1 = abs(icir_dict.get(feature1, 0.0))
                icir2 = abs(icir_dict.get(feature2, 0.0))
                
                if icir1 < icir2:
                    features_to_drop.add(feature1)
                else:
                    features_to_drop.add(feature2)
                    
    surviving_features = [f for f in features if f not in features_to_drop]
    print(f"Step C1 Complete: Reduced from {len(features)} to {len(surviving_features)} features.")
    
    # Save the surviving features to a text file to pass to Step 2
    with open('surviving_features_c1.txt', 'w') as f:
        for item in surviving_features:
            f.write("%s\n" % item)
            
    return surviving_features

if __name__ == '__main__':
    run_step_c1('/home/soq/__shutupandbendover/devansh-swc-wala/Block2/Combined_features_XG_Boost/nifty50_feature_table_2022_present.parquet')