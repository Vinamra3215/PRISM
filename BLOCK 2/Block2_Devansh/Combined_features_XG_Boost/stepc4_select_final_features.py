import pandas as pd
import numpy as np

def select_final_features():
    print("--- STEP C4: Automatic Balanced Feature Selection (15-20 Target) ---")
    
    # 1. Load the combined matrix from Step C3
    matrix_path = '/home/soq/__shutupandbendover/devansh-swc-wala/Block2/Combined_features_XG_Boost/final_combined_feature_matrix.csv'
    df = pd.read_csv(matrix_path)
    
    # Map each feature to its respective category
    category_mapping = {
        'return_1d': 'Price & Returns', 'return_5d': 'Price & Returns', 'ret_10d': 'Price & Returns', 'ret_20d': 'Price & Returns',
        'intraday_range': 'Price & Returns', 'open_close_spread': 'Price & Returns', '52w_high_dist': 'Price & Returns',
        
        'rolling_volatility_10d': 'Volatility', 'volatility_ratio_5d_20d': 'Volatility', 'volatility_ratio_20d_60d': 'Volatility',
        'rolling_skewness_20d': 'Volatility', 'rolling_kurtosis_20d': 'Volatility', 'atr_14d': 'Volatility',
        
        'ma_ratio_50_200': 'Momentum & Trend', 'macd_signal': 'Momentum & Trend', 'macd_hist': 'Momentum & Trend',
        'rsi_14': 'Momentum & Trend',
        
        'volume_ratio_5d': 'Volume', 'volume_ratio_20d': 'Volume', 'vwap_deviation': 'Volume',
        'volume_momentum': 'Volume', 'mfi': 'Volume', 'volume_price_trend': 'Volume',
        
        'upper_shadow_ratio': 'Candlestick', 'candle_direction': 'Candlestick', 'consecutive_down_days': 'Candlestick',
        'keltner_channel_position': 'Candlestick',
        
        'nifty_daily_return': 'Market-Level', 'nifty_5d_return': 'Market-Level', 'nifty_vol_20d': 'Market-Level',
        'vix_level': 'Market-Level', 'bank_return': 'Market-Level', 'pharma_return': 'Market-Level', 'auto_return': 'Market-Level',
        
        'cs_return_z': 'Cross-Sectional', 'sector_relative_return': 'Cross-Sectional', 'market_relative_return': 'Cross-Sectional',
        'relative_strength_vs_nifty': 'Cross-Sectional', 'return_rank_pct_20d': 'Cross-Sectional',
        
        'breakout_signal': 'Derived', 'mean_reversion': 'Derived', 'rsi_vol_diverg': 'Derived',
        'trend_strength': 'Derived', 'vol_weight_mom': 'Derived'
    }
    
    # Clean feature names to match dictionary keys
    df['clean_feature'] = df['feature'].str.lower().str.strip()
    df['category'] = df['clean_feature'].map(category_mapping).fillna('Derived')
    
    # 2. Compute a balanced Composite Score (lower is better rank)
    # Give equal weight (50% XGBoost importance rank, 50% ICIR rank)
    df['composite_score'] = (0.5 * df['xgb_rank']) + (0.5 * df['icir_rank'])
    
    # 3. Apply category target quotas to ensure a well-distributed portfolio
    target_quotas = {
        'Price & Returns': 3,
        'Volatility': 3,
        'Momentum & Trend': 3,
        'Volume': 2,
        'Candlestick': 2,
        'Market-Level': 2,
        'Cross-Sectional': 2,
        'Derived': 2
    }  # Total target = 19 features
    
    selected_rows = []
    for cat, quota in target_quotas.items():
        cat_df = df[df['category'] == cat].sort_values('composite_score')
        top_cat = cat_df.head(quota)
        selected_rows.append(top_cat)
        
    final_selected = pd.concat(selected_rows).reset_index(drop=True)
    
    # Assign final sequential numbering (#1 to #19)
    final_selected['#'] = final_selected.index + 1
    
    # Format final deliverable columns
    deliverable = pd.DataFrame({
        '#': final_selected['#'],
        'Feature': final_selected['feature'],
        'Category': final_selected['category'],
        'RankIC': '---',  # Placeholder for explicit user table formatting
        'ICIR': final_selected['abs_icir'].round(4),
        'XGBoost Rank': final_selected['xgb_rank'],
        'Why Selected': final_selected.apply(
            lambda r: f"Balanced composite rank ({r['composite_score']:.1f}), strong multi-category signal", axis=1
        )
    })
    
    print("\n=======================================================")
    print("FINAL DELIVERABLE: 19-FEATURE ROBUST SUBSET")
    print("=======================================================")
    print(deliverable.to_string(index=False))
    
    output_path = '/home/soq/__shutupandbendover/devansh-swc-wala/Block2/Combined_features_XG_Boost/final_selected_19_features.csv'
    deliverable.to_csv(output_path, index=False)
    print(f"\nFinal feature set successfully saved to '{output_path}'")

if __name__ == '__main__':
    select_final_features()