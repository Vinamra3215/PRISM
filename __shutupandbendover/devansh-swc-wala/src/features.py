import os
import pandas as pd
import numpy as np

def generate_systemic_features(df_smoothed, rolling_window=12, rsi_window=14, bb_window=20):
    """
    Calculates statistical market relationship indicators, volatility bands, 
    and momentum structures over the pre-smoothed pricing layers.
    """
    df_features = df_smoothed.copy()
    
    # ==========================================
    # 1. PRICE RETURN & SENSITIVITY MATH (Beta, Alpha, Correlation)
    # ==========================================
    df_features['Stock_Return'] = df_features['Stock_Smoothed'].pct_change()
    df_features['Nifty_Return'] = df_features['Nifty_Smoothed'].pct_change()
    
    covariance = df_features['Stock_Return'].rolling(window=rolling_window).cov(df_features['Nifty_Return'])
    variance = df_features['Nifty_Return'].rolling(window=rolling_window).var()
    
    # Beta and Correlation
    df_features['Beta'] = covariance / variance
    df_features['Correlation'] = df_features['Stock_Return'].rolling(window=rolling_window).corr(df_features['Nifty_Return'])
    
    # Jensen's Alpha (Simplified for zero nominal short-term risk-free rate)
    df_features['Alpha'] = df_features['Stock_Return'] - (df_features['Beta'] * df_features['Nifty_Return'])
    
    # ==========================================
    # 2. VOLATILITY ENGINE (Standard Deviation & Bollinger Bands)
    # ==========================================
    # Rolling standard deviation of the smoothed price
    df_features['Rolling_Std'] = df_features['Stock_Smoothed'].rolling(window=bb_window).std()
    
    # Bollinger Bands construction
    df_features['BB_Middle'] = df_features['Stock_Smoothed'].rolling(window=bb_window).mean()
    df_features['BB_Upper'] = df_features['BB_Middle'] + (2 * df_features['Rolling_Std'])
    df_features['BB_Lower'] = df_features['BB_Middle'] - (2 * df_features['Rolling_Std'])
    
    # ==========================================
    # 3. MOMENTUM ENGINE (Relative Strength Index - RSI)
    # ==========================================
    delta = df_features['Stock_Smoothed'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_window).mean()
    
    # Prevent division-by-zero errors during low-volatility tracking patches
    rs = gain / loss.replace(0, np.nan)
    df_features['RSI'] = 100 - (100 / (1 + rs))
    df_features['RSI'] = df_features['RSI'].fillna(50)  # Neutral default for flat asset regimes
    
    # ==========================================
    # 4. EXPORT COMPRESSION ENGINE (.parquet)
    # ==========================================
    # ==========================================
    # 4. EXPORT COMPRESSION ENGINE (.parquet)
    # ==========================================
    df_features.dropna(inplace=True)
    df_features = df_features.sort_index()
    df_features['Day_Number'] = np.arange(1, len(df_features) + 1)
    
    # Save directly to root directory to avoid folder creation issues
    output_path = "/home/soq/__shutupandbendover/devansh-swc-wala/smoothed_features.parquet"
    df_features.to_parquet(output_path, engine="pyarrow")
    
    print(f"Feature calculation pipeline completed. Checkpoint written to root: {os.path.abspath(output_path)}")
    return df_features