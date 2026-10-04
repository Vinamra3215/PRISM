"""
NIFTY 50 Quant Feature Pipeline
--------------------------------
Generates a consolidated table: Rows = (date x stock), Columns = ~43 Features + Target
Target: next_day_return

Input Source: 'nifty50_raw_data.parquet'
Expected Parquet Columns: 
- Stock Data: ['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'sector']
- Market Data (if in separate file or merged): ['nifty_close', 'vix_close', 'bank_close', 'pharma_close', 'auto_close']
  (If market data is in a separate parquet/csv, pass df_market explicitly, or load it inside the script)
"""

import pandas as pd
import numpy as np

def compute_technical_features(df_stocks, df_market=None):
    # Ensure dates are datetime and sorted
    df = df_stocks.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    if df_market is not None:
        df_market = df_market.copy()
        df_market['date'] = pd.to_datetime(df_market['date'])
        df_market = df_market.sort_values('date').reset_index(drop=True)

        # Calculate Market-Level Returns & Volatility (Category 6)
        df_market['nifty_daily_return'] = df_market['nifty_close'].pct_change()
        df_market['nifty_5d_return'] = df_market['nifty_close'].pct_change(5)
        df_market['nifty_vol_20d'] = df_market['nifty_daily_return'].rolling(20).std() * np.sqrt(252)
        df_market['vix_level'] = df_market['vix_close']
        df_market['bank_return'] = df_market['bank_close'].pct_change()
        df_market['pharma_return'] = df_market['pharma_close'].pct_change()
        df_market['auto_return'] = df_market['auto_close'].pct_change()

        # Merge Market Data onto Stocks Data by date
        df = pd.merge(df, df_market[['date', 'nifty_close', 'nifty_daily_return', 'nifty_5d_return', 
                                     'nifty_vol_20d', 'vix_level', 'bank_return', 
                                     'pharma_return', 'auto_return']], on='date', how='left')

    # Group by ticker for stock-specific time series features
    grouped = df.groupby('ticker')

    # --- CATEGORY 1: Price & Returns ---
    df['return_1d'] = grouped['close'].pct_change(1)
    df['Return 5-day'] = grouped['close'].pct_change(5)
    df['Return 10-day'] = grouped['close'].pct_change(10)
    df['Return 20-day'] = grouped['close'].pct_change(20)
    df['Intraday Range'] = (df['high'] - df['low']) / df['open']
    df['Open-Close Spread'] = (df['close'] - df['open']) / df['open']
    
    # 52-Week High Distance (~252 trading days)
    rolling_52w_high = grouped['high'].transform(lambda x: x.rolling(252, min_periods=50).max())
    df['52-Week High Distance'] = (df['close'] - rolling_52w_high) / rolling_52w_high

    # --- CATEGORY 2: Volatility Features ---
    # Rolling Volatilities (Annualized)
    df['rolling_volatility_10d'] = grouped['return_1d'].transform(lambda x: x.rolling(10).std() * np.sqrt(252))
    vol_20d = grouped['return_1d'].transform(lambda x: x.rolling(20).std() * np.sqrt(252))
    vol_5d = grouped['return_1d'].transform(lambda x: x.rolling(5).std() * np.sqrt(252))
    vol_60d = grouped['return_1d'].transform(lambda x: x.rolling(60).std() * np.sqrt(252))

    # Volatility Ratios
    df['volatility_ratio_5d_20d'] = vol_5d / (vol_20d + 1e-8)
    df['volatility_ratio_20d_60d'] = vol_20d / (vol_60d + 1e-8)

    # Higher Moments: Rolling Skewness and Kurtosis
    df['rolling_skewness_20d'] = grouped['return_1d'].transform(lambda x: x.rolling(20).skew())
    df['rolling_kurtosis_20d'] = grouped['return_1d'].transform(lambda x: x.rolling(20).kurt())

    # Average True Range (ATR 14d) - Relative ATR normalized by Close
    prev_close = grouped['close'].shift(1)
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - prev_close).abs()
    tr3 = (df['low'] - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_14d'] = true_range.groupby(df['ticker']).transform(lambda x: x.rolling(14).mean()) / df['close']

    # --- CATEGORY 3: Momentum & Trend ---
    sma50 = grouped['close'].transform(lambda x: x.rolling(50).mean())
    sma200 = grouped['close'].transform(lambda x: x.rolling(200).mean())
    df['ma_ratio_50_200'] = sma50 / (sma200 + 1e-8)

    # MACD (12, 26, 9)
    ema12 = grouped['close'].transform(lambda x: x.ewm(span=12, adjust=False).mean())
    ema26 = grouped['close'].transform(lambda x: x.ewm(span=26, adjust=False).mean())
    macd = ema12 - ema26
    macd_signal = macd.groupby(df['ticker']).transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df['macd_signal'] = macd_signal
    df['macd_hist'] = macd - macd_signal

    # RSI (14)
    delta = grouped['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(df['ticker']).transform(lambda x: x.ewm(com=13, adjust=False).mean())
    avg_loss = loss.groupby(df['ticker']).transform(lambda x: x.ewm(com=13, adjust=False).mean())
    rs = avg_gain / (avg_loss + 1e-8)
    df['rsi_14'] = 100 - (100 / (1 + rs))

    # --- CATEGORY 4: Volume ---
    vol_sma20 = grouped['volume'].transform(lambda x: x.rolling(20).mean())
    vol_sma5 = grouped['volume'].transform(lambda x: x.rolling(5).mean())
    df['Volume Ratio 20-day'] = df['volume'] / (vol_sma20 + 1e-8)
    df['Volume Ratio 5-day'] = df['volume'] / (vol_sma5 + 1e-8)

    # VWAP Deviation (Rolling 20-day VWAP)
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    pv = typical_price * df['volume']
    rolling_pv = pv.groupby(df['ticker']).transform(lambda x: x.rolling(20).sum())
    rolling_v = df['volume'].groupby(df['ticker']).transform(lambda x: x.rolling(20).sum())
    vwap = rolling_pv / (rolling_v + 1e-8)
    df['VWAP Deviation'] = (df['close'] - vwap) / (vwap + 1e-8)

    # Volume Momentum
    df['Volume Momentum'] = grouped['volume'].pct_change(5)

    # Money Flow Index (MFI 14)
    raw_money_flow = typical_price * df['volume']
    price_diff = typical_price.groupby(df['ticker']).diff()
    pos_flow = pd.Series(np.where(price_diff > 0, raw_money_flow, 0), index=df.index)
    neg_flow = pd.Series(np.where(price_diff < 0, raw_money_flow, 0), index=df.index)
    pos_mf = pos_flow.groupby(df['ticker']).transform(lambda x: x.rolling(14).sum())
    neg_mf = neg_flow.groupby(df['ticker']).transform(lambda x: x.rolling(14).sum())
    mfi_ratio = pos_mf / (neg_mf + 1e-8)
    df['MFI'] = 100 - (100 / (1 + mfi_ratio))

    # Volume-Price Trend (VPT)
    vpt_change = df['volume'] * df['return_1d']
    df['Volume-Price Trend (VPT)'] = vpt_change.groupby(df['ticker']).cumsum()

    # --- CATEGORY 5: Candlestick ---
    body_max = np.maximum(df['open'], df['close'])
    total_range = df['high'] - df['low'] + 1e-8
    df['upper_shadow_ratio'] = (df['high'] - body_max) / total_range
    df['candle_direction'] = np.sign(df['close'] - df['open'])

    # Consecutive down days
    is_down = (df['close'] < df['close'].shift(1)).astype(int)
    down_groups = (is_down == 0).groupby(df['ticker']).cumsum()
    df['consecutive_down_days'] = is_down.groupby([df['ticker'], down_groups]).cumsum()

    # Keltner Channel Position
    atr10 = (df['high'] - df['low']).groupby(df['ticker']).transform(lambda x: x.rolling(10).mean())
    ema20 = grouped['close'].transform(lambda x: x.ewm(span=20, adjust=False).mean())
    upper_keltner = ema20 + (2 * atr10)
    lower_keltner = ema20 - (2 * atr10)
    df['keltner_channel_position'] = (df['close'] - lower_keltner) / (upper_keltner - lower_keltner + 1e-8)

    # --- CATEGORY 7: Cross-Sectional Features ---
    date_group = df.groupby('date')
    mean_ret_1d = date_group['return_1d'].transform('mean')
    std_ret_1d = date_group['return_1d'].transform('std')
    df['Cross Sectional Return Z score'] = (df['return_1d'] - mean_ret_1d) / (std_ret_1d + 1e-8)

    if 'sector' in df.columns:
        sector_mean_ret = df.groupby(['date', 'sector'])['return_1d'].transform('mean')
        df['Sector Relative Return'] = df['return_1d'] - sector_mean_ret

    if 'nifty_daily_return' in df.columns:
        df['Market Relative Return'] = df['return_1d'] - df['nifty_daily_return']

    df['Return Rank Percentile (20-day)'] = date_group['Return 20-day'].rank(pct=True)

    # --- CATEGORY 8: Derived Features ---
    rolling_20_high = grouped['high'].transform(lambda x: x.shift(1).rolling(20).max())
    df['breakout_signal'] = (df['close'] > rolling_20_high).astype(int)

    # Mean Reversion (Bollinger Z-Score)
    sma20 = grouped['close'].transform(lambda x: x.rolling(20).mean())
    std20 = grouped['close'].transform(lambda x: x.rolling(20).std())
    df['mean_reversion'] = (df['close'] - sma20) / (std20 + 1e-8)

    # RSI Vol Divergence
    df['rsi_vol_diverg'] = df['rsi_14'] * df['Volume Ratio 20-day']

    # Trend Strength
    df['trend_strength'] = (sma20 - sma50).abs() / (sma50 + 1e-8)

    # Volume Weighted Momentum
    df['vol_weight_mom'] = df['Return 20-day'] * df['Volume Ratio 20-day']

    # --- TARGET VARIABLE ---
    df['next_day_return'] = grouped['close'].shift(-1) / df['close'] - 1

    return df

if __name__ == '__main__':
    # 1. Update the input path to your specific directory
    input_filepath = '/home/soq/__shutupandbendover/devansh-swc-wala/Block2/feature_selection/nifty50_complete_raw_data.parquet'
    
    # 2. Set the output path to save in the same directory
    output_filepath = '/home/soq/__shutupandbendover/devansh-swc-wala/Block2/Combined_features_XG_Boost/nifty50_feature_table_2022_present.parquet'
    
    print(f"Loading {input_filepath}...")
    try:
        # Load the Parquet File directly from the absolute path
        df_raw = pd.read_parquet(input_filepath)
        print(f"Data loaded successfully. Shape: {df_raw.shape}")
        
        # Process technical features
        df_features = compute_technical_features(df_raw)
        
        # Save output table to the absolute path
        df_features.to_parquet(output_filepath, index=False)
        print(f"Feature table generated and saved to {output_filepath}!")
        
    except Exception as e:
        print(f"Error loading or processing file: {e}")