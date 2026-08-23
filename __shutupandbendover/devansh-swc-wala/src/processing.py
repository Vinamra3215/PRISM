import pandas as pd

def calculate_noise_reduction(df, method="EMA", window=20):
    """
    Applies dedicated tracking filters to strip high-frequency macro variance anomalies.
    
    Parameters:
    - df: Input pandas DataFrame containing 'Stock_Raw' and 'Nifty_Raw'.
    - method: 'SMA', 'EMA', or 'Median'.
    - window: Integer sizing window for the rolling calculations.
    """
    df_smoothed = df.copy()
    
    if method == "SMA":
        df_smoothed['Stock_Smoothed'] = df['Stock_Raw'].rolling(window=window, min_periods=1).mean()
        df_smoothed['Nifty_Smoothed'] = df['Nifty_Raw'].rolling(window=window, min_periods=1).mean()
    elif method == "EMA":
        df_smoothed['Stock_Smoothed'] = df['Stock_Raw'].ewm(span=window, adjust=False).mean()
        df_smoothed['Nifty_Smoothed'] = df['Nifty_Raw'].ewm(span=window, adjust=False).mean()
    elif method == "Median":
        df_smoothed['Stock_Smoothed'] = df['Stock_Raw'].rolling(window=window, min_periods=1).median()
        df_smoothed['Nifty_Smoothed'] = df['Nifty_Raw'].rolling(window=window, min_periods=1).median()
    else:
        raise ValueError("Invalid strategy token profile. Choose between SMA, EMA, or Median.")
        
    return df_smoothed