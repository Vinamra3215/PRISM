import os
import pandas as pd
import yfinance as yf

def download_market_data(ticker="RELIANCE.NS", benchmark="^NSEI", start_date="2021-01-01", end_date="2026-07-01"):
    """
    Downloads raw daily pricing information from yfinance and persists it inside a raw Parquet directory.
    """
    print(f"Initializing data ingestion for Stock: {ticker} and Benchmark: {benchmark}...")
    
    # Download raw values
    raw_stock = yf.download(ticker, start=start_date, end=end_date)
    raw_nifty = yf.download(benchmark, start=start_date, end=end_date)
    
    # Standardize headers for modern multi-index yfinance output structures
    raw_stock.columns = raw_stock.columns.get_level_values(0)
    raw_nifty.columns = raw_nifty.columns.get_level_values(0)
    
    # Consolidate standard close vectors
    df_raw = pd.DataFrame({
        'Stock_Raw': raw_stock['Close'],
        'Nifty_Raw': raw_nifty['Close']
    }).dropna()
    
    # Establish structural path requirements
    output_dir = os.path.join("data", "raw")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "raw_market_data.parquet")
    df_raw.to_parquet(output_path, engine="pyarrow")
    
    print(f"Ingestion successful! Raw checkpoint written to: {output_path}")
    return df_raw