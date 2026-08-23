import os
import pandas as pd
from src.processing import calculate_noise_reduction
from src.features import generate_systemic_features

def run_pipeline():
    print("=== STARTING QUANT PIPELINE EXECUTION ===\n")
    
    # Path to the transferred offline parquet file
    parquet_path = "/home/soq/__shutupandbendover/devansh-swc-wala/market_data.parquet"
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}. Please transfer it via SCP.")
        
    print(f"Loading offline market data from: {parquet_path}")
    df_raw = pd.read_parquet(parquet_path)
    
    # Step 1: Run Processing / Noise Reduction
    df_smoothed = calculate_noise_reduction(df_raw, method="EMA", window=20)
    
    # Step 2: Run Feature Engineering
    df_features = generate_systemic_features(df_smoothed, rolling_window=12)
    
    print("\n=== PIPELINE SUCCESS: Ready for sandbox environments ===")
    print(df_features.head())
    print(df_smoothed)

if __name__ == "__main__":
    run_pipeline()