import sys
import pandas as pd

from model import Kronos, KronosTokenizer, KronosPredictor

# 2. Load the Tokenizer and Model using your renamed local 'weights' directory
tokenizer_path = "/home/soq/__shutupandbendover/het-uchiha/weights/Kronos-Tokenizer-base"
model_path = "/home/soq/__shutupandbendover/het-uchiha/weights/Kronos-base"

print("Loading local models...")
tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
model = Kronos.from_pretrained(model_path)

# 3. Instantiate the Predictor
# The max_context for Kronos-base is 512
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 4. Prepare Input Data
data_file_path = "./Kronos/data/NIFTY50_5Y_OHLCV.parquet" 

print("Loading Parquet data...")
df = pd.read_parquet(data_file_path)

# Rename 'timestamp' to 'timestamps' to match Kronos's expected input
df = df.rename(columns={"timestamp": "timestamps"})

# Ensure timestamps are standard pandas datetime objects
df['timestamps'] = pd.to_datetime(df['timestamps'])

# Define context window and prediction length
lookback = 400
pred_len = 120

print("Preparing recent data context...")
# Grab the most recent 'lookback' rows for the context
recent_data = df.tail(lookback).reset_index(drop=True)

# Prepare inputs for the predictor
x_df = recent_data[['open', 'high', 'low', 'close', 'volume']]
x_timestamp = recent_data['timestamps']

# Generate future timestamps for the prediction
# Wrap the date_range in pd.Series() to fix the AttributeError
last_time = x_timestamp.iloc[-1]
y_timestamp = pd.Series(pd.date_range(
    start=last_time + pd.Timedelta(days=1), # Add 1 day instead of 1 minute
    periods=pred_len, 
    freq='B'                                # 'B' stands for Business Day (skips weekends)
))

# 5. Generate Forecasts
print("Generating forecast...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,           # Temperature for sampling
    top_p=0.9,       # Nucleus sampling probability
    sample_count=1   # Number of forecast paths to generate and average
)

print("Forecasted Data Head:")
print(pred_df.head())

# 6. Save the Forecast
output_path = "./Kronos/_results/forecast.parquet"
pred_df.to_parquet(output_path)
print(f"Forecast successfully saved to {output_path}")