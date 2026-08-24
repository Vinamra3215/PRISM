import sys
import pandas as pd

# 1. Point Python to the actual source code directory
#sys.path.append("./Kronos/Kronos-src")

from model import Kronos, KronosTokenizer, KronosPredictor

# 2. Load the Tokenizer and Model using your local 'weights' directory
tokenizer_path = "./Kronos/weights/Kronos-Tokenizer-base"
model_path = "./Kronos/weights/Kronos-base"

print("Loading local models...")
tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)
model = Kronos.from_pretrained(model_path)

# 3. Instantiate the Predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)

# 4. Prepare Input Data
data_file_path = "./Kronos/data/NIFTY50_5Y_OHLCV.parquet" 

print("Loading Parquet data...")
df = pd.read_parquet(data_file_path)

# Flatten yfinance MultiIndex columns if present
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# Extract the date from the index if it is hidden there
index_name = str(df.index.name).lower()
if 'date' in index_name or 'time' in index_name:
    df = df.reset_index()

# Dynamically map the columns to Kronos standard lowercase names
rename_map = {}
for col in df.columns:
    col_str = str(col).lower()
    if 'date' in col_str or 'timestamp' in col_str or 'time' in col_str:
        rename_map[col] = 'timestamps'
    elif 'open' in col_str:
        rename_map[col] = 'open'
    elif 'high' in col_str:
        rename_map[col] = 'high'
    elif 'low' in col_str:
        rename_map[col] = 'low'
    elif 'close' in col_str:
        rename_map[col] = 'close'
    elif 'volume' in col_str:
        rename_map[col] = 'volume'

# Apply the renaming and convert to standard datetime
df = df.rename(columns=rename_map)
df['timestamps'] = pd.to_datetime(df['timestamps'])

# Define context window and prediction length (in days)
lookback = 400
pred_len = 120

print("Preparing recent data context...")
# Grab the most recent 'lookback' rows for the context
recent_data = df.tail(lookback).reset_index(drop=True)

# Prepare inputs for the predictor
x_df = recent_data[['open', 'high', 'low', 'close', 'volume']]
x_timestamp = recent_data['timestamps']

# Generate future daily timestamps (skipping weekends using 'B' for Business Day)
# Wrapped in pd.Series to ensure compatibility with .dt accessor inside Kronos source code
last_time = x_timestamp.iloc[-1]
y_timestamp = pd.Series(pd.date_range(
    start=last_time + pd.Timedelta(days=1), 
    periods=pred_len, 
    freq='B'
))

# 5. Generate Forecasts
print("Generating forecast...")
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,           
    top_p=0.9,       
    sample_count=1   
)

print("Forecasted Data Head:")
print(pred_df.head())

# 6. Save the Forecast
output_path = "./Kronos/_results/NIFTY50_forecast.parquet"
pred_df.to_parquet(output_path)
print(f"Forecast successfully saved to {output_path}")