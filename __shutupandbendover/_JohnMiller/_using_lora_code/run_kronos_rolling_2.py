import sys
import pandas as pd
import torch
from tqdm import tqdm
from peft import PeftModel

sys.path.append("./__shutupandbendover/_JohnMiller/Kronos")
from model import Kronos, KronosTokenizer, KronosPredictor

tokenizer_path = "/home/soq/__shutupandbendover/_JohnMiller/Kronos/weights/Kronos-Tokenizer-base"
model_path = "/home/soq/__shutupandbendover/_JohnMiller/Kronos/weights/Kronos-base"
lora_path = "./__shutupandbendover/_JohnMiller/Kronos/weights/Infosys_LoRA"

print("Loading local models...")
tokenizer = KronosTokenizer.from_pretrained(tokenizer_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Target compute device: {device.upper()}")

base_model = Kronos.from_pretrained(model_path)
print(f"Loading LoRA adapter from {lora_path}...")
model = PeftModel.from_pretrained(base_model, lora_path)
model.to(device)
model.eval()

predictor = KronosPredictor(model, tokenizer, max_context=512)

data_file_path = "./__shutupandbendover/_JohnMiller/stock_data/infosys_5y_1d.parquet"
df = pd.read_parquet(data_file_path)
df['timestamps'] = pd.to_datetime(df['timestamps'])
df = df.sort_values('timestamps').reset_index(drop=True)

initial_start_date = '2022-01-01'
initial_end_date = '2023-12-31'
pred_len = 1          
step_size = 1         
sample_count = 20     
max_rolling_steps = 120 

initial_mask = (df['timestamps'] >= initial_start_date) & (df['timestamps'] <= initial_end_date)
initial_indices = df.index[initial_mask]

if len(initial_indices) == 0:
    raise ValueError("No data found matching the initial start/end date range.")

start_idx = initial_indices[0]
end_idx = initial_indices[-1]
window_size = end_idx - start_idx + 1

print(f"Context Window Size: {window_size} trading days")
print(f"Initial Context: {df.loc[start_idx, 'timestamps'].strftime('%Y-%m-%d')} -> {df.loc[end_idx, 'timestamps'].strftime('%Y-%m-%d')}")

total_available_steps = (len(df) - 1 - end_idx) // step_size
total_steps = min(max_rolling_steps, total_available_steps) if max_rolling_steps else total_available_steps

print(f"Executing {total_steps} LoRA rolling forecast iterations starting from 2024-01-01...")

all_forecasts = []

with torch.no_grad():
    for step in tqdm(range(total_steps), desc="LoRA Rolling Forecasts"):
        curr_start = start_idx + (step * step_size)
        curr_end = end_idx + (step * step_size)
        
        window_data = df.iloc[curr_start : curr_end + 1].reset_index(drop=True)
        
        x_df = window_data[['open', 'high', 'low', 'close', 'volume']]
        x_timestamp = window_data['timestamps']
        origin_date = x_timestamp.iloc[-1]
        
        y_timestamp = pd.Series(pd.date_range(
            start=origin_date + pd.Timedelta(days=1),
            periods=pred_len,
            freq='B'
        ))
        
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=pred_len,
            T=0.3,
            top_p=0.8,
            sample_count=sample_count
        )
        
        pred_df = pred_df.copy()
        pred_df['origin_date'] = origin_date
        pred_df['target_date'] = pred_df.index
        pred_df['horizon_step'] = [f"t+{i+1}" for i in range(pred_len)]
        pred_df['iteration'] = step
        
        all_forecasts.append(pred_df)

final_forecast_df = pd.concat(all_forecasts, ignore_index=True)

output_path = "./__shutupandbendover/_JohnMiller/predicted_data/infosys_rolling_lora_predictions.parquet"
final_forecast_df.to_parquet(output_path)
print(f"\nAll {total_steps} LoRA rolling predictions successfully saved to {output_path}")
print("\nPreview of Rolling Output:")
print(final_forecast_df[['origin_date', 'target_date', 'horizon_step', 'close']].head(6))