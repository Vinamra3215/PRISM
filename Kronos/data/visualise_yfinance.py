import pandas as pd
import plotly.graph_objects as go

# 1. Define File Paths
# Pointing to your NIFTY50 daily data and forecast
historical_data_path = "./Kronos/data/NIFTY50_5Y_OHLCV.parquet"
forecast_data_path = "./Kronos/_results/NIFTY50_forecast.parquet"

print("Loading data files...")

# 2. Load and Prepare the Historical Data
hist_df = pd.read_parquet(historical_data_path)

# Flatten yfinance MultiIndex columns if present
if isinstance(hist_df.columns, pd.MultiIndex):
    hist_df.columns = hist_df.columns.get_level_values(0)

# Extract the date from the index if it is hidden there
index_name = str(hist_df.index.name).lower()
if 'date' in index_name or 'time' in index_name:
    hist_df = hist_df.reset_index()

# Dynamically map the columns to standard lowercase names
rename_map = {}
for col in hist_df.columns:
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

# Apply the renaming and convert to datetime
hist_df = hist_df.rename(columns=rename_map)
hist_df['timestamps'] = pd.to_datetime(hist_df['timestamps'])

# ISOLATE CONTEXT WINDOW: Keep ONLY the exact 400 daily candlesticks used for the prediction
lookback = 400
hist_df = hist_df.tail(lookback).reset_index(drop=True)

# 3. Load and Prepare the Forecast Data
pred_df = pd.read_parquet(forecast_data_path)

# The Kronos model saves the future timestamps as the DataFrame's index
pred_df.index = pd.to_datetime(pred_df.index)

# 4. Initialize the Plotly Figure
print("Building the interactive chart...")
fig = go.Figure()

# 5. Add Historical Data Trace (The 400 Input Bars)
fig.add_trace(go.Candlestick(
    x=hist_df['timestamps'],
    open=hist_df['open'],
    high=hist_df['high'],
    low=hist_df['low'],
    close=hist_df['close'],
    name='Historical Context (400 Days)',
    increasing_line_color='#17BECF', # Cyan
    decreasing_line_color='#7F7F7F'  # Gray
))

# 6. Add Forecasted Data Trace (The 120 Predicted Bars)
fig.add_trace(go.Candlestick(
    x=pred_df.index,
    open=pred_df['open'],
    high=pred_df['high'],
    low=pred_df['low'],
    close=pred_df['close'],
    name='Kronos Forecast (120 Days)',
    increasing_line_color='#FF8C00', # Orange
    decreasing_line_color='#FF0000'  # Red
))

# 7. Configure Layout
fig.update_layout(
    title='NIFTY50: Model Input Context vs Kronos Forecast',
    yaxis_title='Index Price',
    xaxis_title='Date',
    xaxis_rangeslider_visible=True, # Interactive timeline slider at the bottom
    template='plotly_dark',         # Dark mode background
    xaxis=dict(
        type='date',
        rangebreaks=[dict(bounds=["sat", "mon"])] # Hides weekends for a seamless chart
    )
)

# 8. Export to HTML
chart_path = "./Kronos/_results/interactive_forecast.html"
fig.write_html(chart_path)
print(f"Interactive chart successfully saved to {chart_path}")