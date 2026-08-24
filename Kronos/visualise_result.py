import pandas as pd
import plotly.graph_objects as go

# 1. Define File Paths (Adjust if your folder structure is different)
historical_data_path = "./Kronos/data/RELIANCE-EQ_1min.parquet"
forecast_data_path = "./Kronos/_results/forecast.parquet"

print("Loading data files...")

# 2. Load and Prepare the Historical Data
hist_df = pd.read_parquet(historical_data_path)

# Rename 'timestamp' to 'timestamps' if necessary, and convert to datetime
if 'timestamp' in hist_df.columns:
    hist_df = hist_df.rename(columns={"timestamp": "timestamps"})
hist_df['timestamps'] = pd.to_datetime(hist_df['timestamps'])

# ISOLATE CONTEXT WINDOW: Keep ONLY the exact 400 candlesticks used for the prediction
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
    name='Historical Context (400 mins)',
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
    name='Kronos Forecast (120 mins)',
    increasing_line_color='#FF8C00', # Orange
    decreasing_line_color='#FF0000'  # Red
))

# 7. Configure Layout
fig.update_layout(
    title='RELIANCE-EQ: Model Input Context vs Kronos Forecast',
    yaxis_title='Stock Price',
    xaxis_title='Date / Time',
    xaxis_rangeslider_visible=True, # Interactive timeline slider at the bottom
    template='plotly_dark',         # Dark mode background
    xaxis=dict(
        type='date', # Ensures the x-axis properly handles the datetime timeline
        rangebreaks=[dict(bounds=["sat", "mon"])]
    )
)

# 8. Export to HTML
chart_path = "./Kronos/_results/interactive_forecast.html"
fig.write_html(chart_path)
print(f"Interactive chart successfully saved to {chart_path}")