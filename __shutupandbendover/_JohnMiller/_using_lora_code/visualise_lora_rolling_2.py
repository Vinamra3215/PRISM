import pandas as pd
import plotly.graph_objects as go

# 1. Define File Paths (Points to the LoRA rolling predictions file)
historical_data_path = "./__shutupandbendover/_JohnMiller/stock_data/infosys_5y_1d.parquet"
forecast_data_path = "./__shutupandbendover/_JohnMiller/predicted_data/infosys_rolling_lora_predictions.parquet"

# 2. Context timeframe matching the rolling model's initial window
start_time = '2022-01-01'
end_time = '2023-12-31'

print("Loading data files...")

# 3. Load Full Historical Data
hist_df = pd.read_parquet(historical_data_path)
hist_df['timestamps'] = pd.to_datetime(hist_df['timestamps'])

# 4. Filter Historical Context
context_mask = (hist_df['timestamps'] >= start_time) & (hist_df['timestamps'] <= end_time)
hist_context = hist_df.loc[context_mask].reset_index(drop=True)

# 5. Load and Filter LoRA Predictions
pred_df = pd.read_parquet(forecast_data_path)

# Isolate t+1 predictions
pred_t1_df = pred_df[pred_df['horizon_step'] == 't+1'].copy()
pred_t1_df = pred_t1_df.sort_values('target_date')

# 6. Extract Actual Future Reality
pred_start = pred_t1_df['target_date'].min()
pred_end = pred_t1_df['target_date'].max()

future_mask = (hist_df['timestamps'] >= pred_start) & (hist_df['timestamps'] <= pred_end)
hist_actual_future = hist_df.loc[future_mask].reset_index(drop=True)

# 7. Build Plotly Interactive Chart
print("Building interactive comparison chart...")
fig = go.Figure()

# Trace 1: Historical In-Sample Context (Cyan)
fig.add_trace(go.Scatter(
    x=hist_context['timestamps'],
    y=hist_context['close'],
    mode='lines',
    name='Actual Past Close',
    line=dict(color='#17BECF', width=2)
))

# Trace 2: LoRA Kronos Forecast (Neon green dotted)
fig.add_trace(go.Scatter(
    x=pred_t1_df['target_date'],
    y=pred_t1_df['close'],
    mode='lines',
    name='LoRA Kronos Forecast (t+1)',
    line=dict(color='#00FF00', width=2, dash='dot')
))

# Trace 3: Ground Truth Reality (Orange)
fig.add_trace(go.Scatter(
    x=hist_actual_future['timestamps'],
    y=hist_actual_future['close'],
    mode='lines',
    name='Actual Future Close (Reality)',
    line=dict(color='#FF8C00', width=2)
))

# 8. Configure Layout
fig.update_layout(
    title=f'Infosys Close: Past Context ({start_time} to {end_time}) vs LoRA t+1 Forecast vs Reality',
    yaxis_title='Closing Price',
    xaxis_title='Date',
    xaxis_rangeslider_visible=True,
    template='plotly_dark',
    xaxis=dict(
        type='date',
        rangebreaks=[dict(bounds=["sat", "mon"])]
    )
)

# 9. Export to HTML
chart_path = "./__shutupandbendover/_JohnMiller/predicted_data/infosys_lora_rolling_forecast_chart.html"
fig.write_html(chart_path)
print(f"Interactive chart successfully saved to {chart_path}")