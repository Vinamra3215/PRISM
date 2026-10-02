import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Calculate Daily Returns & Error Metrics
# ---------------------------------------------------------------------------
# Slice out the valid out-of-sample evaluation period
eval_df = pred_export_df.iloc[start_eval_idx:].copy().reset_index(drop=True)

# Previous close values for percentage return calculation
prev_close = pred_export_df['close'].iloc[start_eval_idx - 1 : -1].values

eval_df['actual_return_%'] = ((eval_df['close'] - prev_close) / prev_close) * 100.0
eval_df['pred_return_%'] = ((eval_df['pred_close'] - prev_close) / prev_close) * 100.0
eval_df['return_diff_%'] = eval_df['actual_return_%'] - eval_df['pred_return_%']

# Color difference bars: Green for positive difference, Orange for negative
diff_bar_colors = ['#10b981' if d >= 0 else '#f97316' for d in eval_df['return_diff_%']]

# ---------------------------------------------------------------------------
# Create Multi-Panel Subplot Figure
# ---------------------------------------------------------------------------
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    row_heights=[0.68, 0.32],
    subplot_titles=(
        "NIFTY 50: Actual vs Predicted Candlesticks",
        "Daily Return Spread: (Actual Return % − Predicted Return %)"
    )
)

# 1. Top Panel: Actual Candlesticks
fig.add_trace(go.Candlestick(
    x=df_nifty['date'],
    open=df_nifty['open'],
    high=df_nifty['high'],
    low=df_nifty['low'],
    close=df_nifty['close'],
    name='Actual NIFTY 50 Candles',
    increasing_line_color='#10b981',
    decreasing_line_color='#ef4444',
    opacity=0.65
), row=1, col=1)

# Top Panel: Predicted Candlesticks
valid_mask = ~df_preds['close'].isna()
fig.add_trace(go.Candlestick(
    x=df_nifty.loc[valid_mask, 'date'],
    open=df_preds.loc[valid_mask, 'open'],
    high=df_preds.loc[valid_mask, 'high'],
    low=df_preds.loc[valid_mask, 'low'],
    close=df_preds.loc[valid_mask, 'close'],
    name='Predicted Candles',
    increasing_line_color='#06b6d4',
    decreasing_line_color='#f43f5e',
    opacity=0.85
), row=1, col=1)

# Split marker line at prediction start
split_date = df_nifty['date'].iloc[start_eval_idx]
fig.add_vline(
    x=split_date,
    line_width=1.5,
    line_dash="dash",
    line_color="#64748b",
    annotation_text="Forecast Start",
    annotation_position="top right",
    row=1, col=1
)

# 2. Bottom Panel: Daily Return Difference (Bar Chart)
fig.add_trace(go.Bar(
    x=eval_df['date'],
    y=eval_df['return_diff_%'],
    name='Return Diff (Actual − Pred %)',
    marker_color=diff_bar_colors,
    opacity=0.75
), row=2, col=1)

# Bottom Panel: Actual Return (Line)
fig.add_trace(go.Scatter(
    x=eval_df['date'],
    y=eval_df['actual_return_%'],
    name='Actual Return %',
    mode='lines',
    line=dict(color='#10b981', width=1.2),
    opacity=0.8
), row=2, col=1)

# Bottom Panel: Predicted Return (Line)
fig.add_trace(go.Scatter(
    x=eval_df['date'],
    y=eval_df['pred_return_%'],
    name='Pred Return %',
    mode='lines',
    line=dict(color='#8b5cf6', width=1.2, dash='dot'),
    opacity=0.85
), row=2, col=1)

# Zero reference line for difference panel
fig.add_hline(y=0.0, line_width=1.0, line_dash="solid", line_color="#94a3b8", row=2, col=1)

# ---------------------------------------------------------------------------
# Layout Configuration & HTML Export
# ---------------------------------------------------------------------------
fig.update_layout(
    height=900,
    xaxis_rangeslider_visible=False,
    hovermode="x unified",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

fig.update_yaxes(title_text="Index Points", row=1, col=1)
fig.update_yaxes(title_text="Return Spread (%)", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)

out_html = os.path.join(OUTPUT_DIR, "nifty50_walk_forward_with_returns.html")
fig.write_html(out_html)
print(f"✓ Saved interactive dual-panel dashboard to: {out_html}")