import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "outputs")
PRED_FILE = os.path.join(
    OUTPUT_DIR, "nifty50_50stocks_walkforward_predictions.csv"
)
METRICS_FILE = os.path.join(OUTPUT_DIR, "cross_sectional_stock_metrics.csv")
RAW_DATA = os.path.join(CURRENT_DIR, "nifty50_constituents_2022_2026.parquet")


def main():
  if not (os.path.exists(PRED_FILE) and os.path.exists(METRICS_FILE)):
    print("Error: Required CSV files missing. Run steps 2 & 3 first.")
    return

  preds = pd.read_csv(PRED_FILE)
  metrics = pd.read_csv(METRICS_FILE)
  raw = pd.read_parquet(RAW_DATA)

  preds["date"] = pd.to_datetime(preds["date"])
  raw["date"] = pd.to_datetime(raw["date"])

  top_5 = metrics.sort_values("Stock_IC", ascending=False)["Symbol"].head(5).values
  print(f"--> Generating dashboards for top 5 stocks: {list(top_5)}")

  for sym in top_5:
    p_sub = (
        preds[preds["symbol"] == sym].sort_values("date").reset_index(drop=True)
    )
    r_sub = (
        raw[raw["symbol"] == sym].sort_values("date").reset_index(drop=True)
    )
    ctx = r_sub[r_sub["date"] < p_sub["date"].iloc[0]].reset_index(drop=True)

    p_sub["rolling_res_20d"] = (
        p_sub["return_residual_%"].rolling(20, min_periods=1).mean()
    )
    bar_colors = [
        "#10b981" if x >= 0 else "#ef4444" for x in p_sub["return_residual_%"]
    ]

    mae = np.mean(np.abs(p_sub["close"] - p_sub["pred_close"]))
    rmse = np.sqrt(np.mean((p_sub["close"] - p_sub["pred_close"]) ** 2))
    mda = (
        np.mean((p_sub["actual_return"] * p_sub["pred_return"]) > 0) * 100.0
    )
    stk_ic = metrics[metrics["Symbol"] == sym]["Stock_IC"].iloc[0]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3],
        subplot_titles=(
            f"<b>{sym} Historical Ingestion & Walk-Forward Candlestick"
            " Forecast</b>",
            "<b>Daily Return Residual [%] (Actual Return − Predicted Return) &"
            " 20D Bias</b>",
        ),
    )

    # Context Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=ctx["date"],
            open=ctx["open"],
            high=ctx["high"],
            low=ctx["low"],
            close=ctx["close"],
            name="Context (2022-Mid 2023)",
            increasing_line_color="#94a3b8",
            decreasing_line_color="#64748b",
            opacity=0.6,
        ),
        row=1,
        col=1,
    )

    # Actual Market Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=p_sub["date"],
            open=p_sub["open"],
            high=p_sub["high"],
            low=p_sub["low"],
            close=p_sub["close"],
            name="Actual Market",
            increasing_line_color="#10b981",
            decreasing_line_color="#ef4444",
        ),
        row=1,
        col=1,
    )

    # LoRA Predicted Candlesticks
    fig.add_trace(
        go.Candlestick(
            x=p_sub["date"],
            open=p_sub["pred_open"],
            high=p_sub["pred_high"],
            low=p_sub["pred_low"],
            close=p_sub["pred_close"],
            name="LoRA Predicted",
            increasing_line_color="#06b6d4",
            decreasing_line_color="#f43f5e",
        ),
        row=1,
        col=1,
    )

    # Residual Bars & 20D Moving Average
    fig.add_trace(
        go.Bar(
            x=p_sub["date"],
            y=p_sub["return_residual_%"],
            marker_color=bar_colors,
            name="Return Residual (%)",
            opacity=0.7,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=p_sub["date"],
            y=p_sub["rolling_res_20d"],
            line=dict(color="#1e3a8a", width=2),
            name="20D Rolling Bias",
        ),
        row=2,
        col=1,
    )

    fig.add_hline(y=0.0, line_color="#334155", row=2, col=1)

    banner_text = (
        f"<b>Stock:</b> {sym} &nbsp;|&nbsp; <b>MAE:</b> {mae:.2f} &nbsp;|&nbsp;"
        f" <b>RMSE:</b> {rmse:.2f} &nbsp;|&nbsp; <b>MDA:</b> {mda:.1f}%"
        f" &nbsp;|&nbsp; <b>Stock IC:</b> {stk_ic:.4f}"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.06,
        text=banner_text,
        showarrow=False,
        font=dict(size=13, color="#0f172a"),
        align="center",
        bgcolor="#f1f5f9",
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=5,
    )

    fig.update_layout(
        height=850,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        margin=dict(t=120, b=40, l=60, r=40),
    )

    out_html = os.path.join(OUTPUT_DIR, f"{sym}_benchmark_dashboard.html")
    fig.write_html(out_html)
    print(f"✓ Saved {sym} dashboard to {out_html}")


if __name__ == "__main__":
  main()