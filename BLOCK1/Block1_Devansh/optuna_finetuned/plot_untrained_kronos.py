def plot_benchmark_dashboard(context_df, eval_df, metrics, output_html="outputs/kronos_untrained_benchmark.html"):
    os.makedirs("outputs", exist_ok=True)
    
    # ---------------------------------------------------------------------------
    # Ensure strict datetime format (prevents epoch 1970 fallback)
    # ---------------------------------------------------------------------------
    context_df['date'] = pd.to_datetime(context_df['date'])
    eval_df['date'] = pd.to_datetime(eval_df['date'])
    eval_df['norm_dev'] = metrics['norm_dev']

    x_min = context_df['date'].iloc[0]
    x_max = eval_df['date'].iloc[-1]
    cutoff_ts = eval_df['date'].iloc[0]

    # Create 2-row subplot layout
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=(
            "<b>NIFTY 50: Untrained Kronos Foundation Forecast vs. Market Actuals</b>",
            "<b>Normalized Deviation Spread [% Error = ((Actual - Pred) / Actual) × 100]</b>"
        )
    )

    # 1. Historical Ingestion Window (Context)
    fig.add_trace(go.Candlestick(
        x=context_df['date'],
        open=context_df['open'],
        high=context_df['high'],
        low=context_df['low'],
        close=context_df['close'],
        name="Historical Context (2022 - Mid 2023)",
        increasing_line_color="#94a3b8",
        decreasing_line_color="#64748b",
        opacity=0.6
    ), row=1, col=1)

    # 2. Out-of-Sample Actual Market Candles
    fig.add_trace(go.Candlestick(
        x=eval_df['date'],
        open=eval_df['open'],
        high=eval_df['high'],
        low=eval_df['low'],
        close=eval_df['close'],
        name="Actual Market Candles",
        increasing_line_color="#10b981",
        decreasing_line_color="#ef4444",
        opacity=0.85
    ), row=1, col=1)

    # 3. Model Predictions (High Contrast Cyan/Pink)
    fig.add_trace(go.Candlestick(
        x=eval_df['date'],
        open=eval_df['pred_open'],
        high=eval_df['pred_high'],
        low=eval_df['pred_low'],
        close=eval_df['pred_close'],
        name="Untrained Kronos Zero-Shot",
        increasing_line_color="#06b6d4",
        decreasing_line_color="#f43f5e",
        opacity=0.9
    ), row=1, col=1)

    # Divider Line for Forecast Start
    fig.add_vline(
        x=cutoff_ts,
        line_width=2,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text="<b>Forecast Start</b>",
        annotation_position="top left",
        row=1, col=1
    )

    # 4. Bottom Panel: High-Visibility Deviation Curve & Filled Area
    fig.add_trace(go.Scatter(
        x=eval_df['date'],
        y=eval_df['norm_dev'],
        name="Deviation (% Error)",
        mode="lines",
        line=dict(color="#2563eb", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.2)"
    ), row=2, col=1)

    # Add 0% baseline reference
    fig.add_hline(y=0.0, line_width=1.2, line_color="#334155", row=2, col=1)

    # ---------------------------------------------------------------------------
    # Top Horizontal Metrics Banner
    # ---------------------------------------------------------------------------
    banner_text = (
        f"<b>MAE:</b> {metrics['mae']:.1f} pts &nbsp;|&nbsp; "
        f"<b>RMSE:</b> {metrics['rmse']:.1f} pts &nbsp;|&nbsp; "
        f"<b>MAPE:</b> {metrics['mape']:.2f}% &nbsp;|&nbsp; "
        f"<b>MDA:</b> {metrics['mda']:.1f}% &nbsp;|&nbsp; "
        f"<b>Actual Return:</b> {metrics['cum_true']:.1f}% &nbsp;|&nbsp; "
        f"<b>Pred Return:</b> {metrics['cum_pred']:.1f}% &nbsp;|&nbsp; "
        f"<b>Gap:</b> {metrics['cum_gap']:+.2f}%"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.5, y=1.065,
        text=banner_text,
        showarrow=False,
        font=dict(family="Arial, sans-serif", size=13, color="#0f172a"),
        align="center",
        bgcolor="#f1f5f9",
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=6
    )

    # ---------------------------------------------------------------------------
    # Axes Formatting & Tight Bounding
    # ---------------------------------------------------------------------------
    fig.update_layout(
        title=dict(
            text="<b>Kronos Zero-Shot Foundation Benchmark (Mid-2023 to 2026)</b>",
            x=0.5,
            y=0.98,
            xanchor="center",
            font=dict(size=18, color="#0f172a")
        ),
        margin=dict(t=120, b=50, l=60, r=40),
        height=880,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.005,
            xanchor="right",
            x=1.0
        )
    )

    # Lock x-axis boundaries to dataset dates only
    fig.update_xaxes(range=[x_min, x_max], row=1, col=1)
    fig.update_xaxes(range=[x_min, x_max], title_text="Date", row=2, col=1)

    # Configure y-axes ranges
    fig.update_yaxes(title_text="Index Points", row=1, col=1)
    fig.update_yaxes(title_text="Deviation (%)", zeroline=True, row=2, col=1)

    fig.write_html(output_html)
    print(f"\n✓ Saved updated dashboard to: {os.path.abspath(output_html)}")