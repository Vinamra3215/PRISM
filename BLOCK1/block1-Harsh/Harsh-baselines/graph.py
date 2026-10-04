import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 1. Exact Data from Kronos Baseline Report (RELIANCE.NS)
dates = [
    "2024-11-14",
    "2024-11-22",
    "2024-11-28",
    "2024-12-04",
    "2024-12-10",
    "2024-12-16",
    "2024-12-20",
    "2024-12-30",
]

actual_close = [1256.75, 1254.57, 1259.92, 1297.74, 1273.85, 1257.44, 1194.98, 1200.33]
naive_pred   = [1241.33, 1241.33, 1241.33, 1241.33, 1241.33, 1241.33, 1241.33, 1241.33]
kronos_pred  = [1243.60, 1229.99, 1218.64, 1211.62, 1195.61, 1196.66, 1177.20, 1165.52]
kronos_err   = [-13.15,  -24.58,  -41.28,  -86.13,  -78.24,  -60.78,  -17.78,  -34.82]

# 2. Initialize Subplots (Top: Price Trajectory, Bottom: Error Bar Chart)
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.12,
    row_heights=[0.7, 0.3],
    subplot_titles=(
        "<b>30-Day Evaluation: Price Trajectory (RELIANCE.NS)</b>",
        "<b>Kronos Forecast Horizon Drift Error (₹)</b>"
    )
)

# --- TOP PANEL TRACES ---

# Trace 1: Actual Close Price
fig.add_trace(
    go.Scatter(
        x=dates,
        y=actual_close,
        mode="lines+markers",
        name="Actual Close",
        line=dict(color="#10b981", width=3),  # Emerald Green
        marker=dict(size=8),
        hovertemplate="<b>Date:</b> %{x}<br><b>Actual:</b> ₹%{y:.2f}<extra></extra>",
    ),
    row=1, col=1
)

# Trace 2: Kronos Candidate Prediction
fig.add_trace(
    go.Scatter(
        x=dates,
        y=kronos_pred,
        mode="lines+markers",
        name="Kronos-base",
        line=dict(color="#f43f5e", width=2.5, dash="dash"),  # Rose Pink
        marker=dict(size=7),
        hovertemplate="<b>Date:</b> %{x}<br><b>Kronos Pred:</b> ₹%{y:.2f}<extra></extra>",
    ),
    row=1, col=1
)

# Trace 3: Naive Persistence Baseline
fig.add_trace(
    go.Scatter(
        x=dates,
        y=naive_pred,
        mode="lines",
        name="Naive Baseline (₹1,241.33)",
        line=dict(color="#eab308", width=2, dash="dot"),  # Amber
        hovertemplate="<b>Date:</b> %{x}<br><b>Naive Pred:</b> ₹%{y:.2f}<extra></extra>",
    ),
    row=1, col=1
)

# --- BOTTOM PANEL TRACE ---

# Trace 4: Kronos Forecast Error Bars
fig.add_trace(
    go.Bar(
        x=dates,
        y=kronos_err,
        name="Kronos Error",
        marker_color="#f43f5e",
        opacity=0.6,
        hovertemplate="<b>Date:</b> %{x}<br><b>Error:</b> ₹%{y:.2f}<extra></extra>",
    ),
    row=2, col=1
)

# 3. Interactive Date Slider Steps (Step through Checkpoint Focus)
steps = []
for i, date in enumerate(dates):
    step = dict(
        method="relayout",
        args=[{
            "title.text": f"<b>RELIANCE.NS Baseline Evaluation — Highlight: Checkpoint {date}</b><br>"
                          f"<span style='font-size:13px; color:#94a3b8;'>Actual: ₹{actual_close[i]:.2f} | Kronos: ₹{kronos_pred[i]:.2f} | Error: ₹{kronos_err[i]:.2f}</span>"
        }],
        label=date,
    )
    steps.append(step)

# Add "Reset View" step at start
steps.insert(0, dict(
    method="relayout",
    args=[{
        "title.text": "<b>RELIANCE.NS 30-Day Model Baseline Evaluation</b><br>"
                      "<span style='font-size:13px; color:#94a3b8;'>MAE — Naive: ₹31.84 (2.52%) | Kronos-base: ₹53.75 (4.24%)</span>"
    }],
    label="All Checkpoints"
))

sliders = [
    dict(
        active=0,
        currentvalue={"prefix": "Evaluation Date: ", "font": {"size": 14, "color": "#f8fafc"}},
        pad={"t": 35, "b": 10},
        steps=steps,
        bgcolor="#1e293b",
        activebgcolor="#38bdf8",
        bordercolor="#334155",
        font=dict(color="#94a3b8"),
    )
]

# 4. Fullscreen & Dark Mode Layout Configuration
fig.update_layout(
    title=dict(
        text="<b>RELIANCE.NS 30-Day Model Baseline Evaluation</b><br>"
             "<span style='font-size:13px; color:#94a3b8;'>MAE — Naive: ₹31.84 (2.52%) | Kronos-base: ₹53.75 (4.24%)</span>",
        font=dict(size=18, color="#f8fafc"),
        x=0.01,
        xanchor="left",
    ),
    template="plotly_dark",
    autosize=True,
    paper_bgcolor="#0f172a",
    plot_bgcolor="#0f172a",
    margin=dict(l=60, r=40, t=90, b=90),
    sliders=sliders,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=12),
        bgcolor="rgba(15, 23, 42, 0.8)",
    ),
)

# Configure Axes & Range Slider
fig.update_xaxes(
    showgrid=True,
    gridcolor="#334155",
    type="category",  # Clean categorical spacing for trading dates
    row=1, col=1
)
fig.update_xaxes(
    showgrid=True,
    gridcolor="#334155",
    type="category",
    rangeslider=dict(visible=False),
    row=2, col=1
)

fig.update_yaxes(
    title_text="Price (₹)",
    showgrid=True,
    gridcolor="#334155",
    zerolinecolor="#334155",
    row=1, col=1
)
fig.update_yaxes(
    title_text="Error (₹)",
    showgrid=True,
    gridcolor="#334155",
    zerolinecolor="#f43f5e",
    row=2, col=1
)

# 5. Render Responsive Fullscreen in Browser
fig.show(config={"responsive": True, "displayModeBar": True})

# Optional: Export directly to a standalone full-screen HTML file
# fig.write_html("reliance_kronos_report.html", full_html=True, include_plotlyjs="cdn", config={"responsive": True})