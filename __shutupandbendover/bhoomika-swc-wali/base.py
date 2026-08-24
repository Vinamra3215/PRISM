import os
import glob
import pandas as pd
import torch
import plotly.graph_objects as go
from datetime import datetime

# 1. Dynamic Unique Output HTML File
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_HTML_NAME = f"kronos_nifty_base_{timestamp}.html"
OUTPUT_HTML = os.path.join(BASE_DIR, OUTPUT_HTML_NAME)

# 2. Automated Smart Data Loader
def get_nifty_data():
    parquet_path = os.path.join(BASE_DIR, "Kronos", "data", "NIFTY50_5Y_OHLCV.parquet")
    if os.path.exists(parquet_path):
        return pd.read_parquet(parquet_path)
    
    possible_files = glob.glob("/home/soq/**/market data*.csv", recursive=True) + \
                     glob.glob("/home/soq/**/NIFTY50*.parquet", recursive=True)
    if possible_files:
        f = possible_files[0]
        return pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
    return None

df = get_nifty_data()
if df is None:
    raise FileNotFoundError("Error: NIFTY Data file nahi mili!")

# 3. Clean Columns & Robust Column Mapping
df = df.reset_index()

# Clean spaces and normalize column names
df.columns = [str(c).strip() for c in df.columns]

date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
close_col = next((c for c in df.columns if 'close' in c.lower()), None)

if not close_col:
    raise KeyError("Dataset me Close price ka column nahi mila!")

# Standardize main column names
df = df.rename(columns={date_col: 'Date', close_col: 'Close'})

# Convert Date Column
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)

# Clean numeric values (remove commas and force float conversion)
for target in ['Open', 'High', 'Low', 'Close', 'Volume']:
    found = next((c for c in df.columns if c.lower() == target.lower()), None)
    if found:
        df[found] = pd.to_numeric(df[found].astype(str).str.replace(',', '').str.strip(), errors='coerce')

# Drop empty price rows safely
df = df.dropna(subset=['Close']).reset_index(drop=True)

# Default Filter Range: 2023 to 2024
df_filtered = df[(df['Date'] >= '2023-01-01') & (df['Date'] <= '2024-12-31')].copy().reset_index(drop=True)

# 4. Interactive Terminal Selection
print("\n" + "="*55)
print("   KRONOS BASE MODEL - STANDALONE GRAPH GENERATOR")
print("="*55)
print("Default Data Range Loaded: 2023-01-01 to 2024-12-31")

date_input = input("\nEnter Target Date Vector (YYYY-MM-DD) [Press ENTER for Full Graph]: ").strip()

x_range = None
selected_annotation = []

if date_input:
    target_dt = pd.to_datetime(date_input)
    row_data = df_filtered[df_filtered['Date'] == target_dt]
    
    if row_data.empty:
        print("\n[!] Market holiday date. Fetching closest active date...")
        closest_idx = (df_filtered['Date'] - target_dt).abs().idxmin()
        target_dt = df_filtered.loc[closest_idx, 'Date']
        row_data = df_filtered[df_filtered['Date'] == target_dt]
    
    feature_cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in df_filtered.columns]
    vector_values = row_data[feature_cols].values[0]
    vector_tensor = torch.tensor(vector_values, dtype=torch.float32)
    
    print(f"\n[+] Target Vector Extracted for {target_dt.date()}:")
    print(f"    PyTorch Tensor: {vector_tensor}")
    
    x_range = [target_dt - pd.Timedelta(days=7), target_dt + pd.Timedelta(days=7)]
    
    selected_annotation = [dict(
        x=target_dt,
        y=float(row_data['Close'].values[0]),
        xref="x", yref="y",
        text=f"Vector Date: {target_dt.date()}<br>Close: ₹{row_data['Close'].values[0]:,.2f}",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#00E5FF",
        arrowsize=1.2,
        ax=0, ay=-40,
        bgcolor="#1E293B",
        bordercolor="#00E5FF",
        font=dict(color="#FFFFFF", size=12)
    )]
else:
    print("\n[+] Full 2023-2024 NIFTY Price Trajectory plot generate ho raha hai...")
    x_range = [df_filtered['Date'].min(), df_filtered['Date'].max()]

# 5. Plotly Interactive Graph Generation
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_filtered['Date'],
    y=df_filtered['Close'],
    name="NIFTY Price",
    mode='lines',
    line=dict(color='#00E5FF', width=2.0),
    hovertemplate="<b>Date:</b> %{x|%Y-%m-%d}<br><b>Price:</b> ₹%{y:,.2f}<extra></extra>"
))

fig.update_layout(
    title=dict(
        text=f"<b>NIFTY 50 Kronos Base Run ({'Focused Date: ' + date_input if date_input else '2023-2024'})</b>",
        font=dict(size=16, color="#FFFFFF")
    ),
    template="plotly_dark",
    paper_bgcolor="#0A0E14",
    plot_bgcolor="#0A0E14",
    hovermode="x",
    annotations=selected_annotation,
    
    yaxis=dict(
        title=dict(text="<b>NIFTY Close Price (₹)</b>", font=dict(color="#FFFFFF", size=13)),
        tickfont=dict(color="#A0AAB0"),
        gridcolor="#1E2638",
        showgrid=True,
        autorange=True
    ),
    
    xaxis=dict(
        title=dict(text="<b>Time / Date Axis</b>", font=dict(color="#FFFFFF")),
        type="date",
        range=x_range,
        gridcolor="#1E2638",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#888888",
        spikethickness=1,
        spikedash="dash",
        rangeslider=dict(visible=True, thickness=0.12, bgcolor="#0F172A", bordercolor="#1E293B")
    ),
    margin=dict(l=50, r=50, t=70, b=50)
)

# Fully embedded bundle (Offline viewable anywhere)
fig.write_html(OUTPUT_HTML, include_plotlyjs=True, full_html=True, auto_open=False)

print("\n" + "="*60)
print(f"SUCCESS! Unique Graph File Created:")
print(f"--> {OUTPUT_HTML_NAME}")
print("="*60 + "\n")




