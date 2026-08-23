import os
import sys
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import config

repo_path = getattr(config, "KRONOS_REPO_PATH", "/home/soq/Kronos")
for p in [config.PROJECT_ROOT, repo_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_utils import get_base_model, get_tokenizer
from model.kronos import KronosPredictor

def normalize_dates(series) -> pd.Series:
    dt_series = pd.to_datetime(series)
    if hasattr(dt_series, "dt") and dt_series.dt.tz is not None:
        dt_series = dt_series.dt.tz_localize(None)
    return pd.Series(pd.to_datetime(dt_series.values)).reset_index(drop=True)

def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower()
    df['date'] = normalize_dates(df['date'])
    
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'amount' not in df.columns:
        df['amount'] = df['close'] * df['volume']
    else:
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
    return df.ffill().bfill().sort_values('date').reset_index(drop=True)

def load_local_2021_2025_data() -> pd.DataFrame:
    parquet_path = os.path.join(config.PROJECT_ROOT, "data/reliance_2021_2025.parquet")
    py_path = "/home/soq/__shutupandbendover/devansh-swc-wala/my_kronos_project/data/reliance_2021_2025.py"

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(py_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location("reliance_2021_2025", py_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        df = getattr(mod, "df", pd.DataFrame(getattr(mod, "data", [])))
    else:
        raise FileNotFoundError(f"Neither {parquet_path} nor {py_path} was found.")

    return prepare_df(df)

def predict_long_horizon(predictor: KronosPredictor, df_train: pd.DataFrame, target_dates: pd.Series, step_size: int = 150) -> pd.DataFrame:
    current_history = df_train.iloc[-config.LOOKBACK_WINDOW:].copy().reset_index(drop=True)
    target_dates_clean = normalize_dates(target_dates)
    
    all_predictions = []
    total_steps = len(target_dates_clean)
    num_chunks = int(np.ceil(total_steps / step_size))
    
    print(f"--> Running {num_chunks} autoregressive rollout chunks ({total_steps} total target trading days)...")
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']

    for i in range(num_chunks):
        start_idx = i * step_size
        end_idx = min((i + 1) * step_size, total_steps)
        
        chunk_dates = target_dates_clean.iloc[start_idx:end_idx].reset_index(drop=True)
        x_dates = current_history['date'].reset_index(drop=True)
        last_known_close = float(current_history['close'].iloc[-1])
        
        chunk_pred = predictor.predict(
            df=current_history[required_cols],
            x_timestamp=x_dates,
            y_timestamp=chunk_dates,
            pred_len=len(chunk_dates),
            T=0.7,
            top_p=0.85,
            sample_count=1
        )
        
        chunk_pred = pd.DataFrame(chunk_pred)
        chunk_pred.columns = [c.lower() for c in chunk_pred.columns]
        
        for col in required_cols:
            if col in chunk_pred.columns:
                chunk_pred[col] = pd.to_numeric(chunk_pred[col], errors='coerce')
        chunk_pred = chunk_pred.ffill().bfill()
        
        first_pred_close = float(chunk_pred['close'].iloc[0])
        if first_pred_close < 50.0 and last_known_close > 100.0:
            scale_mult = last_known_close / max(first_pred_close, 1e-4)
            for c in ['open', 'high', 'low', 'close']:
                if c in chunk_pred.columns:
                    chunk_pred[c] = chunk_pred[c] * scale_mult

        # Explicitly assign genuine datetime series without NaT conversion
        chunk_pred['date'] = chunk_dates.tolist()
        
        if 'amount' not in chunk_pred.columns or chunk_pred['amount'].isna().any():
            chunk_pred['amount'] = chunk_pred['close'] * chunk_pred['volume']
            
        all_predictions.append(chunk_pred[['date'] + required_cols])
        
        # Roll forward context
        current_history = pd.concat([current_history[['date'] + required_cols], chunk_pred[['date'] + required_cols]], ignore_index=True)
        current_history = current_history.iloc[-config.LOOKBACK_WINDOW:].reset_index(drop=True)

    full_pred_df = pd.concat(all_predictions, ignore_index=True)
    full_pred_df['date'] = pd.to_datetime(target_dates_clean.tolist())
    return full_pred_df

def run_baseline_2021_2025():
    # 1. Load Data
    df_train_raw = pd.read_parquet(config.DATA_TRAIN_PATH)
    df_train = prepare_df(df_train_raw)

    df_actual = load_local_2021_2025_data()
    target_dates = df_actual['date']

    # 2. Run Kronos Base Model
    print("--> Initializing Kronos base model and tokenizer...")
    base_model = get_base_model()
    tokenizer = get_tokenizer()
    predictor = KronosPredictor(base_model, tokenizer, max_context=config.MAX_CONTEXT)

    pred_df = predict_long_horizon(predictor, df_train, target_dates, step_size=150)

    # 3. Print Detailed Verification
    print("\n--- Diagnostic Check ---")
    print(f"Target steps: {len(df_actual)} | Predicted steps: {len(pred_df)}")
    print(f"Actual price range: {df_actual['close'].min():.2f} to {df_actual['close'].max():.2f} INR")
    print(f"Pred price range  : {pred_df['close'].min():.2f} to {pred_df['close'].max():.2f} INR")
    print(f"Pred head values  :\n{pred_df[['date', 'close']].head(3)}")
    print("------------------------\n")

    # 4. Interactive Plotly Graph
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_actual['date'],
        y=df_actual['close'],
        mode='lines',
        name='Actual Reliance Close (2021–2025)',
        line=dict(color='#0f172a', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=pred_df['date'],
        y=pred_df['close'],
        mode='lines',
        name='Untrained Base Kronos (400-day lookback)',
        line=dict(color='#ea580c', width=2.2, dash='dash')
    ))

    fig.update_layout(
        title="Reliance Industries: Untrained Base Kronos Forecast vs Actual (2021–2025)",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    html_path = os.path.join(config.PROJECT_ROOT, "baseline_2021_2025_plot.html")
    fig.write_html(html_path)
    print(f"✓ Saved verified interactive plot to: {html_path}")

if __name__ == "__main__":
    run_baseline_2021_2025()