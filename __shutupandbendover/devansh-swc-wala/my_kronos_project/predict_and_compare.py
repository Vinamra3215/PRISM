import os
import sys
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from peft import PeftModel
import config

repo_path = getattr(config, "KRONOS_REPO_PATH", "/home/soq/Kronos")
for p in [config.PROJECT_ROOT, repo_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_utils import get_base_model, get_tokenizer
from model.kronos import KronosPredictor
from dataset import normalize_dates

def load_data():
    df_train = pd.read_parquet(config.DATA_TRAIN_PATH)
    df_train.columns = df_train.columns.str.lower()
    df_train['date'] = normalize_dates(df_train['date'])
    
    df_eval = pd.read_parquet(config.DATA_EVAL_PATH)
    df_eval.columns = df_eval.columns.str.lower()
    df_eval['date'] = normalize_dates(df_eval['date'])
    
    for df in [df_train, df_eval]:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'amount' not in df.columns or df['amount'].isnull().all():
            df['amount'] = df['close'] * df['volume']
            
    df_train = df_train.ffill().bfill().sort_values('date').reset_index(drop=True)
    df_eval = df_eval.ffill().bfill().sort_values('date').reset_index(drop=True)
    return df_train, df_eval

def rolling_predict(predictor, df_history, df_eval, step_size=30):
    """Generates sequential forecasts passing x_timestamp and y_timestamp to KronosPredictor."""
    pred_records = []
    current_context = df_history.copy()
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    
    target_dates = df_eval['date'].reset_index(drop=True)
    total_steps = len(target_dates)
    max_context = getattr(config, "MAX_CONTEXT", 400)
    
    for start_idx in range(0, total_steps, step_size):
        end_idx = min(start_idx + step_size, total_steps)
        chunk_len = end_idx - start_idx
        chunk_dates = target_dates.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # Slicing lookback context and matching historical timestamps
        context_slice = current_context.iloc[-max_context:].reset_index(drop=True)
        x_features = context_slice[feature_cols]
        x_timestamp = pd.to_datetime(context_slice['date'])
        y_timestamp = pd.to_datetime(chunk_dates)
        
        # Native Kronos prediction with timestamp embeddings
        pred_chunk = predictor.predict(
            df=x_features,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=chunk_len
        )
        
        if isinstance(pred_chunk, pd.DataFrame):
            pred_chunk = pred_chunk.copy()
            pred_chunk['date'] = chunk_dates.values
            pred_records.append(pred_chunk)
            current_context = pd.concat([current_context, pred_chunk], ignore_index=True)
        else:
            temp_df = pd.DataFrame(pred_chunk, columns=feature_cols)
            temp_df['date'] = chunk_dates.values
            pred_records.append(temp_df)
            current_context = pd.concat([current_context, temp_df], ignore_index=True)

    return pd.concat(pred_records, ignore_index=True).reset_index(drop=True)

def run_evaluation():
    df_train, df_eval = load_data()
    tokenizer = get_tokenizer()
    max_ctx = getattr(config, "MAX_CONTEXT", 400)

    # 1. Base Untrained Model
    print("--> Forecasting with Untrained Base Model (Native Kronos Decoder)...")
    base_model = get_base_model().to(config.DEVICE)
    base_model.eval()
    base_predictor = KronosPredictor(base_model, tokenizer, max_context=max_ctx)
    base_preds = rolling_predict(base_predictor, df_train, df_eval, step_size=30)

    # 2. Fine-Tuned Model
    print(f"--> Forecasting with LoRA Fine-Tuned Model ({config.BEST_ADAPTER_DIR})...")
    ft_model = PeftModel.from_pretrained(base_model, config.BEST_ADAPTER_DIR).to(config.DEVICE)
    ft_model.eval()
    ft_predictor = KronosPredictor(ft_model, tokenizer, max_context=max_ctx)
    ft_preds = rolling_predict(ft_predictor, df_train, df_eval, step_size=30)

    # 3. Metrics
    actual_close = df_eval['close'].values[:len(base_preds)]
    base_close = base_preds['close'].values[:len(actual_close)]
    ft_close = ft_preds['close'].values[:len(actual_close)]

    rmse_base = np.sqrt(mean_squared_error(actual_close, base_close))
    rmse_ft = np.sqrt(mean_squared_error(actual_close, ft_close))
    mae_ft = mean_absolute_error(actual_close, ft_close)

    print(f"\n================ 2021–2025 STABILIZED EVALUATION ================")
    print(f"Untrained Base Model RMSE : {rmse_base:.2f} INR")
    print(f"LoRA Fine-Tuned Model RMSE: {rmse_ft:.2f} INR")
    print(f"LoRA Fine-Tuned Model MAE : {mae_ft:.2f} INR")
    print(f"==================================================================")

    # 4. Interactive Plotly Graph
    eval_dates = df_eval['date'].values[:len(actual_close)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eval_dates, y=actual_close, mode='lines', name='Actual Price (Ground Truth)', line=dict(color='#0f172a', width=2.5)))
    fig.add_trace(go.Scatter(x=eval_dates, y=base_close, mode='lines', name='Untrained Kronos (Rolling Horizon)', line=dict(color='#ea580c', width=2.0, dash='dash')))
    fig.add_trace(go.Scatter(x=eval_dates, y=ft_close, mode='lines', name='LoRA Fine-Tuned Kronos', line=dict(color='#dc2626', width=2.2)))

    fig.update_layout(
        title="Reliance Industries (2021–2025): Actual vs Untrained vs LoRA Fine-Tuned Kronos",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        hovermode="x unified",
        template="plotly_white"
    )

    html_path = os.path.join(config.PROJECT_ROOT, "reliance_comparison.html")
    fig.write_html(html_path)
    print(f"✓ Saved stabilized interactive comparison HTML to: {html_path}")

if __name__ == "__main__":
    run_evaluation()