import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error
from peft import LoraConfig, get_peft_model

import config

repo_path = getattr(config, "KRONOS_REPO_PATH", "/home/soq/Kronos")
for p in [config.PROJECT_ROOT, repo_path]:
    if p not in sys.path:
        sys.path.insert(0, p)

from model_utils import get_base_model, get_tokenizer
from model.kronos import KronosPredictor
from walk_forward_dataset import load_nifty_data, tokenize_window, RollingPatchDataset

def build_lora_model(base_model):
    """Wraps LoRA adapter layers around Kronos attention projections without HF CausalLM constraints."""
    # Find matching linear layer keys in Kronos
    available_modules = set()
    for name, _ in base_model.named_modules():
        for target in config.LORA_TARGET_MODULES:
            if name.endswith(target) or target in name:
                available_modules.add(name.split(".")[-1])

    target_mods = list(available_modules) if available_modules else ["q_proj", "v_proj", "k_proj", "out_proj"]

    lora_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=target_mods,
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type=None  # Explicitly set to None for custom PyTorch modules
    )
    return get_peft_model(base_model, lora_config)

def adapt_lora_step(model, dataloader, epochs=getattr(config, "ADAPT_EPOCHS", 3)):
    """Runs fast online gradient steps on the current 442-day window."""
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(epochs):
        for batch in dataloader:
            s1_in = batch['s1_ids'].to(config.DEVICE)
            s2_in = batch['s2_ids'].to(config.DEVICE)
            s1_tgt = batch['s1_targets'].to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(s1_ids=s1_in, s2_ids=s2_in)
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('s1_logits', list(outputs.values())[0]))
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            elif hasattr(outputs, 's1_logits'):
                logits = outputs.s1_logits
            elif isinstance(outputs, (tuple, list)):
                logits = outputs[0]
            else:
                logits = outputs
                
            loss = criterion(logits.reshape(-1, logits.size(-1)), s1_tgt.reshape(-1))
            loss.backward()
            optimizer.step()

def run_walk_forward_pipeline():
    df_nifty = load_nifty_data()
    total_candles = len(df_nifty)
    window_size = getattr(config, "ROLLING_TRAIN_WINDOW", 442)
    seq_len = getattr(config, "STEP_SEQ_LEN", getattr(config, "SEQ_LEN", 32))
    
    print(f"--> Loaded Nifty50 Data: {total_candles} total candles from {df_nifty['date'].min().date()} to {df_nifty['date'].max().date()}")
    
    if total_candles <= window_size:
        raise ValueError(f"Dataset has {total_candles} candles, which is not enough for a {window_size}-day training window!")

    tokenizer = get_tokenizer()
    if hasattr(tokenizer, 'to'):
        tokenizer = tokenizer.to(config.DEVICE)
    tokenizer.eval()

    predicted_close = [np.nan] * window_size
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    
    total_steps_to_predict = total_candles - window_size
    print(f"--> Starting Walk-Forward Training Loop ({total_steps_to_predict} days to predict)...")

    for step in range(total_steps_to_predict):
        train_start = step
        train_end = step + window_size
        predict_idx = train_end
        
        target_date = df_nifty['date'].iloc[predict_idx]
        actual_target_close = df_nifty['close'].iloc[predict_idx]
        
        # 1. Slice 442-day rolling history
        window_train_df = df_nifty.iloc[train_start:train_end].reset_index(drop=True)
        
        # 2. Tokenize window
        s1_stream, s2_stream = tokenize_window(window_train_df, tokenizer)
        train_ds = RollingPatchDataset(s1_stream, s2_stream, seq_len=seq_len)
        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        
        # 3. Instantiate fresh base model + LoRA adapter
        base_model = get_base_model().to(config.DEVICE)
        model = build_lora_model(base_model)
        adapt_lora_step(model, train_loader, epochs=getattr(config, "ADAPT_EPOCHS", 3))
        
        # 4. Predict Day 443 with native Kronos temporal embeddings
        model.eval()
        predictor = KronosPredictor(model, tokenizer, max_context=window_size)
        
        x_features = window_train_df[feature_cols]
        x_timestamp = pd.to_datetime(window_train_df['date'])
        y_timestamp = pd.to_datetime(df_nifty['date'].iloc[predict_idx:predict_idx+1])
        
        with torch.no_grad():
            pred_res = predictor.predict(
                df=x_features,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=1
            )
            
        if isinstance(pred_res, pd.DataFrame):
            pred_c = float(pred_res['close'].iloc[-1])
        else:
            pred_c = float(pred_res[-1, 3])
            
        predicted_close.append(pred_c)
        
        # Clean up models from memory per step
        del model
        del base_model
        torch.cuda.empty_cache()

        if (step + 1) % 10 == 0 or step == total_steps_to_predict - 1:
            print(f"Step [{step + 1:03d}/{total_steps_to_predict:03d}] Date: {target_date.date()} | "
                  f"Actual: {actual_target_close:.2f} | Pred: {pred_c:.2f} | Error: {abs(actual_target_close - pred_c):.2f}")

    # Save CSV outputs
    df_nifty['predicted_close'] = predicted_close
    df_nifty.to_csv(config.WALK_FORWARD_CSV, index=False)
    print(f"✓ Saved walk-forward predictions to: {config.WALK_FORWARD_CSV}")

    # Metrics on predicted portion (from Day 443 onward)
    valid_mask = ~df_nifty['predicted_close'].isna()
    y_true = df_nifty.loc[valid_mask, 'close'].values
    y_pred = df_nifty.loc[valid_mask, 'predicted_close'].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\n================ NIFTY 50 WALK-FORWARD EVALUATION ================")
    print(f"Total Predicted Days : {len(y_true)}")
    print(f"Walk-Forward RMSE    : {rmse:.2f} pts")
    print(f"Walk-Forward MAE     : {mae:.2f} pts")
    print(f"Walk-Forward MAPE    : {mape:.2f}%")
    print(f"==================================================================")

    # Plotly interactive chart
    fig = go.Figure()
    
    # 1. Ground Truth Trace (Full 2022–2026 timeline)
    fig.add_trace(go.Scatter(
        x=df_nifty['date'],
        y=df_nifty['close'],
        mode='lines',
        name='NIFTY 50 Actual (Ground Truth)',
        line=dict(color='#0f172a', width=2.2)
    ))
    
    # 2. Walk-Forward LoRA Trace (Starts strictly at Day 443)
    fig.add_trace(go.Scatter(
        x=df_nifty['date'],
        y=df_nifty['predicted_close'],
        mode='lines',
        name='LoRA Walk-Forward Prediction (442-day rolling)',
        line=dict(color='#ef4444', width=2.0)
    ))
    
    # 3. Vertical boundary line where predictions begin
    split_date = df_nifty['date'].iloc[window_size]
    fig.add_vline(
        x=split_date,
        line_width=1.5,
        line_dash="dash",
        line_color="#64748b",
        annotation_text="Walk-Forward Predictions Begin (Day 443)",
        annotation_position="top right"
    )

    fig.update_layout(
        title="NIFTY 50 (2022–2026): Walk-Forward 442-Day Rolling LoRA Prediction vs Actual",
        xaxis_title="Date",
        yaxis_title="Index Level (Points)",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    fig.write_html(config.WALK_FORWARD_OUTPUT_HTML)
    print(f"✓ Saved interactive comparison chart to: {config.WALK_FORWARD_OUTPUT_HTML}")

if __name__ == "__main__":
    run_walk_forward_pipeline()