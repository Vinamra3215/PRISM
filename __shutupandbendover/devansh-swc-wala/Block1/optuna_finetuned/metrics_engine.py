import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

def calculate_return_and_kline_metrics(df_actual: pd.DataFrame, df_pred: pd.DataFrame) -> dict:
    """
    Calculates comprehensive risk, return-based, and candlestick color mismatch metrics.
    All core risk metrics are calculated strictly on percentage returns.
    """
    valid_mask = ~df_pred['close'].isna()
    act = df_actual.loc[valid_mask].copy().reset_index(drop=True)
    prd = df_pred.loc[valid_mask].copy().reset_index(drop=True)
    
    metrics = {}
    
    # 1. 1-Day Return Calculations
    prev_actual_close = act['close'].shift(1).dropna().values
    act_close_slice = act['close'].iloc[1:].values
    pred_close_slice = prd['close'].iloc[1:].values
    
    r_true = (act_close_slice - prev_actual_close) / prev_actual_close
    r_pred = (pred_close_slice - prev_actual_close) / prev_actual_close
    
    # 2. Return-Based Regression & Risk Metrics
    metrics['Return_RMSE_%'] = float(np.sqrt(mean_squared_error(r_true, r_pred)) * 100)
    metrics['Return_MAE_%'] = float(mean_absolute_error(r_true, r_pred) * 100)
    metrics['Return_MAPE_%'] = float(np.mean(np.abs((r_true - r_pred) / (np.abs(r_true) + 1e-8))) * 100)
    metrics['Return_R2'] = float(r2_score(r_true, r_pred))
    
    corr, _ = pearsonr(r_true, r_pred)
    metrics['Pearson_Correlation_r'] = float(corr)
    
    # 3. Directional Accuracy (Mean Directional Accuracy - MDA)
    correct_dir = (np.sign(r_true) == np.sign(r_pred))
    metrics['Directional_Accuracy_%'] = float(np.mean(correct_dir) * 100)
    
    # 4. Cumulative Return Analysis
    cum_act_return = ((act['close'].iloc[-1] - act['close'].iloc[0]) / act['close'].iloc[0]) * 100
    cum_pred_return = ((prd['close'].iloc[-1] - prd['close'].iloc[0]) / prd['close'].iloc[0]) * 100
    metrics['Cumulative_Actual_Return_%'] = float(cum_act_return)
    metrics['Cumulative_Predicted_Return_%'] = float(cum_pred_return)
    metrics['Return_Tracking_Error_%'] = float(abs(cum_act_return - cum_pred_return))
    
    # 5. Candlestick Color (Body Direction) Mismatch Analysis
    act_green = (act['close'] >= act['open']).values
    pred_green = (prd['close'] >= prd['open']).values
    
    total_candles = len(act_green)
    color_mismatch = (act_green != pred_green)
    mismatch_count = int(np.sum(color_mismatch))
    
    false_green = int(np.sum(~act_green & pred_green))  # Predicted Green, Market was Red
    false_red = int(np.sum(act_green & ~pred_green))    # Predicted Red, Market was Green
    
    metrics['Total_Candles_Evaluated'] = total_candles
    metrics['Color_Mismatch_Count'] = mismatch_count
    metrics['Color_Mismatch_Rate_%'] = float((mismatch_count / total_candles) * 100)
    metrics['False_Green_Count'] = false_green
    metrics['False_Red_Count'] = false_red
    metrics['Color_Accuracy_%'] = float(100.0 - metrics['Color_Mismatch_Rate_%'])
    
    # 6. OHLC Raw Envelope Point Errors
    for col in ['open', 'high', 'low', 'close']:
        metrics[f'{col.capitalize()}_MAE_Pts'] = float(mean_absolute_error(act[col].values, prd[col].values))
        
    return metrics

def save_and_print_metrics(metrics: dict, output_dir: str, title: str = "QUANTITATIVE RISK & RETURN REPORT"):
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, "risk_metrics_report.txt")
    json_path = os.path.join(output_dir, "risk_metrics_report.json")
    
    report_lines = [
        f"{'=' * 22} {title} {'=' * 22}",
        " [1] RETURN-BASED RISK & PERFORMANCE METRICS",
        f"  • Return RMSE (Vol. of Error)     : {metrics.get('Return_RMSE_%', 0.0):.4f} %",
        f"  • Return MAE                      : {metrics.get('Return_MAE_%', 0.0):.4f} %",
        f"  • Return MAPE                     : {metrics.get('Return_MAPE_%', 0.0):.2f} %",
        f"  • Return R² Score                 : {metrics.get('Return_R2', 0.0):.4f}",
        f"  • Pearson Correlation (r)         : {metrics.get('Pearson_Correlation_r', 0.0):.4f}",
        f"  • Directional Accuracy (MDA)      : {metrics.get('Directional_Accuracy_%', 0.0):.2f} %",
        f"{'-' * 70}",
        " [2] CUMULATIVE RETURN TRACKING",
        f"  • Actual Cumulative Return        : {metrics.get('Cumulative_Actual_Return_%', 0.0):.2f} %",
        f"  • Predicted Cumulative Return     : {metrics.get('Cumulative_Predicted_Return_%', 0.0):.2f} %",
        f"  • Tracking Error Gap              : {metrics.get('Return_Tracking_Error_%', 0.0):.2f} %",
        f"{'-' * 70}",
        " [3] CANDLESTICK COLOR / DIRECTION CONFUSION MATRIX",
        f"  • Total Candles Evaluated         : {metrics.get('Total_Candles_Evaluated', 0)}",
        f"  • Candle Color Match Accuracy     : {metrics.get('Color_Accuracy_%', 0.0):.2f} %",
        f"  • Total Color Mismatches          : {metrics.get('Color_Mismatch_Count', 0)} ({metrics.get('Color_Mismatch_Rate_%', 0.0):.2f} %)",
        f"  • False Green (Pred UP, Act DOWN) : {metrics.get('False_Green_Count', 0)}",
        f"  • False Red   (Pred DOWN, Act UP) : {metrics.get('False_Red_Count', 0)}",
        f"{'-' * 70}",
        " [4] RAW OHLC MAE POINT ERRORS",
        f"  • Open MAE: {metrics.get('Open_MAE_Pts', 0.0):.2f} | High MAE: {metrics.get('High_MAE_Pts', 0.0):.2f} | Low MAE: {metrics.get('Low_MAE_Pts', 0.0):.2f} | Close MAE: {metrics.get('Close_MAE_Pts', 0.0):.2f}",
        f"{'=' * 70}\n"
    ]
    
    report_text = "\n".join(report_lines)
    print(report_text)
    
    with open(txt_path, "w") as f:
        f.write(report_text)
    print(f"✓ Saved Risk Metrics text report to: {txt_path}")
    
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Saved Risk Metrics JSON to: {json_path}")