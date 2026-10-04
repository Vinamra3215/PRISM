import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def optimize_and_rank_xgboost(filepath, features_file):
    print("--- STEP C2: XGBoost Feature Ranking (Quick Fix: Excluding Long-Window Features) ---")
    df = pd.read_parquet(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Load surviving features from Step C1
    with open(features_file, 'r') as f:
        surviving_features = [line.strip() for line in f]
        
    # Features to exclude due to long lookback / NaN time-leak effects
    long_window_features = [
        '52-Week High Distance', 
        'ma_ratio_50_200'
    ]
    
    # Filter out long-window features
    surviving_features = [f for f in surviving_features if f not in long_window_features]
    print(f"Excluded long-window features: {long_window_features}")
    print(f"Active features for XGBoost evaluation: {len(surviving_features)}")
    
    # Standard cleanup: Drop rows where target is missing
    df_clean = df.dropna(subset=['next_day_return'])
    
    train_data = df_clean[df_clean['date'].dt.year == 2022]
    valid_data = df_clean[df_clean['date'].dt.year == 2023]
    
    X_train, y_train = train_data[surviving_features], train_data['next_day_return']
    X_valid, y_valid = valid_data[surviving_features], valid_data['next_day_return']

    print(f"Training Rows (2022): {len(X_train)}")
    print(f"Validation Rows (2023): {len(X_valid)}")
    
    if len(X_train) == 0 or len(X_valid) == 0:
        raise ValueError("Error: Train or Validation set is empty. Check your dataset dates.")

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 2, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'importance_type': 'gain',
            'random_state': 42,
            'early_stopping_rounds': 10  
        }
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
        preds = model.predict(X_valid)
        return np.sqrt(mean_squared_error(y_valid, preds))

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)
    
    print("\nOptuna Best Parameters:", study.best_params)
    
    # Train final model with best params
    final_params = study.best_params.copy()
    final_params['importance_type'] = 'gain'
    final_params['random_state'] = 42
    final_params['early_stopping_rounds'] = 10  
    
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    
    xgb_df = pd.DataFrame({
        'Feature': surviving_features,
        'XGB_Gain': final_model.feature_importances_
    }).sort_values(by='XGB_Gain', ascending=False).reset_index(drop=True)
    
    xgb_df['XGB_Rank'] = xgb_df.index + 1
    
    # Save output to Combined_features_XG_Boost directory
    output_csv = '/home/soq/__shutupandbendover/devansh-swc-wala/Block2/Combined_features_XG_Boost/xgboost_rankings_c2.csv'
    xgb_df.to_csv(output_csv, index=False)
    print(f"Step C2 Complete: Clean rankings saved to '{output_csv}'.")

if __name__ == '__main__':
    optimize_and_rank_xgboost(
        '/home/soq/__shutupandbendover/devansh-swc-wala/Block2/Combined_features_XG_Boost/nifty50_feature_table_2022_present.parquet',
        'surviving_features_c1.txt'
    )