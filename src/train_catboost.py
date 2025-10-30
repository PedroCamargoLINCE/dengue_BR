"""CatBoost training functions."""

import pandas as pd
import numpy as np
from typing import Dict

from .features import assemble_tabular_features, make_calendar_features
from .folds import filter_fold_data, get_prediction_targets


def train_one_fold_catboost(
    city_id: int,
    fold: Dict,
    city_df: pd.DataFrame,
    neighbor_lags_df: pd.DataFrame,
    cfg: dict
) -> pd.DataFrame:
    """Train CatBoost model for one fold and return predictions.
    
    Args:
        city_id: Municipality ID
        fold: Fold information
        city_df: City-specific DataFrame
        neighbor_lags_df: Neighbor lag features
        cfg: Configuration dictionary
        
    Returns:
        DataFrame with predictions
    """
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError("CatBoost not installed. Please install with: pip install catboost")
    
    # Create calendar features
    calendar_df = make_calendar_features(city_df)
    
    # Assemble tabular features
    features_df = assemble_tabular_features(
        city_df, neighbor_lags_df, cfg['climate_cols'], 
        cfg['static_cols'], calendar_df, cfg['lookback_L'], cfg['target_col']
    )
    
    # Get training data
    train_data = filter_fold_data(features_df, fold, split='all_train')
    
    # Prepare features and targets
    feature_cols = []
    
    # Own target lags
    for lag in cfg['lags_target']:
        lag_col = f"{cfg['target_col']}_lag{lag}"
        if lag_col in train_data.columns:
            feature_cols.append(lag_col)
    
    # Neighbor lags
    neighbor_cols = [col for col in train_data.columns if col.startswith('neighbor_')]
    feature_cols.extend(neighbor_cols)
    
    # Climate lags
    for col in cfg['climate_cols']:
        for lag in cfg['lags_climate']:
            lag_col = f"{col}_lag{lag}"
            if lag_col in train_data.columns:
                feature_cols.append(lag_col)
    
    # Static features
    feature_cols.extend(cfg['static_cols'])
    
    # Calendar features
    calendar_cols = ['weekofyear_sin', 'weekofyear_cos']
    if all(col in train_data.columns for col in calendar_cols):
        feature_cols.extend(calendar_cols)
    
    # Remove any missing columns
    feature_cols = [col for col in feature_cols if col in train_data.columns]
    
    if not feature_cols:
        raise ValueError(f"No valid features found for city {city_id}")
    
    # Prepare training data
    X_train = train_data[feature_cols].fillna(0)
    
    # Create multi-step targets
    targets = []
    valid_indices = []
    
    for i, row in train_data.iterrows():
        # Get future values for each horizon
        future_values = []
        current_date = row['week_start_date']
        
        all_valid = True
        for h in range(1, cfg['horizon'] + 1):
            future_date = current_date + pd.Timedelta(weeks=h)
            future_row = city_df[city_df['week_start_date'] == future_date]
            
            if not future_row.empty:
                future_values.append(future_row[cfg['target_col']].iloc[0])
            else:
                all_valid = False
                break
        
        if all_valid and len(future_values) == cfg['horizon']:
            targets.append(future_values)
            valid_indices.append(i)
    
    if not targets:
        raise ValueError(f"No valid targets found for city {city_id}")
    
    # Filter to valid samples
    X_train_valid = X_train.loc[valid_indices].reset_index(drop=True)
    y_train = np.array(targets)
    
    # Train single multi-output model
    print(f"Training CatBoost multi-output model for {cfg['horizon']} horizons...")
    
    model = CatBoostRegressor(
        loss_function='MultiRMSE',  # Multi-output loss
        depth=cfg['catboost']['depth'],
        learning_rate=cfg['catboost']['learning_rate'],
        iterations=cfg['catboost']['iterations'],
        early_stopping_rounds=cfg['catboost']['early_stopping_rounds'],
        verbose=50,  # Print every 50 iterations
        random_seed=cfg['seed'] + fold['fold_id']
    )
    
    # Fit on all horizons at once
    model.fit(X_train_valid, y_train)  # y_train is [samples, horizons]
    print(f"  ✓ Multi-output model completed")
    
    # Generate predictions for this fold
    predictions = _generate_catboost_predictions(
        model, city_df, neighbor_lags_df, fold, cfg, city_id, feature_cols
    )
    
    return predictions


def _generate_catboost_predictions(
    model,  # Single multi-output model instead of dict of models
    city_df: pd.DataFrame,
    neighbor_lags_df: pd.DataFrame,
    fold: Dict,
    cfg: dict,
    city_id: int,
    feature_cols: list
) -> pd.DataFrame:
    """Generate predictions for fold origins."""
    from .folds import filter_fold_data, get_prediction_targets
    
    predictions = []
    
    # Get all training data
    calendar_df = make_calendar_features(city_df)
    features_df = assemble_tabular_features(
        city_df, neighbor_lags_df, cfg['climate_cols'], 
        cfg['static_cols'], calendar_df, cfg['lookback_L'], cfg['target_col']
    )
    
    train_data = filter_fold_data(features_df, fold, split='all_train')
    
    # Get the last training sample for prediction
    if not train_data.empty:
        last_sample = train_data.iloc[-1:][feature_cols].fillna(0)
        
        # Get true values
        targets = get_prediction_targets(city_df, fold, cfg['target_col'])
        
        # Generate multi-output predictions
        pred_values = model.predict(last_sample)[0]  # Returns [h1, h2, h3, h4]
        
        # Create prediction records for each horizon
        for h in range(cfg['horizon']):
            target_row = targets[targets['horizon'] == h + 1]
            if not target_row.empty:
                predictions.append({
                    'municipality_id': city_id,
                    'date_origin': fold['origin_date'],
                    'horizon': h + 1,
                    'y_true': target_row['y_true'].iloc[0],
                    'y_pred': pred_values[h],  # Use h-th output
                    'model_name': 'CatBoost',
                    'fold_id': fold['fold_id'],
                    'train_end_date': fold['train_end_date']
                })
    
    return pd.DataFrame(predictions)