"""TFT training functions."""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict

from .models_tft import TFTForecast
from .datasets import create_dataloaders
from .utils import seed_everything, device, clear_gpu_memory


def train_one_fold_tft(
    city_id: int,
    fold: Dict,
    city_df: pd.DataFrame,
    neighbor_lags_df: pd.DataFrame,
    cfg: dict,
    target_scaler=None
) -> pd.DataFrame:
    """Train TFT model for one fold and return predictions.
    
    Args:
        city_id: Municipality ID
        fold: Fold information
        city_df: City-specific DataFrame
        neighbor_lags_df: Neighbor lag features
        cfg: Configuration dictionary
        target_scaler: Optional target scaler
        
    Returns:
        DataFrame with predictions
    """
    # Set seed for reproducibility
    seed_everything(cfg['seed'] + fold['fold_id'])
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        city_df, neighbor_lags_df, cfg, fold, target_scaler, batch_size=32
    )
    
    if len(train_loader) == 0:
        raise ValueError(f"No training data for city {city_id}, fold {fold['fold_id']}")
    
    if len(val_loader) < 2:
        print(f"Warning: Very few validation batches ({len(val_loader)}) for city {city_id}")
    
    # Get feature dimension
    for batch in train_loader:
        input_size = batch[0].shape[-1]
        break
    
    # Initialize model
    model = TFTForecast(
        input_size=input_size,
        hidden_size=cfg['tft']['hidden_size'],
        num_heads=cfg['tft']['num_heads'],
        encoder_layers=cfg['tft']['encoder_layers'],
        decoder_layers=cfg['tft']['decoder_layers'],
        horizon=cfg['horizon'],
        dropout=cfg['tft']['dropout']
    ).to(device())
    
    # Loss and optimizer
    if cfg['tft']['loss'] == 'mae':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['tft']['lr'],
        weight_decay=cfg['tft']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg['tft']['rlrop']['factor'],
        patience=cfg['tft']['rlrop']['patience'],
        cooldown=cfg['tft']['rlrop']['cooldown']
    )
    
    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    patience = cfg['tft']['early_stopping']['patience']
    min_delta = cfg['tft']['early_stopping']['min_delta']
    
    # Training loop
    print(f"Starting TFT training for {cfg['tft']['max_epochs']} epochs...")
    for epoch in range(cfg['tft']['max_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device())
            batch_y = batch_y.to(device())
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            if cfg['tft']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['tft']['grad_clip'])
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device())
                batch_y = batch_y.to(device())
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        # Print progress every 20 epochs or if early stopping
        if (epoch + 1) % 20 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}, LR={current_lr:.2e}")
        
        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.6f})")
            break
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Generate predictions for this fold
    predictions = _generate_tft_predictions(
        model, city_df, neighbor_lags_df, fold, cfg, target_scaler, city_id
    )
    
    # Clear GPU memory
    clear_gpu_memory()
    
    return predictions


def _generate_tft_predictions(
    model: TFTForecast,
    city_df: pd.DataFrame,
    neighbor_lags_df: pd.DataFrame,
    fold: Dict,
    cfg: dict,
    target_scaler,
    city_id: int
) -> pd.DataFrame:
    """Generate predictions for fold origins."""
    from .folds import filter_fold_data, get_prediction_targets
    
    model.eval()
    predictions = []
    
    # Get all training data (including validation)
    train_data = filter_fold_data(city_df, fold, split='all_train')
    train_data = train_data.merge(neighbor_lags_df, on='week_start_date', how='left')
    train_data = train_data.fillna(0)
    train_data = train_data.sort_values('week_start_date').reset_index(drop=True)
    
    # Build features similar to SeqDataset
    feature_cols = [cfg['target_col']]
    
    # Add neighbor columns
    neighbor_cols = [col for col in train_data.columns if col.startswith('neighbor_')]
    feature_cols.extend(neighbor_cols)
    
    # Add climate lags
    for col in cfg['climate_cols']:
        for lag in cfg['lags_climate']:
            lag_col = f"{col}_lag{lag}"
            if lag_col in train_data.columns:
                feature_cols.append(lag_col)
    
    # Add static features
    feature_cols.extend(cfg['static_cols'])
    
    # Add calendar features
    calendar_cols = ['weekofyear_sin', 'weekofyear_cos']
    if all(col in train_data.columns for col in calendar_cols):
        feature_cols.extend(calendar_cols)
    
    # Create input sequence from last L weeks of training data
    if len(train_data) >= cfg['lookback_L']:
        input_data = train_data.iloc[-cfg['lookback_L']:][feature_cols].values
        
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data).unsqueeze(0).to(device())
            pred_tensor = model(input_tensor)
            pred_values = pred_tensor.cpu().numpy().flatten()
            
            # Inverse transform if scaler was used
            if target_scaler is not None:
                pred_values = target_scaler.inverse_transform(pred_values)
            
            # Get true values
            targets = get_prediction_targets(city_df, fold, cfg['target_col'])
            
            # Create prediction records
            for h in range(1, cfg['horizon'] + 1):
                target_row = targets[targets['horizon'] == h]
                if not target_row.empty:
                    predictions.append({
                        'municipality_id': city_id,
                        'date_origin': fold['origin_date'],
                        'horizon': h,
                        'y_true': target_row['y_true'].iloc[0],
                        'y_pred': pred_values[h-1],
                        'model_name': 'TFT',
                        'fold_id': fold['fold_id'],
                        'train_end_date': fold['train_end_date']
                    })
    
    return pd.DataFrame(predictions)