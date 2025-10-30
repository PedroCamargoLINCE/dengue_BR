"""GRU training functions with target transformation support."""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Dict, Optional
import warnings
import time
from tqdm import tqdm

from .models_gru import GRUForecast
from .datasets import create_dataloaders
from .utils import seed_everything, device, clear_gpu_memory
from .transformers import (
    create_target_transformer,
    create_feature_transformer,
    create_general_feature_scaler,
    check_negative_predictions
)
def train_one_fold_gru(
    city_id: int,
    fold: Dict,
    city_df: pd.DataFrame,
    neighbor_lags_df: pd.DataFrame,
    cfg: dict,
    target_scaler=None  # Legacy parameter for compatibility
) -> pd.DataFrame:
    """Train GRU model for one fold and return predictions.
    
    Args:
        city_id: Municipality ID
        fold: Fold information
        city_df: City-specific DataFrame
        neighbor_lags_df: Neighbor lag features
        cfg: Configuration dictionary
        target_scaler: Legacy parameter (ignored - will use target_transformer)
        
    Returns:
        DataFrame with predictions in original scale
    """
    # Set seed for reproducibility
    seed_everything(cfg['seed'] + fold['fold_id'])
    
    # Create target transformer
    target_transformer = create_target_transformer(cfg)
    target_scaling_method = cfg.get('gru', {}).get('target_scaling', 'none')
    
    print(f"🎯 Target scaling for fold {fold['fold_id']}: {target_scaling_method}")
    
    # Create feature transformer for neighbor target-derived features
    neighbor_cols = [col for col in neighbor_lags_df.columns if col.startswith('neighbor_')]
    feature_transformer = create_feature_transformer(cfg, neighbor_cols)
    
    if feature_transformer:
        print(f"🔧 Feature transformation enabled for {len(feature_transformer.feature_transformers)} neighbor features")
    
    # Create general feature scaler for climate/static/calendar features
    # Get all feature columns that will be used (excluding target and neighbor target-derived features)
    climate_lagged_cols = [f"{col}_lag{lag}" for col in cfg['climate_cols'] for lag in cfg['lags_climate']]
    static_cols = cfg['static_cols']
    calendar_cols = cfg['calendar_features']
    general_feature_cols = climate_lagged_cols + static_cols + calendar_cols
    
    general_feature_scaler = create_general_feature_scaler(cfg, general_feature_cols)
    feature_scaler_method = cfg.get('gru', {}).get('feature_scaler', 'standard')
    
    if general_feature_scaler:
        print(f"⚖️ General feature scaling enabled: {feature_scaler_method} for {len(general_feature_cols)} feature columns")
    else:
        print(f"⚖️ General feature scaling disabled (method: {feature_scaler_method})")
    
    # Get training data for fitting transformers
    from .folds import filter_fold_data
    train_data = filter_fold_data(city_df, fold, split='train')
    
    # Fit target transformer on training targets only
    train_targets = train_data[cfg['target_col']].values
    target_transformer.fit(train_targets)
    
    # Fit feature transformer on training features only
    if feature_transformer:
        train_features_with_neighbors = train_data.merge(neighbor_lags_df, on='week_start_date', how='left').fillna(0)
        feature_transformer.fit(train_features_with_neighbors)
    
    # Fit general feature scaler on training features only
    if general_feature_scaler:
        train_features_with_neighbors = train_data.merge(neighbor_lags_df, on='week_start_date', how='left').fillna(0)
        general_feature_scaler.fit(train_features_with_neighbors)
    
    # Create dataloaders with transformers
    if cfg['gru'].get('use_validation', True):
        # Standard train/val split
        train_loader, val_loader = create_dataloaders(
            city_df, neighbor_lags_df, cfg, fold, target_transformer, 
            feature_transformer=feature_transformer, general_feature_scaler=general_feature_scaler,
            batch_size=32
        )
        use_validation = True
    else:
        # Train-only mode: use all training data (including validation period)
        from .datasets import SeqDataset
        all_train_dataset = SeqDataset(
            city_df, neighbor_lags_df, cfg, fold, 'all_train', 
            target_transformer, feature_transformer=feature_transformer,
            general_feature_scaler=general_feature_scaler
        )
        train_loader = torch.utils.data.DataLoader(
            all_train_dataset, batch_size=32, shuffle=True, drop_last=False
        )
        val_loader = None
        use_validation = False
        print(f"🚫 Validation disabled - using all training data ({len(all_train_dataset)} sequences)")
    
    if len(train_loader) == 0:
        raise ValueError(f"No training data for city {city_id}, fold {fold['fold_id']}")
    
    # Handle validation warnings only if using validation
    if use_validation:
        if len(val_loader) == 0:
            print(f"Warning: No validation data for city {city_id}, fold {fold['fold_id']} - using train data for validation")
            val_loader = train_loader
        elif len(val_loader) < 2:
            print(f"Warning: Very few validation batches ({len(val_loader)}) for city {city_id}")
    
    # Get feature dimension
    for batch in train_loader:
        input_size = batch[0].shape[-1]
        break
    
    # Initialize model
    model = GRUForecast(
        input_size=input_size,
        hidden_size=cfg['gru']['hidden_size'],
        num_layers=cfg['gru']['num_layers'],
        dropout=cfg['gru']['dropout'],
        horizon=cfg['horizon'],
        output_activation=cfg['gru'].get('output_activation', 'softplus'),  # Default to softplus for backward compatibility
        min_output=cfg['gru'].get('min_output', 0.0),
        max_output=cfg['gru'].get('max_output', None)
    ).to(device())
    
    # Loss and optimizer with better setup
    if cfg['gru']['loss'] == 'mae':
        criterion = nn.L1Loss()
    elif cfg['gru']['loss'] == 'huber':
        huber_beta = cfg['gru'].get('huber_beta', 0.03)
        criterion = nn.SmoothL1Loss(beta=huber_beta)
    else:
        criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['gru']['lr'],
        weight_decay=cfg['gru']['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler with warmup support
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=cfg['gru']['rlrop']['factor'],
        patience=cfg['gru']['rlrop']['patience'],
        cooldown=cfg['gru']['rlrop']['cooldown'],
        min_lr=1e-7  # Prevent learning rate from getting too small
    )
    
    # Early stopping
    best_val_loss = float('inf')  # For checkpointing based on val_loss
    best_early_stop_metric = float('inf')  # For early stopping based on val_MAE
    patience_counter = 0
    patience = cfg['gru']['early_stopping']['patience']
    min_delta = cfg['gru']['early_stopping']['min_delta']
    
    # Training history tracking
    history = {
        'fold_id': fold['fold_id'],
        'train_losses': [],
        'val_losses': [],
        'val_maes': []
    }
    
    # Clean training setup display
    print(f"\n� Training Fold {fold['fold_id']} - City {city_id}")
    print(f"📊 Data: {len(train_loader)} train batches" + (f", {len(val_loader)} val batches" if use_validation else ", no validation"))
    print(f"🎯 Config: {cfg['gru']['loss']} loss, lr={cfg['gru']['lr']:.0e}, patience={patience}")
    
    # Single interactive progress bar for the entire fold
    progress_bar = tqdm(
        total=cfg['gru']['max_epochs'],
        desc=f"Fold {fold['fold_id']}",
        bar_format="{l_bar}{bar:30}| {n:3d}/{total:3d} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
        leave=True,
        ncols=120
    )
    
    # Training timing
    training_start_time = time.time()
    
    for epoch in range(cfg['gru']['max_epochs']):
        # Training phase - no individual batch progress bars
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
            if cfg['gru']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gru']['grad_clip'])
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Record training loss
        history['train_losses'].append(train_loss)
        
        # Quick negative prediction check (occasionally, lightweight)
        neg_pct = 0.0
        if epoch % 10 == 0 or epoch < 5:
            model.eval()
            with torch.no_grad():
                # Check just the first batch for speed
                sample_batch = next(iter(train_loader))
                batch_x = sample_batch[0].to(device())
                raw_outputs = model(batch_x, return_raw_output=True)
                neg_pct = (raw_outputs < 0).float().mean().item() * 100
        
        # Validation processing (clean, no progress bars)
        if use_validation:
            model.eval()
            val_loss = 0.0
            val_mae_original = 0.0
            val_count = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x = batch_x.to(device())
                    batch_y = batch_y.to(device())
                    
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    # Convert to original scale for MAE calculation
                    if target_transformer and hasattr(target_transformer, 'inverse_transform'):
                        y_original = target_transformer.inverse_transform(batch_y.cpu().numpy())
                        pred_original = target_transformer.inverse_transform(outputs.cpu().numpy())
                    else:
                        y_original = batch_y.cpu().numpy()
                        pred_original = outputs.cpu().numpy()
                    
                    # Calculate MAE in original scale
                    batch_mae = np.mean(np.abs(y_original - pred_original))
                    val_mae_original += batch_mae * len(batch_y)
                    val_count += len(batch_y)
            
            val_loss /= len(val_loader)
            val_mae_original /= val_count
            
            # Record validation metrics
            history['val_losses'].append(val_loss)
            history['val_maes'].append(val_mae_original)
            
            monitor_loss = val_mae_original
            scheduler.step(val_loss)
            
        else:
            # Train-only mode: monitor train loss
            monitor_loss = train_loss
            scheduler.step(train_loss)
        
        # Update progress bar with current metrics
        current_lr = optimizer.param_groups[0]['lr']
        postfix_dict = {
            'loss': f'{train_loss:.4f}',
            'lr': f'{current_lr:.0e}'
        }
        
        if use_validation:
            postfix_dict.update({
                'val_loss': f'{val_loss:.4f}',
                'val_mae': f'{val_mae_original:.3f}'
            })
            
            # Add negative percentage occasionally
            if neg_pct > 0:
                postfix_dict['neg%'] = f'{neg_pct:.0f}'
        
        progress_bar.set_postfix(postfix_dict, refresh=True)
        
        # Early stopping and checkpoint logic - USE SAME METRIC FOR BOTH
        # This fixes the inconsistency that was hurting training
        if use_validation:
            # Use val_MAE for both checkpointing AND early stopping for consistency
            checkpoint_metric = val_mae_original  # Changed from val_loss to val_mae_original
            early_stop_metric = val_mae_original
        else:
            checkpoint_metric = train_loss
            early_stop_metric = train_loss
        
        # Save best checkpoint
        if checkpoint_metric < best_val_loss - min_delta:
            best_val_loss = checkpoint_metric
            best_state = model.state_dict().copy()
        
        # Early stopping logic
        if early_stop_metric < best_early_stop_metric - min_delta:
            best_early_stop_metric = early_stop_metric
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Update progress bar and check early stopping
        progress_bar.update(1)
        
        if patience_counter >= patience:
            # Update progress bar with early stopping info
            final_postfix = postfix_dict.copy()
            final_postfix['stopped'] = f'epoch {epoch+1-patience}'
            progress_bar.set_postfix(final_postfix)
            break
    
    # Close progress bar and show completion message
    progress_bar.close()
    
    final_msg = f"✓ Completed {epoch+1} epochs"
    if use_validation:
        final_msg += f" | Best val_MAE: {best_early_stop_metric:.4f}"
    print(final_msg)
    
    # Load best model
    model.load_state_dict(best_state)
    
    # Generate predictions for this fold (returns in original scale)
    predictions = _generate_gru_predictions(
        model, city_df, neighbor_lags_df, fold, cfg, 
        target_transformer, feature_transformer, general_feature_scaler, city_id
    )
    
    # Clear GPU memory
    clear_gpu_memory()
    
    return predictions, history


def _generate_gru_predictions(
    model: GRUForecast,
    city_df: pd.DataFrame,
    neighbor_lags_df: pd.DataFrame,
    fold: Dict,
    cfg: dict,
    target_transformer,
    feature_transformer,
    general_feature_scaler,
    city_id: int
) -> pd.DataFrame:
    """Generate predictions for fold origins and return in original scale."""
    import numpy as np
    from .datasets import SeqDataset
    from .folds import get_prediction_targets
    
    model.eval()
    predictions = []
    
    # Get all training data (including validation)
    from .folds import filter_fold_data
    train_data = filter_fold_data(city_df, fold, split='all_train')
    train_data = train_data.merge(neighbor_lags_df, on='week_start_date', how='left')
    train_data = train_data.fillna(0)
    train_data = train_data.sort_values('week_start_date').reset_index(drop=True)
    
    # Apply feature transformations to match training
    if feature_transformer:
        train_data = feature_transformer.transform(train_data)
    
    # Apply general feature scaling to match training
    if general_feature_scaler:
        train_data = general_feature_scaler.transform(train_data)
    
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
    
    # Add static features (only if they exist)
    for col in cfg['static_cols']:
        if col in train_data.columns:
            feature_cols.append(col)
    
    # Add calendar features
    calendar_cols = ['weekofyear_sin', 'weekofyear_cos']
    for col in calendar_cols:
        if col in train_data.columns:
            feature_cols.append(col)
    
    # Ensure all feature columns are numeric
    for col in feature_cols:
        if col in train_data.columns:
            train_data[col] = pd.to_numeric(train_data[col], errors='coerce').fillna(0.0).astype(np.float32)
    
    # Transform target column for input sequence
    if target_transformer.method != "none":
        train_data[cfg['target_col']] = target_transformer.transform(train_data[cfg['target_col']].values)
    
    # Create input sequence from last L weeks of training data
    if len(train_data) >= cfg['lookback_L']:
        input_data = train_data.iloc[-cfg['lookback_L']:][feature_cols].values.astype(np.float32)
        
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(device())
            pred_tensor = model(input_tensor)
            pred_values = pred_tensor.cpu().numpy().flatten().astype(np.float32)
            
            # Check for negative predictions before inverse transform
            neg_stats = check_negative_predictions(pred_values)
            if neg_stats['should_warn']:
                print(f"⚠️ WARNING: {neg_stats['negative_pct']*100:.2f}% negative predictions before clamp (fold {fold['fold_id']})")
                print(f"   Min value: {neg_stats['min_value']:.6f}")
            
            # Apply safety clamp in transformed space (before inverse transform)
            pred_values_clamped = np.maximum(pred_values, 0)
            
            # Inverse transform to original scale
            pred_values_original = target_transformer.inverse_transform(pred_values_clamped)
            
            # Apply final safety clamp in original space
            pred_values_original = np.maximum(pred_values_original, 0)
            
            # Get true values (always in original scale)
            targets = get_prediction_targets(city_df, fold, cfg['target_col'])
            
            # Create prediction records (all in original scale)
            for h in range(1, cfg['horizon'] + 1):
                target_row = targets[targets['horizon'] == h]
                if not target_row.empty:
                    predictions.append({
                        'municipality_id': city_id,
                        'date_origin': fold['origin_date'],
                        'horizon': h,
                        'y_true': float(target_row['y_true'].iloc[0]),
                        'y_pred': float(pred_values_original[h-1]),
                        'model_name': 'GRU',
                        'fold_id': fold['fold_id'],
                        'train_end_date': fold['train_end_date']
                    })
            
            # Log negative predictions summary for monitoring
            final_neg_stats = check_negative_predictions(pred_values_original)
            if final_neg_stats['negative_count'] > 0:
                print(f"📊 Final predictions: {final_neg_stats['negative_count']}/{final_neg_stats['total_count']} negative values clamped to 0")
    
    return pd.DataFrame(predictions)