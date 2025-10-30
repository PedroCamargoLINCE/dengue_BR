"""PyTorch sequence dataset for time series forecasting models."""

import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional


class SeqDataset(torch.utils.data.Dataset):
    """
    PyTorch dataset for sequence-to-sequence time series forecasting.
    
    Creates sliding windows of sequences with proper train/val splitting.
    Ensures sequences do NOT cross train/val/test boundaries.
    """
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Dict, List


class SeqDataset(Dataset):
    """Dataset for sequence models (GRU/TFT).
    
    Builds sliding windows X[L,F] and y[H] for a given city & fold.
    Includes own target and each neighbor target as separate channels.
    Ensures sequences do NOT cross train/val/test boundaries.
    """
    
    def __init__(
        self,
        city_df: pd.DataFrame,
        neighbor_lags_df: pd.DataFrame,
        cfg: dict,
        fold: Dict,
        split: str = 'train',
        target_scaler=None,  # Legacy parameter for compatibility
        feature_transformer=None,  # New parameter for neighbor target-derived features
        general_feature_scaler=None  # New parameter for general features (climate/static/calendar)
    ):
        """Initialize sequence dataset.
        
        Args:
            city_df: City-specific DataFrame with all features
            neighbor_lags_df: Neighbor lag features
            cfg: Configuration dictionary
            fold: Fold information
            split: 'train' or 'val'
            target_scaler: Legacy target scaler (for compatibility)
            feature_transformer: New feature transformer for neighbor features
        """
        self.cfg = cfg
        self.L = cfg['lookback_L']
        self.H = cfg['horizon']
        self.target_col = cfg['target_col']
        self.climate_cols = cfg['climate_cols']
        self.static_cols = cfg['static_cols']
        self.target_scaler = target_scaler  # Legacy support
        self.feature_transformer = feature_transformer  # New transformer for neighbor features
        self.general_feature_scaler = general_feature_scaler  # New scaler for general features
        
        # Filter data for this split
        from .folds import filter_fold_data
        import pandas as pd
        
        if split == 'val':
            # For validation, we need extended data to create sequences with proper lookback
            # Get validation period + enough preceding data for lookback
            all_train_data = filter_fold_data(city_df, fold, split='all_train')
            val_data = filter_fold_data(city_df, fold, split='val')
            
            if len(val_data) == 0:
                # No validation data available
                self.data = pd.DataFrame()
            else:
                # Get the minimum date we need for lookback
                val_start_date = val_data['week_start_date'].min()
                lookback_start_date = val_start_date - pd.Timedelta(weeks=self.L)
                
                # Get extended data that includes lookback period
                extended_mask = all_train_data['week_start_date'] >= lookback_start_date
                extended_data = all_train_data[extended_mask]
                
                # Combine with validation data
                self.data = pd.concat([extended_data, val_data]).drop_duplicates().sort_values('week_start_date')
                
                # Mark validation sequences (targets must be in validation period)
                self.val_start_date = val_start_date
                self.is_validation = True
        else:
            self.data = filter_fold_data(city_df, fold, split=split)
            self.is_validation = False
        
        # Merge neighbor features
        self.data = self.data.merge(neighbor_lags_df, on='week_start_date', how='left')
        
        # Fill NaN values
        self.data = self.data.fillna(0)
        
        # Apply feature transformations if provided
        if self.feature_transformer:
            self.data = self.feature_transformer.transform(self.data)
        
        # Apply general feature scaling if provided
        if self.general_feature_scaler:
            self.data = self.general_feature_scaler.transform(self.data)
        
        # Build feature matrix
        self._build_features()
        
        # Create sequences
        self._create_sequences()
    
    def _build_features(self):
        """Build feature matrix for sequences."""
        import pandas as pd
        import numpy as np
        
        features = []
        
        # Own target history (will be lagged in sequence creation)
        features.append(self.target_col)
        
        # Neighbor target histories (already lagged appropriately)
        neighbor_cols = [col for col in self.data.columns if col.startswith('neighbor_')]
        features.extend(neighbor_cols)
        
        # Climate features (with causal lags)
        for col in self.climate_cols:
            for lag in self.cfg['lags_climate']:
                lag_col = f"{col}_lag{lag}"
                if lag_col in self.data.columns:
                    features.append(lag_col)
        
        # Static/slow features (treated as normal covariates)
        for col in self.static_cols:
            if col in self.data.columns:
                features.append(col)
        
        # Calendar features
        calendar_cols = ['weekofyear_sin', 'weekofyear_cos']
        for col in calendar_cols:
            if col in self.data.columns:
                features.append(col)
        
        self.feature_cols = features
        
        # Ensure all feature columns exist and are numeric
        missing_cols = [col for col in self.feature_cols if col not in self.data.columns]
        if missing_cols:
            print(f"[SeqDataset] Warning: Missing columns {missing_cols}, removing from features")
            self.feature_cols = [col for col in self.feature_cols if col in self.data.columns]
        
        # Convert all feature columns to numeric immediately
        for col in self.feature_cols:
            if col in self.data.columns:
                # Check if column is non-numeric
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    print(f"[SeqDataset] Converting non-numeric column '{col}' to numeric")
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                
                # Fill NaN and convert to float32
                self.data[col] = self.data[col].fillna(0.0).astype(np.float32)
    
    def _create_sequences(self):
        """Create sequences that don't cross boundaries."""
        import numpy as np
        import pandas as pd
        
        self.sequences = []
        
        # Sort data by date
        data_sorted = self.data.sort_values('week_start_date').reset_index(drop=True)
        
        # Ensure target column is also numeric
        if not pd.api.types.is_numeric_dtype(data_sorted[self.target_col]):
            data_sorted[self.target_col] = pd.to_numeric(data_sorted[self.target_col], errors='coerce')
        data_sorted[self.target_col] = data_sorted[self.target_col].fillna(0.0).astype(np.float32)
        
        # Create sliding windows
        for i in range(len(data_sorted) - self.L - self.H + 1):
            # Extract feature window (X) and target window (y)
            subX = data_sorted.iloc[i:i+self.L][self.feature_cols]
            suby = data_sorted.iloc[i+self.L:i+self.L+self.H][self.target_col]
            
            # For validation, only include sequences where targets are in validation period
            if hasattr(self, 'is_validation') and self.is_validation:
                target_start_date = data_sorted.iloc[i+self.L]['week_start_date']
                if target_start_date < self.val_start_date:
                    continue  # Skip sequences with targets before validation period
            
            # Convert to numpy arrays with explicit float32 dtype
            X_data = subX.values.astype(np.float32)
            y_data = suby.values.astype(np.float32)
            
            # Only add sequence if target values are valid (no NaN)
            if not np.isnan(y_data).any() and not np.isnan(X_data).any():
                self.sequences.append((X_data, y_data))
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx):
        import torch
        import numpy as np
        
        X, y = self.sequences[idx]
        
        # Data should already be float32, but ensure it
        X = X.astype(np.float32) if X.dtype != np.float32 else X
        y = y.astype(np.float32) if y.dtype != np.float32 else y
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X)  # [L, F]
        y_tensor = torch.from_numpy(y)  # [H]
        
        # Apply target scaling if available (legacy support or new transformer)
        if hasattr(self.target_scaler, 'transform') and self.target_scaler is not None:
            if hasattr(self.target_scaler, 'method'):
                # New transformer
                y_scaled = self.target_scaler.transform(y_tensor.numpy())
            else:
                # Legacy scaler
                y_scaled = self.target_scaler.transform(y_tensor.numpy().reshape(-1, 1)).flatten()
            y_tensor = torch.from_numpy(y_scaled.astype(np.float32))
        
        return X_tensor, y_tensor
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        if self.sequences:
            return self.sequences[0][0].shape[1]
        return len(self.feature_cols)


def create_dataloaders(
    city_df: pd.DataFrame,
    neighbor_lags_df: pd.DataFrame,
    cfg: dict,
    fold: Dict,
    target_scaler=None,  # Legacy parameter
    feature_transformer=None,  # For neighbor target-derived features
    general_feature_scaler=None,  # For general features (climate/static/calendar)
    batch_size: int = 32
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders.
    
    Args:
        city_df: City-specific DataFrame
        neighbor_lags_df: Neighbor lag features
        cfg: Configuration dictionary
        fold: Fold information
        target_scaler: Legacy target scaler
        feature_transformer: Feature transformer for neighbor features
        batch_size: Batch size for dataloaders
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = SeqDataset(city_df, neighbor_lags_df, cfg, fold, 'train', target_scaler, feature_transformer)
    val_dataset = SeqDataset(city_df, neighbor_lags_df, cfg, fold, 'val', target_scaler, feature_transformer)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        drop_last=False
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        drop_last=False
    )
    
    return train_loader, val_loader