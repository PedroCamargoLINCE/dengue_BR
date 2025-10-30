"""Scaling utilities for models."""

import numpy as np
import pandas as pd
from typing import Dict, Any


class TargetScalerNone:
    """No-op target scaler."""
    
    def fit(self, y):
        """Fit scaler (no-op)."""
        return self
    
    def transform(self, y):
        """Transform targets (no-op)."""
        return y
    
    def inverse_transform(self, y):
        """Inverse transform targets (no-op)."""
        return y


class TargetScalerMaxAbs:
    """MaxAbs target scaler - scales by maximum absolute value."""
    
    def __init__(self):
        self.scale_ = 1.0
    
    def fit(self, y):
        """Fit scaler on training data only."""
        y_array = np.array(y)
        max_abs = np.max(np.abs(y_array))
        self.scale_ = max_abs if max_abs > 0 else 1.0
        return self
    
    def transform(self, y):
        """Transform targets."""
        return np.array(y) / self.scale_
    
    def inverse_transform(self, y):
        """Inverse transform targets."""
        return np.array(y) * self.scale_


def fit_input_scalers(X_train: pd.DataFrame) -> Dict[str, Any]:
    """Fit input scalers on training data.
    
    Args:
        X_train: Training features DataFrame
        
    Returns:
        Dictionary of fitted scalers
    """
    from sklearn.preprocessing import StandardScaler
    
    scalers = {}
    
    # Identify numeric columns (exclude categorical/ID columns)
    numeric_cols = []
    for col in X_train.columns:
        if X_train[col].dtype in ['int64', 'float64']:
            # Skip ID columns and binary features
            if not col.startswith('cd_mun') and X_train[col].nunique() > 2:
                numeric_cols.append(col)
    
    if numeric_cols:
        scaler = StandardScaler()
        scaler.fit(X_train[numeric_cols])
        scalers['standard'] = {
            'scaler': scaler,
            'columns': numeric_cols
        }
    
    return scalers


def apply_input_scalers(X: pd.DataFrame, scalers: Dict[str, Any]) -> pd.DataFrame:
    """Apply fitted input scalers to features.
    
    Args:
        X: Features DataFrame
        scalers: Dictionary of fitted scalers
        
    Returns:
        Scaled features DataFrame
    """
    X_scaled = X.copy()
    
    if 'standard' in scalers:
        scaler_info = scalers['standard']
        scaler = scaler_info['scaler']
        columns = scaler_info['columns']
        
        # Only scale columns that exist in current DataFrame
        available_cols = [col for col in columns if col in X.columns]
        if available_cols:
            X_scaled[available_cols] = scaler.transform(X[available_cols])
    
    return X_scaled