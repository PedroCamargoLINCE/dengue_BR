"""Target and feature transformation classes for forecasting pipeline."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler, RobustScaler
from typing import Dict, Any, Optional, List
import warnings


class TargetTransformer:
    """
    Handle target variable transformations with proper fitting on training data only.
    
    Supports:
    - "none": Identity transformation
    - "maxabs": MaxAbsScaler from scikit-learn
    - "log1p": Natural log(1+x) transformation
    """
    
    def __init__(self, method: str = "none"):
        """
        Initialize transformer.
        
        Args:
            method: Transformation method ("none", "maxabs", "log1p")
        """
        if method not in ["none", "maxabs", "log1p"]:
            raise ValueError(f"Unknown transformation method: {method}")
        
        self.method = method
        self.scaler = None
        self.is_fitted = False
        
        if method == "maxabs":
            self.scaler = MaxAbsScaler()
    
    def fit(self, y_train: np.ndarray) -> 'TargetTransformer':
        """
        Fit transformer on training data only.
        
        Args:
            y_train: Training target values (1D or 2D array)
            
        Returns:
            self
        """
        y_train = np.asarray(y_train)
        
        if self.method == "none":
            # No fitting needed for identity
            pass
        elif self.method == "maxabs":
            # Fit MaxAbsScaler
            if y_train.ndim == 1:
                y_train = y_train.reshape(-1, 1)
            self.scaler.fit(y_train)
        elif self.method == "log1p":
            # Check for negative values
            if np.any(y_train < 0):
                warnings.warn(f"Found negative values in target for log1p transform. Min value: {y_train.min()}")
        
        self.is_fitted = True
        return self
    
    def transform(self, y: np.ndarray) -> np.ndarray:
        """
        Transform target values.
        
        Args:
            y: Target values to transform
            
        Returns:
            Transformed target values
        """
        if not self.is_fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        y = np.asarray(y)
        original_shape = y.shape
        
        if self.method == "none":
            return y
        elif self.method == "maxabs":
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            transformed = self.scaler.transform(y)
            return transformed.reshape(original_shape)
        elif self.method == "log1p":
            # Apply log1p (handles zeros gracefully)
            return np.log1p(np.maximum(y, 0))  # Clamp negative values to 0 before log1p
        
        return y
    
    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform target values back to original space.
        
        Args:
            y_transformed: Transformed target values
            
        Returns:
            Original scale target values
        """
        if not self.is_fitted:
            raise ValueError("Transformer not fitted. Call fit() first.")
        
        y_transformed = np.asarray(y_transformed)
        original_shape = y_transformed.shape
        
        if self.method == "none":
            return y_transformed
        elif self.method == "maxabs":
            if y_transformed.ndim == 1:
                y_transformed = y_transformed.reshape(-1, 1)
            inverse = self.scaler.inverse_transform(y_transformed)
            return inverse.reshape(original_shape)
        elif self.method == "log1p":
            # Apply expm1 (inverse of log1p)
            return np.expm1(y_transformed)
        
        return y_transformed
    
    def get_params(self) -> Dict[str, Any]:
        """Get transformer parameters for saving/loading."""
        params = {"method": self.method, "is_fitted": self.is_fitted}
        
        if self.method == "maxabs" and self.is_fitted:
            params["scaler_scale"] = self.scaler.scale_
            params["scaler_max_abs"] = self.scaler.max_abs_
            
        return params
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set transformer parameters from saved state."""
        self.method = params["method"]
        self.is_fitted = params["is_fitted"]
        
        if self.method == "maxabs" and self.is_fitted:
            self.scaler = MaxAbsScaler()
            self.scaler.scale_ = params["scaler_scale"]
            self.scaler.max_abs_ = params["scaler_max_abs"]


class FeatureTransformer:
    """
    Handle feature transformations, particularly for neighbor target-derived features.
    
    Applies the same TYPE of transformation as the target, but with separate fitting
    for each feature to avoid data leakage.
    """
    
    def __init__(self, feature_transformers: Dict[str, str]):
        """
        Initialize feature transformer.
        
        Args:
            feature_transformers: Dict mapping feature names to transformation methods
                                 e.g., {"neighbor_target_lag1": "maxabs"}
        """
        self.feature_transformers = {}
        
        for feature_name, method in feature_transformers.items():
            self.feature_transformers[feature_name] = TargetTransformer(method)
    
    def fit(self, X_train: pd.DataFrame) -> 'FeatureTransformer':
        """
        Fit transformers on training features only.
        
        Args:
            X_train: Training feature DataFrame
            
        Returns:
            self
        """
        for feature_name, transformer in self.feature_transformers.items():
            if feature_name in X_train.columns:
                feature_values = X_train[feature_name].values
                transformer.fit(feature_values)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features.
        
        Args:
            X: Feature DataFrame to transform
            
        Returns:
            Transformed feature DataFrame
        """
        X_transformed = X.copy()
        
        for feature_name, transformer in self.feature_transformers.items():
            if feature_name in X_transformed.columns:
                original_values = X_transformed[feature_name].values
                transformed_values = transformer.transform(original_values)
                X_transformed[feature_name] = transformed_values
        
        return X_transformed
    
    def get_params(self) -> Dict[str, Dict[str, Any]]:
        """Get all transformer parameters."""
        return {
            feature_name: transformer.get_params()
            for feature_name, transformer in self.feature_transformers.items()
        }
    
    def set_params(self, params: Dict[str, Dict[str, Any]]) -> None:
        """Set all transformer parameters."""
        for feature_name, transformer_params in params.items():
            if feature_name in self.feature_transformers:
                self.feature_transformers[feature_name].set_params(transformer_params)


def create_target_transformer(cfg: Dict[str, Any]) -> TargetTransformer:
    """
    Create target transformer from configuration.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        TargetTransformer instance
    """
    target_scaling = cfg.get('gru', {}).get('target_scaling', 'none')
    return TargetTransformer(method=target_scaling)


def create_feature_transformer(cfg: Dict[str, Any], neighbor_features: list) -> Optional[FeatureTransformer]:
    """
    Create feature transformer for neighbor target-derived features.
    
    Args:
        cfg: Configuration dictionary  
        neighbor_features: List of neighbor feature column names
        
    Returns:
        FeatureTransformer instance or None if no transformation needed
    """
    target_scaling = cfg.get('gru', {}).get('target_scaling', 'none')
    
    if target_scaling == "none":
        return None
    
    # Identify neighbor target-derived features
    neighbor_target_features = {}
    for feature in neighbor_features:
        # Look for features that are derived from neighbor targets (e.g., contain "target" or "morbidity")
        if any(keyword in feature.lower() for keyword in ['target', 'morbidity', 'mortality']):
            neighbor_target_features[feature] = target_scaling
    
    if not neighbor_target_features:
        return None
    
    return FeatureTransformer(neighbor_target_features)


def check_negative_predictions(y_pred: np.ndarray, threshold: float = 0.005) -> Dict[str, Any]:
    """
    Check for negative predictions and return statistics.
    
    Args:
        y_pred: Prediction array
        threshold: Warning threshold for percentage of negative predictions
        
    Returns:
        Dictionary with negative prediction statistics
    """
    y_pred_flat = y_pred.flatten()
    negative_mask = y_pred_flat < 0
    negative_count = negative_mask.sum()
    total_count = len(y_pred_flat)
    negative_pct = negative_count / total_count if total_count > 0 else 0.0
    
    stats = {
        'negative_count': negative_count,
        'total_count': total_count,
        'negative_pct': negative_pct,
        'should_warn': negative_pct > threshold,
        'min_value': y_pred_flat.min(),
        'negative_values': y_pred_flat[negative_mask] if negative_count > 0 else np.array([])
    }
    
    return stats


class GeneralFeatureScaler:
    """
    General feature scaler for non-target features (climate, static, calendar).
    
    Supports multiple scaling methods:
    - "none": No scaling (identity)
    - "standard": StandardScaler (mean=0, std=1) 
    - "minmax": MinMaxScaler (range [0,1])
    - "robust": RobustScaler (median=0, IQR=1)
    """
    
    def __init__(self, method: str = "standard", feature_columns: Optional[List[str]] = None):
        """
        Initialize general feature scaler.
        
        Args:
            method: Scaling method ("none", "standard", "minmax", "robust")
            feature_columns: List of column names to scale. If None, will be determined during fit.
        """
        self.method = method.lower()
        self.feature_columns = feature_columns
        self.scalers = {}
        self.is_fitted = False
        
        if self.method not in ["none", "standard", "minmax", "robust"]:
            raise ValueError(f"Unsupported scaling method: {method}")
    
    def fit(self, X: pd.DataFrame) -> 'GeneralFeatureScaler':
        """
        Fit scalers on training data.
        
        Args:
            X: Training feature DataFrame
            
        Returns:
            self
        """
        if self.method == "none":
            self.is_fitted = True
            return self
        
        # Determine feature columns if not provided
        if self.feature_columns is None:
            self.feature_columns = list(X.columns)
        
        # Create and fit scalers for each feature column
        for col in self.feature_columns:
            if col in X.columns:
                if self.method == "standard":
                    scaler = StandardScaler()
                elif self.method == "minmax":
                    scaler = MinMaxScaler()
                elif self.method == "robust":
                    scaler = RobustScaler()
                
                # Fit on the column data
                column_data = X[col].values.reshape(-1, 1)
                scaler.fit(column_data)
                self.scalers[col] = scaler
        
        self.is_fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using fitted scalers.
        
        Args:
            X: Feature DataFrame to transform
            
        Returns:
            Transformed feature DataFrame
        """
        if self.method == "none":
            return X.copy()
        
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
        
        X_transformed = X.copy()
        
        for col, scaler in self.scalers.items():
            if col in X_transformed.columns:
                column_data = X_transformed[col].values.reshape(-1, 1)
                scaled_data = scaler.transform(column_data)
                X_transformed[col] = scaled_data.flatten()
        
        return X_transformed
    
    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform features back to original scale.
        
        Args:
            X: Transformed feature DataFrame
            
        Returns:
            Original scale feature DataFrame
        """
        if self.method == "none":
            return X.copy()
        
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before inverse_transform")
        
        X_original = X.copy()
        
        for col, scaler in self.scalers.items():
            if col in X_original.columns:
                column_data = X_original[col].values.reshape(-1, 1)
                original_data = scaler.inverse_transform(column_data)
                X_original[col] = original_data.flatten()
        
        return X_original
    
    def get_params(self) -> Dict[str, Any]:
        """Get scaler parameters for serialization."""
        if self.method == "none":
            return {"method": self.method, "feature_columns": self.feature_columns}
        
        params = {
            "method": self.method,
            "feature_columns": self.feature_columns,
            "scalers": {}
        }
        
        for col, scaler in self.scalers.items():
            if hasattr(scaler, 'mean_'):  # StandardScaler
                params["scalers"][col] = {
                    "type": "standard",
                    "mean_": scaler.mean_,
                    "scale_": scaler.scale_
                }
            elif hasattr(scaler, 'data_min_'):  # MinMaxScaler
                params["scalers"][col] = {
                    "type": "minmax", 
                    "data_min_": scaler.data_min_,
                    "data_max_": scaler.data_max_,
                    "data_range_": scaler.data_range_,
                    "scale_": scaler.scale_,
                    "min_": scaler.min_
                }
            elif hasattr(scaler, 'center_'):  # RobustScaler
                params["scalers"][col] = {
                    "type": "robust",
                    "center_": scaler.center_,
                    "scale_": scaler.scale_
                }
        
        return params
    
    def set_params(self, params: Dict[str, Any]) -> None:
        """Set scaler parameters from serialized data."""
        self.method = params["method"]
        self.feature_columns = params["feature_columns"]
        
        if self.method == "none":
            self.is_fitted = True
            return
        
        self.scalers = {}
        
        for col, scaler_params in params["scalers"].items():
            scaler_type = scaler_params["type"]
            
            if scaler_type == "standard":
                scaler = StandardScaler()
                scaler.mean_ = scaler_params["mean_"]
                scaler.scale_ = scaler_params["scale_"]
            elif scaler_type == "minmax":
                scaler = MinMaxScaler()
                scaler.data_min_ = scaler_params["data_min_"]
                scaler.data_max_ = scaler_params["data_max_"]
                scaler.data_range_ = scaler_params["data_range_"]
                scaler.scale_ = scaler_params["scale_"]
                scaler.min_ = scaler_params["min_"]
            elif scaler_type == "robust":
                scaler = RobustScaler()
                scaler.center_ = scaler_params["center_"]
                scaler.scale_ = scaler_params["scale_"]
            
            self.scalers[col] = scaler
        
        self.is_fitted = True


def create_general_feature_scaler(cfg: Dict[str, Any], feature_columns: List[str]) -> Optional[GeneralFeatureScaler]:
    """
    Create general feature scaler from configuration.
    
    Args:
        cfg: Configuration dictionary
        feature_columns: List of feature column names to scale
        
    Returns:
        GeneralFeatureScaler instance or None if no scaling needed
    """
    feature_scaler_method = cfg.get('gru', {}).get('feature_scaler', 'standard')
    
    if feature_scaler_method == "none":
        return None
    
    return GeneralFeatureScaler(method=feature_scaler_method, feature_columns=feature_columns)