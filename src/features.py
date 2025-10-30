"""Feature engineering functions with strict causal constraints."""

import pandas as pd
import numpy as np
from typing import List


def make_calendar_features(df: pd.DataFrame, date_col: str = 'week_start_date') -> pd.DataFrame:
    """Create calendar features from date column.
    
    Args:
        df: DataFrame with date column
        date_col: Name of the date column
        
    Returns:
        DataFrame with calendar features
    """
    result = df[[date_col]].copy()
    dates = pd.to_datetime(df[date_col])
    
    # Week of year as sin/cos encoding
    week_of_year = dates.dt.isocalendar().week
    result['weekofyear_sin'] = np.sin(2 * np.pi * week_of_year / 52)
    result['weekofyear_cos'] = np.cos(2 * np.pi * week_of_year / 52)
    
    return result


def make_lagged(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    """Create lagged features for specified columns.
    
    Args:
        df: DataFrame with time series data (must be sorted by time)
        cols: List of column names to lag
        lags: List of lag values (positive integers)
        
    Returns:
        DataFrame with original columns plus lagged versions
    """
    result = df.copy()
    
    for col in cols:
        if col not in df.columns:
            raise ValueError(f"Column {col} not found in DataFrame")
        
        for lag in lags:
            if lag <= 0:
                raise ValueError(f"Lag must be positive, got {lag}")
            result[f"{col}_lag{lag}"] = df[col].shift(lag)
    
    return result


def build_neighbor_lag_matrix(
    df_all: pd.DataFrame, 
    neighbor_map_df: pd.DataFrame, 
    city_id: int, 
    target_col: str, 
    lags: List[int],
    model_type: str = 'tabular'
) -> pd.DataFrame:
    """Build neighbor lag matrix for a specific city.
    
    For city_id, fetch all listed neighbor_cd_mun from neighbor_map_df.
    For tabular models (CatBoost), creates explicit lag columns.
    For sequence models (GRU/TFT), delegates to build_neighbor_raw_history.
    
    Args:
        df_all: Full dataset with all cities
        neighbor_map_df: Neighbor mapping with columns ['cd_mun', 'neighbor_cd_mun']
        city_id: Target city ID
        target_col: Target column name (e.g., 'morbidity_rate')
        lags: List of lag values (only used for tabular models)
        model_type: 'tabular' for explicit lags, 'sequence' for raw history
        
    Returns:
        DataFrame keyed by week_start_date with neighbor features
    """
    # For sequence models, use raw neighbor history instead of explicit lags
    if model_type == 'sequence':
        return build_neighbor_raw_history(df_all, neighbor_map_df, city_id, target_col)
    # Get neighbors for this city
    neighbors = neighbor_map_df[neighbor_map_df['cd_mun'] == city_id]['neighbor_cd_mun'].unique()
    
    if len(neighbors) == 0:
        # No neighbors found, return empty DataFrame with correct structure
        city_dates = df_all[df_all['cd_mun'] == city_id][['week_start_date']].copy()
        return city_dates
    
    # Get city's timeline for alignment
    city_timeline = df_all[df_all['cd_mun'] == city_id][['week_start_date']].copy()
    city_timeline = city_timeline.sort_values('week_start_date').reset_index(drop=True)
    
    neighbor_features = []
    
    for neighbor_id in neighbors:
        # Get neighbor data
        neighbor_data = df_all[df_all['cd_mun'] == neighbor_id][['week_start_date', target_col]].copy()
        neighbor_data = neighbor_data.sort_values('week_start_date').reset_index(drop=True)
        
        if neighbor_data.empty:
            continue
            
        # Create lagged features for this neighbor
        for lag in lags:
            if lag <= 0:
                continue
                
            # Create lag using shift (ensures no leakage)
            neighbor_lagged = neighbor_data.copy()
            neighbor_lagged[f'neighbor_{neighbor_id}_lag{lag}'] = neighbor_lagged[target_col].shift(lag)
            neighbor_lagged = neighbor_lagged[['week_start_date', f'neighbor_{neighbor_id}_lag{lag}']]
            
            neighbor_features.append(neighbor_lagged)
    
    # Merge all neighbor features with city timeline
    result = city_timeline.copy()
    
    for neighbor_feat in neighbor_features:
        result = result.merge(neighbor_feat, on='week_start_date', how='left')
    
    neighbor_cols = [col for col in result.columns if col.startswith('neighbor_')]

    if neighbor_cols:
        result[neighbor_cols] = (
            result[neighbor_cols]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0.0)
            .astype(np.float32)
        )
    # Verify no leakage: for any origin t, all neighbor features use timestamps <= t
    # This is guaranteed by the shift operation above
    
    return result


def build_neighbor_raw_history(
    df_all: pd.DataFrame,
    neighbor_map_df: pd.DataFrame,
    city_id: int,
    target_col: str
) -> pd.DataFrame:
    """Build raw neighbor history for sequence models.
    
    Returns the raw time series data for all neighbors of a given city,
    without explicit lagging (sequence models handle the temporal patterns internally).
    
    Args:
        df_all: Full dataset with all cities
        neighbor_map_df: Neighbor mapping DataFrame
        city_id: Target city ID
        target_col: Target column name (e.g., 'morbidity_rate')
        
    Returns:
        DataFrame with week_start_date and columns for each neighbor's target values
    """
    # Get neighbors for this city
    neighbors = neighbor_map_df[neighbor_map_df['cd_mun'] == city_id]['neighbor_cd_mun'].unique()
    
    if len(neighbors) == 0:
        # No neighbors found, return empty DataFrame with correct structure
        city_dates = df_all[df_all['cd_mun'] == city_id][['week_start_date']].copy()
        return city_dates
    
    # Get city's timeline for alignment
    city_timeline = df_all[df_all['cd_mun'] == city_id][['week_start_date']].copy()
    city_timeline = city_timeline.sort_values('week_start_date').reset_index(drop=True)
    
    # Collect raw neighbor data (no explicit lags)
    result = city_timeline.copy()
    
    for neighbor_id in neighbors:
        # Get neighbor data
        neighbor_data = df_all[df_all['cd_mun'] == neighbor_id][['week_start_date', target_col]].copy()
        neighbor_data = neighbor_data.sort_values('week_start_date').reset_index(drop=True)
        
        if neighbor_data.empty:
            continue
            
        # Use raw values (no explicit lagging - let the sequence model handle temporal patterns)
        neighbor_data = neighbor_data.rename(columns={target_col: f'neighbor_{neighbor_id}_raw'})
        
        # Merge with result
        result = result.merge(neighbor_data, on='week_start_date', how='left')
    
    # Ensure all neighbor columns are numeric
    neighbor_cols = [col for col in result.columns if col.startswith('neighbor_')]
    
    if neighbor_cols:
        result[neighbor_cols] = (
            result[neighbor_cols]
            .apply(pd.to_numeric, errors='coerce')
            .fillna(0.0)
            .astype(np.float32)
        )
    
    return result


def assemble_tabular_features(
    city_df: pd.DataFrame,
    neighbor_lags_df: pd.DataFrame,
    climate_cols: List[str],
    static_cols: List[str],
    calendar_df: pd.DataFrame,
    L: int,
    target_col: str
) -> pd.DataFrame:
    """Assemble tabular features for CatBoost.
    
    Args:
        city_df: City-specific DataFrame
        neighbor_lags_df: Neighbor lag features
        climate_cols: Climate column names
        static_cols: Static feature column names
        calendar_df: Calendar features
        L: Lookback window (not used for tabular, but kept for API consistency)
        target_col: Target column name
        
    Returns:
        DataFrame with all features for tabular modeling
    """
    # Start with city data
    result = city_df.copy()
    
    # Add own target lags
    target_lags = [1, 2, 3, 4, 8, 12, 16, 24, 52, 104]  # From config
    for lag in target_lags:
        result[f"{target_col}_lag{lag}"] = result[target_col].shift(lag)
    
    # Add climate lags
    climate_lags = [1, 2, 4, 8]  # From config
    for col in climate_cols:
        if col in result.columns:
            for lag in climate_lags:
                result[f"{col}_lag{lag}"] = result[col].shift(lag)
    
    # Merge neighbor features
    result = result.merge(neighbor_lags_df, on='week_start_date', how='left')
    
    # Merge calendar features
    result = result.merge(calendar_df, on='week_start_date', how='left')
    
    # Keep static columns as-is (they're already in city_df)
    
    # Ensure all numeric data types
    feature_cols = [col for col in result.columns if col != 'week_start_date']
    for col in feature_cols:
        if col in result.columns:
            result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0.0).astype(np.float32)
    
    return result