"""Fold generation for external evaluation."""

import pandas as pd
from typing import List, Dict, Tuple, Optional

def compute_min_val_weeks(L, H, min_val_windows=8):
    # Garante pelo menos 'min_val_windows' janelas de validação
    # nº_janelas = VAL_WEEKS - L - H + 1 >= min_val_windows
    return L + H - 1 + min_val_windows

def generate_folds(dates: pd.Series, cfg: dict) -> List[Dict]:
    """Generate folds for external evaluation using moving window.
    
    Returns folds for external evaluation:
    - train_span: (start, end) # 364 weeks default
    - val_span: (start, end) or None # internal val (GRU/TFT)
    - origins: list[date] in the last full calendar year
    - fold_id, train_end_date
    
    Args:
        dates: Series of dates (should be sorted)
        cfg: Configuration dictionary
        
    Returns:
        List of fold dictionaries
    """
    dates = pd.to_datetime(dates).sort_values().reset_index(drop=True)
    
    train_window_weeks = cfg['train_window_weeks']
    step_weeks = cfg['step_weeks']
    horizon = cfg['horizon']
    val_weeks = cfg.get('val_weeks', 12)
    L = cfg.get('lookback_L', 104)
    H = cfg.get('horizon', 4)
    val_weeks = cfg.get('val_weeks', 12)

    min_needed = compute_min_val_weeks(L, H, min_val_windows=8)
    if val_weeks < min_needed:
        print(f"[folds] Increasing VAL_WEEKS from {val_weeks} to {min_needed} (L={L}, H={H})")
        val_weeks = min_needed

    # use 'val_weeks' ajustado ao construir 'val_span' dentro da janela de treino

    # Find last full calendar year
    years = dates.dt.year
    year_counts = years.value_counts()
    
    # Get last year with sufficient data
    last_year = None
    for year in sorted(year_counts.index, reverse=True):
        if year_counts[year] >= 50:  # At least 50 weeks
            last_year = year
            break
    
    if last_year is None:
        raise ValueError("No suitable calendar year found for evaluation")
    
    # Get origins in the last full calendar year
    last_year_dates = dates[dates.dt.year == last_year]
    
    folds = []
    
    for i, origin_date in enumerate(last_year_dates[::step_weeks]):
        # Find train start (364 weeks before origin)
        train_start_idx = None
        target_train_start = origin_date - pd.Timedelta(weeks=train_window_weeks)
        
        # Find closest date to target train start
        for idx, date in enumerate(dates):
            if date >= target_train_start:
                train_start_idx = idx
                break
        
        if train_start_idx is None:
            continue  # Not enough history
        
        # Train end is the day before origin
        train_end_date = origin_date - pd.Timedelta(days=1)
        train_start_date = dates.iloc[train_start_idx]
        
        # Check if we have enough training data
        train_span_days = (train_end_date - train_start_date).days
        if train_span_days < (train_window_weeks - 4) * 7:  # Allow some flexibility
            continue
        
        # Internal validation span (for GRU/TFT only)
        val_start_date = train_end_date - pd.Timedelta(weeks=val_weeks)
        val_end_date = train_end_date
        
        # Check if we have future data for prediction
        max_pred_date = origin_date + pd.Timedelta(weeks=horizon)
        if max_pred_date > dates.max():
            continue  # Can't predict far enough into future
        
        fold = {
            'fold_id': i,
            'train_span': (train_start_date, train_end_date),
            'val_span': (val_start_date, val_end_date),
            'origin_date': origin_date,
            'train_end_date': train_end_date,
            'horizons': list(range(1, horizon + 1))
        }
        
        folds.append(fold)
    
    return folds


def filter_fold_data(
    df: pd.DataFrame, 
    fold: Dict, 
    date_col: str = 'week_start_date',
    split: str = 'train'
) -> pd.DataFrame:
    """Filter data for a specific fold and split.
    
    Args:
        df: Full dataset
        fold: Fold dictionary from generate_folds
        date_col: Date column name
        split: 'train', 'val', or 'all_train' (train + val)
        
    Returns:
        Filtered DataFrame
    """
    dates = pd.to_datetime(df[date_col])
    
    if split == 'train':
        train_start, train_end = fold['train_span']
        val_start, val_end = fold['val_span']
        
        # Training data excludes internal validation period
        mask = (dates >= train_start) & (dates < val_start)
        
    elif split == 'val':
        val_start, val_end = fold['val_span']
        mask = (dates >= val_start) & (dates <= val_end)
        
    elif split == 'all_train':
        # All training data including internal validation
        train_start, train_end = fold['train_span']
        mask = (dates >= train_start) & (dates <= train_end)
        
    else:
        raise ValueError(f"Unknown split: {split}")
    
    return df[mask].copy()


def get_prediction_targets(
    df: pd.DataFrame,
    fold: Dict,
    target_col: str,
    date_col: str = 'week_start_date'
) -> pd.DataFrame:
    """Get prediction targets for a fold.
    
    Args:
        df: Full dataset
        fold: Fold dictionary
        target_col: Target column name
        date_col: Date column name
        
    Returns:
        DataFrame with prediction targets for each horizon
    """
    origin_date = fold['origin_date']
    horizons = fold['horizons']
    
    targets = []
    for h in horizons:
        pred_date = origin_date + pd.Timedelta(weeks=h)
        target_row = df[pd.to_datetime(df[date_col]) == pred_date]
        
        if not target_row.empty:
            targets.append({
                'horizon': h,
                'date': pred_date,
                'y_true': target_row[target_col].iloc[0]
            })
    
    return pd.DataFrame(targets)