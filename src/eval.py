"""Evaluation and metrics functions."""

import pandas as pd
import numpy as np
from typing import Dict, List
import os


def unify_predictions(rows_list: List[pd.DataFrame]) -> pd.DataFrame:
    """Unify predictions from multiple models/folds.
    
    Args:
        rows_list: List of prediction DataFrames
        
    Returns:
        Unified predictions DataFrame with schema:
        [municipality_id, date_origin, horizon, y_true, y_pred, model_name, fold_id, train_end_date]
    """
    if not rows_list:
        return pd.DataFrame(columns=[
            'municipality_id', 'date_origin', 'horizon', 'y_true', 
            'y_pred', 'model_name', 'fold_id', 'train_end_date'
        ])
    
    # Concatenate all predictions
    unified = pd.concat(rows_list, ignore_index=True)
    
    # Ensure proper column order and types
    expected_cols = [
        'municipality_id', 'date_origin', 'horizon', 'y_true', 
        'y_pred', 'model_name', 'fold_id', 'train_end_date'
    ]
    
    # Check that all expected columns exist
    missing_cols = [col for col in expected_cols if col not in unified.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in predictions: {missing_cols}")
    
    # Reorder columns
    unified = unified[expected_cols].copy()
    
    # Ensure proper types
    unified['municipality_id'] = unified['municipality_id'].astype(int)
    unified['horizon'] = unified['horizon'].astype(int)
    unified['fold_id'] = unified['fold_id'].astype(int)
    unified['date_origin'] = pd.to_datetime(unified['date_origin'])
    unified['train_end_date'] = pd.to_datetime(unified['train_end_date'])
    unified['y_true'] = unified['y_true'].astype(float)
    unified['y_pred'] = unified['y_pred'].astype(float)
    
    return unified


def compute_metrics(pred_df: pd.DataFrame) -> Dict:
    """Compute evaluation metrics on predictions in ORIGINAL target space.
    
    Args:
        pred_df: Predictions DataFrame from unify_predictions (should be in original scale)
        
    Returns:
        Dictionary of metrics by horizon and aggregated
    """
    if pred_df.empty:
        return {}
    
    metrics = {}
    
    # Calculate adaptive epsilon for sMAPE using positive targets
    positive_targets = pred_df[pred_df['y_true'] > 0]['y_true'].values
    if len(positive_targets) > 0:
        adaptive_epsilon = 1e-3 * np.median(positive_targets)
    else:
        adaptive_epsilon = 1e-8  # fallback
    
    # Compute metrics per horizon
    for horizon in sorted(pred_df['horizon'].unique()):
        horizon_data = pred_df[pred_df['horizon'] == horizon]
        
        if len(horizon_data) == 0:
            continue
        
        y_true = horizon_data['y_true'].values
        y_pred = horizon_data['y_pred'].values
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]
        
        if len(y_true) == 0:
            continue
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        
        denom = np.maximum(adaptive_epsilon, np.abs(y_true) + np.abs(y_pred))
        smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / denom)
        
        mape = 100 * np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8)))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float('nan') if ss_tot == 0 else 1.0 - ss_res / ss_tot
        
        metrics[f'horizon_{horizon}'] = {
            'MAE': mae,
            'RMSE': rmse,
            'sMAPE': smape,
            'MAPE': mape,
            'R2': r2,
            'count': len(y_true)
        }
    
    # Compute aggregated metrics
    if len(pred_df) > 0:
        y_true_all = pred_df['y_true'].values
        y_pred_all = pred_df['y_pred'].values
        
        valid_mask = np.isfinite(y_true_all) & np.isfinite(y_pred_all)
        y_true_all = y_true_all[valid_mask]
        y_pred_all = y_pred_all[valid_mask]
        
        if len(y_true_all) > 0:
            mae_agg = np.mean(np.abs(y_true_all - y_pred_all))
            rmse_agg = np.sqrt(np.mean((y_true_all - y_pred_all) ** 2))
            
            denom_agg = np.maximum(adaptive_epsilon, np.abs(y_true_all) + np.abs(y_pred_all))
            smape_agg = 100 * np.mean(2 * np.abs(y_true_all - y_pred_all) / denom_agg)
            
            mape_agg = 100 * np.mean(np.abs((y_true_all - y_pred_all) / (np.abs(y_true_all) + 1e-8)))

            ss_res = np.sum((y_true_all - y_pred_all) ** 2)
            ss_tot = np.sum((y_true_all - np.mean(y_true_all)) ** 2)
            r2_agg = float('nan') if ss_tot == 0 else 1.0 - ss_res / ss_tot
            
            metrics['aggregated'] = {
                'MAE': mae_agg,
                'RMSE': rmse_agg,
                'sMAPE': smape_agg,
                'MAPE': mape_agg,
                'R2': r2_agg,
                'count': len(y_true_all),
                'adaptive_epsilon': adaptive_epsilon
            }
    
    return metrics


def write_report(pred_df: pd.DataFrame, metrics: Dict, out_dir: str, city_id: int = None) -> str:
    """Write evaluation report in Markdown format.
    
    Args:
        pred_df: Predictions DataFrame
        metrics: Metrics dictionary from compute_metrics
        out_dir: Output directory
        city_id: Optional city ID for report filename
        
    Returns:
        Path to written report file
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if city_id is not None:
        report_path = os.path.join(out_dir, f'evaluation_report_city_{city_id}.md')
    else:
        report_path = os.path.join(out_dir, 'evaluation_report_overall.md')
    
    with open(report_path, 'w') as f:
        f.write("# Dengue Forecasting Evaluation Report\n\n")
        
        if city_id is not None:
            f.write(f"**Municipality ID:** {city_id}\n\n")
        
        f.write(f"**Report Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary statistics
        if not pred_df.empty:
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Total Predictions:** {len(pred_df)}\n")
            f.write(f"- **Models:** {', '.join(pred_df['model_name'].unique())}\n")
            f.write(f"- **Horizons:** {', '.join(map(str, sorted(pred_df['horizon'].unique())))}\n")
            f.write(f"- **Folds:** {pred_df['fold_id'].nunique()}\n\n")
        
        # Metrics by horizon
        f.write("## Metrics by Horizon\n\n")
        f.write("| Horizon | MAE | RMSE | sMAPE | MAPE | R2 | Count |\n")
        f.write("|---------|-----|------|-------|------|----|-------|\n")
        
        for horizon in sorted([k for k in metrics.keys() if k.startswith('horizon_')]):
            h_num = horizon.split('_')[1]
            h_metrics = metrics[horizon]
            f.write(f"| {h_num} | {h_metrics['MAE']:.4f} | {h_metrics['RMSE']:.4f} | "
                   f"{h_metrics['sMAPE']:.2f}% | {h_metrics['MAPE']:.2f}% | "
                   f"{h_metrics['R2']:.4f} | {h_metrics['count']} |\n")
        
        # Aggregated metrics
        if 'aggregated' in metrics:
            agg = metrics['aggregated']
            f.write(f"| **Overall** | **{agg['MAE']:.4f}** | **{agg['RMSE']:.4f}** | "
                   f"**{agg['sMAPE']:.2f}%** | **{agg['MAPE']:.2f}%** | "
                   f"**{agg['R2']:.4f}** | **{agg['count']}** |\n")
        
        f.write("\n")
        
        # Model comparison if multiple models
        if not pred_df.empty and pred_df['model_name'].nunique() > 1:
            f.write("## Model Comparison\n\n")
            f.write("| Model | MAE | RMSE | sMAPE | Count |\n")
            f.write("|-------|-----|------|-------|-------|\n")
            
            for model in sorted(pred_df['model_name'].unique()):
                model_data = pred_df[pred_df['model_name'] == model]
                y_true = model_data['y_true'].values
                y_pred = model_data['y_pred'].values
                
                valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
                if valid_mask.sum() > 0:
                    y_true = y_true[valid_mask]
                    y_pred = y_pred[valid_mask]
                    
                    mae = np.mean(np.abs(y_true - y_pred))
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    smape = 100 * np.mean(2 * np.abs(y_true - y_pred) / 
                                         (np.abs(y_true) + np.abs(y_pred) + 1e-8))
                    
                    f.write(f"| {model} | {mae:.4f} | {rmse:.4f} | {smape:.2f}% | {len(y_true)} |\n")
        
        f.write("\n")
        f.write("---\n")
        f.write("*Generated by dengue forecasting system*\n")
    
    return report_path
