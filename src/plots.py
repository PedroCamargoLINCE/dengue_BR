"""Plotting and visualization functions."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Optional


def _limit_figsize(figsize, max_pixels_per_dim=1999, dpi=150):
    """Limit figure size to respect pixel constraints.
    
    Args:
        figsize: Tuple of (width, height) in inches
        max_pixels_per_dim: Maximum pixels allowed in any single dimension (e.g., 1999 allows up to 1999x1999)
        dpi: Dots per inch for calculation (should match the DPI used when saving)
        
    Returns:
        Adjusted figsize tuple
    """
    width_inches, height_inches = figsize
    width_pixels = width_inches * dpi
    height_pixels = height_inches * dpi
    
    # Calculate max inches per dimension
    max_inches_per_dim = max_pixels_per_dim / dpi
    
    # Limit each dimension independently
    width_inches = min(width_inches, max_inches_per_dim)
    height_inches = min(height_inches, max_inches_per_dim)
    
    return (width_inches, height_inches)


def plot_forecast_vs_actual(pred_df: pd.DataFrame, city_id: int, out_dir: str) -> str:
    """Plot forecast vs actual values by horizon.
    
    Args:
        pred_df: Predictions DataFrame
        city_id: Municipality ID
        out_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        
        # Filter for this city
        city_data = pred_df[pred_df['municipality_id'] == city_id].copy()
        
        if city_data.empty:
            print(f"No data found for city {city_id}")
            return None
        
        # Ensure datetime conversion is safe
        if 'date_origin' in city_data.columns:
            city_data['date_origin'] = pd.to_datetime(city_data['date_origin'], errors='coerce')
            city_data = city_data.dropna(subset=['date_origin'])
        
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create subplots for each horizon
        horizons = sorted(city_data['horizon'].unique())
        n_horizons = len(horizons)
        
        # Use dynamic subplot layout
        if n_horizons == 1:
            fig, axes = plt.subplots(1, 1, figsize=_limit_figsize((8, 6)))
            axes = [axes]
        elif n_horizons == 2:
            fig, axes = plt.subplots(1, 2, figsize=_limit_figsize((12, 6)))
        elif n_horizons <= 4:
            fig, axes = plt.subplots(2, 2, figsize=_limit_figsize((12, 10)))
            axes = axes.flatten()
        else:
            # For more than 4 horizons, use a grid layout
            rows = int(np.ceil(n_horizons / 3))
            fig, axes = plt.subplots(rows, 3, figsize=_limit_figsize((15, 5*rows)))
            axes = axes.flatten() if rows > 1 else [axes]
        
        for i, horizon in enumerate(horizons):
            if i >= len(axes):
                break
                
            horizon_data = city_data[city_data['horizon'] == horizon].copy()
            
            if not horizon_data.empty:
                # Sort by date for proper time series plotting
                horizon_data = horizon_data.sort_values('date_origin')
                
                ax = axes[i]
                
                # Convert to numpy arrays to avoid pandas plotting issues
                dates = horizon_data['date_origin'].values
                y_true = horizon_data['y_true'].values
                y_pred = horizon_data['y_pred'].values
                
                # Check for valid data
                valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
                dates = dates[valid_mask]
                y_true = y_true[valid_mask]
                y_pred = y_pred[valid_mask]
                
                if len(dates) > 0:
                    ax.plot(dates, y_true, 'o-', label='Actual', alpha=0.7, markersize=3, linewidth=1)
                    ax.plot(dates, y_pred, 's-', label='Predicted', alpha=0.7, markersize=3, linewidth=1)
                    
                    ax.set_title(f'Horizon {horizon} weeks (n={len(dates)})')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Target Value')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Improve date formatting
                    if len(dates) > 20:
                        # Show fewer ticks for clarity
                        ax.locator_params(axis='x', nbins=6)
                    
                    # Rotate x-axis labels
                    for tick in ax.get_xticklabels():
                        tick.set_rotation(45)
                        tick.set_ha('right')
        
        # Remove empty subplots
        for i in range(len(horizons), len(axes)):
            if i < len(axes):
                fig.delaxes(axes[i])
        
        plt.tight_layout()
        
        plot_path = os.path.join(out_dir, f'forecast_vs_actual_city_{city_id}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error creating forecast plot: {e}")
        plt.close('all')
        return None


def plot_errors_by_horizon(pred_df: pd.DataFrame, out_dir: str) -> str:
    """Plot error metrics by horizon.
    
    Args:
        pred_df: Predictions DataFrame
        out_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        
        if pred_df.empty:
            print("No data for error plot")
            return None
        
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Compute errors by horizon with proper error handling
        horizon_errors = []
        horizons = sorted(pred_df['horizon'].unique())
        
        for horizon in horizons:
            horizon_data = pred_df[pred_df['horizon'] == horizon].copy()
            
            if not horizon_data.empty:
                # Remove any NaN values
                valid_mask = ~(horizon_data['y_true'].isna() | horizon_data['y_pred'].isna())
                horizon_data = horizon_data[valid_mask]
                
                if not horizon_data.empty:
                    mae = np.mean(np.abs(horizon_data['y_pred'] - horizon_data['y_true']))
                    rmse = np.sqrt(np.mean((horizon_data['y_pred'] - horizon_data['y_true'])**2))
                    
                    # Use robust sMAPE calculation
                    smape = np.mean(np.abs(horizon_data['y_true'] - horizon_data['y_pred']) / 
                                   (np.abs(horizon_data['y_true']) + np.abs(horizon_data['y_pred']) + 1e-8)) * 200
                    
                    horizon_errors.append({
                        'horizon': horizon,
                        'MAE': mae,
                        'RMSE': rmse,
                        'sMAPE': smape,
                        'count': len(horizon_data)
                    })
        
        if not horizon_errors:
            print("No valid data for error calculation")
            return None
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=_limit_figsize((15, 5)))
        
        # Extract data for plotting
        h_values = [h['horizon'] for h in horizon_errors]
        mae_values = [h['MAE'] for h in horizon_errors]
        rmse_values = [h['RMSE'] for h in horizon_errors]
        smape_values = [h['sMAPE'] for h in horizon_errors]
        
        # Plot MAE
        axes[0].bar(h_values, mae_values, color='skyblue', alpha=0.7)
        axes[0].set_xlabel('Horizon')
        axes[0].set_ylabel('MAE')
        axes[0].set_title('Mean Absolute Error by Horizon')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels
        for h, mae in zip(h_values, mae_values):
            axes[0].text(h, mae + mae*0.01, f'{mae:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot RMSE
        axes[1].bar(h_values, rmse_values, color='lightcoral', alpha=0.7)
        axes[1].set_xlabel('Horizon')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Root Mean Squared Error by Horizon')
        axes[1].grid(True, alpha=0.3)
        
        # Add value labels
        for h, rmse in zip(h_values, rmse_values):
            axes[1].text(h, rmse + rmse*0.01, f'{rmse:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Plot sMAPE
        axes[2].bar(h_values, smape_values, color='lightgreen', alpha=0.7)
        axes[2].set_xlabel('Horizon')
        axes[2].set_ylabel('sMAPE (%)')
        axes[2].set_title('Symmetric MAPE by Horizon')
        axes[2].grid(True, alpha=0.3)
        
        # Add value labels
        for h, smape in zip(h_values, smape_values):
            axes[2].text(h, smape + smape*0.01, f'{smape:.1f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        plot_path = os.path.join(out_dir, 'errors_by_horizon.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error creating horizon error plot: {e}")
        plt.close('all')
        return None


def plot_horizon_errors_backup(predictions_df, city_id, save_path=None):
    """
    Create a plot showing error metrics by prediction horizon.
    
    Args:
        predictions_df: DataFrame with prediction results
        city_id: Municipality ID
        save_path: Optional path to save the plot
    """
    try:
        # Filter for city
        city_data = predictions_df[predictions_df['municipality_id'] == city_id].copy()
        if city_data.empty:
            print(f"No data found for city {city_id}")
            return None
        
        horizon_errors = []
        for horizon in sorted(city_data['horizon'].unique()):
            horizon_data = city_data[city_data['horizon'] == horizon]
            y_true = horizon_data['y_true'].values
            y_pred = horizon_data['y_pred'].values
            
            # Remove invalid values
            valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
            y_true = y_true[valid_mask]
            y_pred = y_pred[valid_mask]
            
            if len(y_true) > 0:
                mae = np.mean(np.abs(y_true - y_pred))
                rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                
                horizon_errors.append({
                    'horizon': horizon,
                    'MAE': mae,
                    'RMSE': rmse
                })
    
        if not horizon_errors:
            print("No valid errors to plot")
            return None
        
        errors_df = pd.DataFrame(horizon_errors)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_limit_figsize((12, 5)))
        
        # MAE plot
        ax1.bar(errors_df['horizon'], errors_df['MAE'], alpha=0.7, color='skyblue')
        ax1.set_title('Mean Absolute Error by Horizon')
        ax1.set_xlabel('Horizon (weeks)')
        ax1.set_ylabel('MAE')
        ax1.grid(True, alpha=0.3)
        
        # RMSE plot
        ax2.bar(errors_df['horizon'], errors_df['RMSE'], alpha=0.7, color='lightcoral')
        ax2.set_title('Root Mean Square Error by Horizon')
        ax2.set_xlabel('Horizon (weeks)')
        ax2.set_ylabel('RMSE')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Horizon errors plot saved: {save_path}")
        
        plt.show()
        return fig
        
    except Exception as e:
        print(f"Error creating horizon error plot: {e}")
        plt.close('all')
        return None
    
    plot_path = os.path.join(out_dir, 'errors_by_horizon.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_residual_hist(pred_df: pd.DataFrame, out_dir: str) -> str:
    """Plot residual histogram.
    
    Args:
        pred_df: Predictions DataFrame
        out_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if pred_df.empty:
        print("No data for residual plot")
        return None
    
    # Compute residuals
    residuals = pred_df['y_true'] - pred_df['y_pred']
    residuals = residuals[np.isfinite(residuals)]
    
    if len(residuals) == 0:
        print("No valid residuals to plot")
        return None
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_limit_figsize((12, 5)))
    
    # Histogram
    ax1.hist(residuals, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
    ax1.set_title('Residuals Distribution')
    ax1.set_xlabel('Residual (True - Predicted)')
    ax1.set_ylabel('Frequency')
    ax1.axvline(0, color='red', linestyle='--', alpha=0.7)
    ax1.grid(True, alpha=0.3)
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normal)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(out_dir, 'residual_histogram.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_model_comparison(pred_df: pd.DataFrame, out_dir: str) -> str:
    """Plot model comparison across horizons.
    
    Args:
        pred_df: Predictions DataFrame
        out_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if pred_df.empty or pred_df['model_name'].nunique() <= 1:
        print("Need multiple models for comparison plot")
        return None
    
    # Compute metrics by model and horizon
    comparison_data = []
    
    for model in pred_df['model_name'].unique():
        for horizon in sorted(pred_df['horizon'].unique()):
            model_horizon_data = pred_df[
                (pred_df['model_name'] == model) & 
                (pred_df['horizon'] == horizon)
            ]
            
            if not model_horizon_data.empty:
                y_true = model_horizon_data['y_true'].values
                y_pred = model_horizon_data['y_pred'].values
                
                valid_mask = np.isfinite(y_true) & np.isfinite(y_pred)
                y_true = y_true[valid_mask]
                y_pred = y_pred[valid_mask]
                
                if len(y_true) > 0:
                    mae = np.mean(np.abs(y_true - y_pred))
                    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
                    
                    comparison_data.append({
                        'model': model,
                        'horizon': horizon,
                        'MAE': mae,
                        'RMSE': rmse
                    })
    
    if not comparison_data:
        print("No valid comparison data")
        return None
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_limit_figsize((14, 6)))
    
    # MAE comparison
    models = comp_df['model'].unique()
    x_pos = np.arange(len(comp_df['horizon'].unique()))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        model_data = comp_df[comp_df['model'] == model]
        ax1.bar(x_pos + i * width, model_data['MAE'], width, 
               label=model, alpha=0.7)
    
    ax1.set_title('MAE Comparison by Model and Horizon')
    ax1.set_xlabel('Horizon (weeks)')
    ax1.set_ylabel('MAE')
    ax1.set_xticks(x_pos + width * (len(models) - 1) / 2)
    ax1.set_xticklabels(sorted(comp_df['horizon'].unique()))
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE comparison
    for i, model in enumerate(models):
        model_data = comp_df[comp_df['model'] == model]
        ax2.bar(x_pos + i * width, model_data['RMSE'], width, 
               label=model, alpha=0.7)
    
    ax2.set_title('RMSE Comparison by Model and Horizon')
    ax2.set_xlabel('Horizon (weeks)')
    ax2.set_ylabel('RMSE')
    ax2.set_xticks(x_pos + width * (len(models) - 1) / 2)
    ax2.set_xticklabels(sorted(comp_df['horizon'].unique()))
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(out_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_training_history(history_data: dict, city_id: int, fold_id: str, out_dir: str) -> str:
    """Plot training history showing loss curves.
    
    Args:
        history_data: Dictionary with 'train_losses' and 'val_losses' lists
        city_id: Municipality ID
        fold_id: Fold identifier  
        out_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        
        train_losses = history_data.get('train_losses', [])
        val_losses = history_data.get('val_losses', [])
        val_maes = history_data.get('val_maes', [])
        
        if not train_losses:
            print("No training history data available")
            return None
        
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create subplots
        if val_maes:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_limit_figsize((14, 6)))
        else:
            fig, ax1 = plt.subplots(1, 1, figsize=_limit_figsize((10, 6)))
        
        # Plot loss curves
        epochs = range(1, len(train_losses) + 1)
        
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2, alpha=0.8)
        if val_losses:
            val_epochs = range(1, len(val_losses) + 1)
            ax1.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2, alpha=0.8)
        
        ax1.set_title(f'Training History - City {city_id}, Fold {fold_id}')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # Add best validation loss marker if available
        if val_losses:
            best_val_idx = np.argmin(val_losses)
            best_val_loss = val_losses[best_val_idx]
            ax1.axvline(x=best_val_idx + 1, color='red', linestyle='--', alpha=0.5, 
                       label=f'Best Val Loss (epoch {best_val_idx + 1})')
            ax1.plot(best_val_idx + 1, best_val_loss, 'ro', markersize=8, markerfacecolor='red')
        
        # Plot validation MAE if available
        if val_maes:
            ax2.plot(val_epochs, val_maes, 'g-', label='Validation MAE', linewidth=2, alpha=0.8)
            ax2.set_title('Validation MAE')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('MAE (Original Scale)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Add best validation MAE marker
            best_mae_idx = np.argmin(val_maes)
            best_mae = val_maes[best_mae_idx]
            ax2.axvline(x=best_mae_idx + 1, color='green', linestyle='--', alpha=0.5,
                       label=f'Best Val MAE (epoch {best_mae_idx + 1})')
            ax2.plot(best_mae_idx + 1, best_mae, 'go', markersize=8, markerfacecolor='green')
        
        plt.tight_layout()
        
        plot_path = os.path.join(out_dir, f'training_history_city_{city_id}_fold_{fold_id}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error plotting training history: {e}")
        plt.close()
        return None


def plot_all_training_histories(all_histories: list, city_id: int, out_dir: str) -> str:
    """Plot aggregated training history from all folds.
    
    Args:
        all_histories: List of history dictionaries from all folds
        city_id: Municipality ID
        out_dir: Output directory
        
    Returns:
        Path to saved plot
    """
    try:
        os.makedirs(out_dir, exist_ok=True)
        
        if not all_histories:
            print("No training histories available")
            return None
        
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=_limit_figsize((14, 6)))
        
        # Colors for different folds
        colors = plt.cm.tab10(np.linspace(0, 1, len(all_histories)))
        
        # Plot individual fold histories
        for i, history in enumerate(all_histories):
            fold_id = history.get('fold_id', i)
            train_losses = history.get('train_losses', [])
            val_losses = history.get('val_losses', [])
            val_maes = history.get('val_maes', [])
            
            if train_losses:
                epochs = range(1, len(train_losses) + 1)
                ax1.plot(epochs, train_losses, '--', color=colors[i], alpha=0.7, 
                        label=f'Fold {fold_id} Train', linewidth=1)
                
            if val_losses:
                val_epochs = range(1, len(val_losses) + 1)
                ax1.plot(val_epochs, val_losses, '-', color=colors[i], alpha=0.8,
                        label=f'Fold {fold_id} Val', linewidth=2)
            
            if val_maes:
                ax2.plot(val_epochs, val_maes, '-', color=colors[i], alpha=0.8,
                        label=f'Fold {fold_id}', linewidth=2)
        
        # Compute average curves if multiple folds
        if len(all_histories) > 1:
            # Find common epoch range
            min_train_epochs = min(len(h.get('train_losses', [])) for h in all_histories if h.get('train_losses'))
            min_val_epochs = min(len(h.get('val_losses', [])) for h in all_histories if h.get('val_losses'))
            
            if min_train_epochs > 0:
                avg_train_losses = np.mean([h['train_losses'][:min_train_epochs] 
                                          for h in all_histories if h.get('train_losses')], axis=0)
                epochs = range(1, min_train_epochs + 1)
                ax1.plot(epochs, avg_train_losses, 'b-', linewidth=3, alpha=1.0, label='Avg Train')
            
            if min_val_epochs > 0:
                avg_val_losses = np.mean([h['val_losses'][:min_val_epochs] 
                                        for h in all_histories if h.get('val_losses')], axis=0)
                avg_val_maes = np.mean([h['val_maes'][:min_val_epochs] 
                                      for h in all_histories if h.get('val_maes')], axis=0)
                val_epochs = range(1, min_val_epochs + 1)
                ax1.plot(val_epochs, avg_val_losses, 'r-', linewidth=3, alpha=1.0, label='Avg Val Loss')
                ax2.plot(val_epochs, avg_val_maes, 'g-', linewidth=3, alpha=1.0, label='Avg Val MAE')
        
        # Configure loss plot
        ax1.set_title(f'Training History - City {city_id} (All Folds)')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Configure MAE plot
        ax2.set_title('Validation MAE (All Folds)')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (Original Scale)')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = os.path.join(out_dir, f'training_history_all_folds_city_{city_id}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path
        
    except Exception as e:
        print(f"Error plotting aggregated training history: {e}")
        plt.close()
        return None