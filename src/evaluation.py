from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-3) -> float:
    numerator = np.abs(y_pred - y_true)
    denominator = np.abs(y_true) + np.abs(y_pred) + epsilon
    return float(np.mean(200.0 * numerator / denominator))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    smape_value = smape(y_true, y_pred)
    r2 = float(r2_score(y_true, y_pred)) if np.any(np.isfinite(y_true)) else float("nan")
    return {"mae": mae, "rmse": rmse, "smape": smape_value, "r2": r2}


def evaluate_predictions(predictions: pd.DataFrame) -> Tuple[Dict[str, float], pd.DataFrame]:
    required_columns = {"y_true", "y_pred"}
    missing = required_columns - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions dataframe missing columns: {missing}")

    y_true = predictions["y_true"].to_numpy(dtype=np.float64)
    y_pred = predictions["y_pred"].to_numpy(dtype=np.float64)
    metrics = compute_metrics(y_true, y_pred)

    summary = predictions.copy()
    summary["error"] = summary["y_pred"] - summary["y_true"]
    summary["abs_error"] = summary["error"].abs()
    summary["ape"] = np.where(
        np.abs(summary["y_true"]) > 1e-8,
        np.abs(summary["error"]) / np.abs(summary["y_true"]),
        np.nan,
    )

    return metrics, summary


def compute_horizon_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if "horizon" not in predictions.columns:
        raise ValueError("Predictions dataframe must include a 'horizon' column for horizon metrics.")
    metrics = []
    for horizon, group in predictions.groupby("horizon"):
        metrics.append({"horizon": int(horizon), **compute_metrics(group["y_true"], group["y_pred"])})
    horizon_df = pd.DataFrame(metrics).sort_values("horizon").reset_index(drop=True)
    return horizon_df


def compute_window_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    if "window_id" not in predictions.columns:
        raise ValueError("Predictions dataframe must include a 'window_id' column for window metrics.")
    metrics = []
    for window_id, group in predictions.groupby("window_id"):
        metrics.append({"window_id": int(window_id), **compute_metrics(group["y_true"], group["y_pred"])})
    window_df = pd.DataFrame(metrics).sort_values("window_id").reset_index(drop=True)
    return window_df


def plot_predictions_over_time(
    predictions: pd.DataFrame,
    date_col: str,
    ax: Optional[plt.Axes] = None,
    rolling: Optional[int] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if date_col not in predictions.columns:
        raise ValueError(f"Column '{date_col}' not found in predictions dataframe.")

    df_plot = predictions.sort_values(by=date_col)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(df_plot[date_col], df_plot["y_true"], label="Observed", color="#1f77b4", linewidth=1.5)
    ax.plot(df_plot[date_col], df_plot["y_pred"], label="Forecast", color="#ff7f0e", linewidth=1.5)

    if rolling is not None and rolling > 1:
        ax.plot(
            df_plot[date_col],
            df_plot["y_pred"].rolling(rolling, min_periods=1).mean(),
            label=f"Forecast (rolling {rolling})",
            color="#2ca02c",
            linewidth=1.2,
            linestyle="--",
        )

    ax.set_xlabel("Week")
    ax.set_ylabel("Target value")
    ax.set_title("Observed vs Forecasted values")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig, ax


def plot_residuals_by_horizon(
    predictions: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if "horizon" not in predictions.columns:
        raise ValueError("Predictions dataframe must include a 'horizon' column for residual plots.")

    df_plot = predictions.copy()
    df_plot["residual"] = df_plot["y_pred"] - df_plot["y_true"]

    horizons = sorted(df_plot["horizon"].unique())
    residual_groups = [df_plot.loc[df_plot["horizon"] == h, "residual"].to_numpy() for h in horizons]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.boxplot(residual_groups, labels=[str(h) for h in horizons], showfliers=False)
    ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("Forecast horizon (weeks ahead)")
    ax.set_ylabel("Residual (y_pred - y_true)")
    ax.set_title("Residual distribution by horizon")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def build_history_dataframe(window_histories: list[Dict]) -> pd.DataFrame:
    """Flatten the per-window training history into a dataframe."""
    records = []
    for window in window_histories:
        history = window.get("history") or []
        window_id = window.get("window_id")
        for row in history:
            records.append(
                {
                    "window_id": window_id,
                    "epoch": row.get("epoch"),
                    "train_loss": row.get("train_loss"),
                    "val_loss": row.get("val_loss"),
                    "lr": row.get("lr"),
                }
            )
    if not records:
        return pd.DataFrame(columns=["window_id", "epoch", "train_loss", "val_loss", "lr"])
    return pd.DataFrame(records)



def plot_window_histories(
    window_histories: list[Dict],
    n_cols: int = 3,
    figsize: Tuple[int, int] = (12, 8),
) -> Tuple[plt.Figure, np.ndarray]:
    """Plot train/validation loss curves for each walk-forward window."""
    if not window_histories:
        raise ValueError("window_histories is empty; run the training pipeline first")

    n_windows = len(window_histories)
    n_cols = max(1, n_cols)
    n_rows = math.ceil(n_windows / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    for ax in axes:
        ax.axis("off")

    for idx, window in enumerate(window_histories):
        ax = axes[idx]
        ax.axis("on")
        history = window.get("history") or []
        if not history:
            ax.text(0.5, 0.5, "No history", ha="center", va="center")
            ax.set_title(f"Window {window.get('window_id')}")
            continue
        epochs = [row.get("epoch", i + 1) for i, row in enumerate(history)]
        train_loss = [row.get("train_loss") for row in history]
        if train_loss:
            ax.plot(epochs, train_loss, label="Train", color="#1f77b4", linewidth=1.5)
        val_epochs = [row.get("epoch", i + 1) for i, row in enumerate(history) if row.get("val_loss") is not None]
        val_loss = [row.get("val_loss") for row in history if row.get("val_loss") is not None]
        if val_loss:
            ax.plot(val_epochs, val_loss, label="Val", color="#ff7f0e", linewidth=1.5)
        lr = [row.get("lr") for row in history if row.get("lr") is not None]
        if lr and len(lr) == len(epochs):
            ax2 = ax.twinx()
            ax2.plot(epochs, lr, label="LR", color="#2ca02c", linestyle="--", linewidth=1)
            ax2.set_ylabel("LR", color="#2ca02c")
            ax2.tick_params(axis="y", colors="#2ca02c")
            ax2.grid(False)
        ax.set_title(f"Window {window.get('window_id')}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    for ax in axes[n_windows:]:
        ax.remove()

    fig.tight_layout()
    return fig, axes



def plot_metric_by_horizon(
    predictions: pd.DataFrame,
    metric: str = "mae",
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    horizon_metrics = compute_horizon_metrics(predictions)
    if metric not in horizon_metrics.columns:
        raise ValueError(f"Metric '{metric}' is not available. Choose from {list(horizon_metrics.columns)}")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    ax.bar(horizon_metrics["horizon"], horizon_metrics[metric], color="#9467bd", alpha=0.8)
    ax.set_xlabel("Forecast horizon (weeks ahead)")
    ax.set_ylabel(metric.upper())
    ax.set_title(f"{metric.upper()} by forecast horizon")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax, horizon_metrics


__all__ = [
    "compute_metrics",
    "build_history_dataframe",
    "plot_window_histories",
    "evaluate_predictions",
    "smape",
    "compute_horizon_metrics",
    "compute_window_metrics",
    "plot_predictions_over_time",
    "plot_residuals_by_horizon",
    "plot_metric_by_horizon",
]
