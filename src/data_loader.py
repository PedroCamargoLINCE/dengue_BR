from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class DataPreparationArtifacts:
    """Container holding the prepared dataset and metadata for downstream steps."""

    data: pd.DataFrame
    feature_columns: List[str]
    target_column: str
    date_column: str
    municipality_code: str
    neighbor_columns: List[str]
    seasonal_columns: List[str]
    exogenous_columns: List[str]
    engineered_columns: List[str]


class DataPreparationError(Exception):
    """Raised when data preparation encounters unrecoverable issues."""


def _parse_config(config: Dict) -> Dict:
    for section in ("data", "features"):
        if section not in config:
            raise KeyError("Missing configuration section: {0}".format(section))
    return config


def _load_raw_frames(config: Dict, project_root: Optional[Path] = None) -> Dict[str, pd.DataFrame]:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]
    data_paths = {
        "timeseries": project_root / "data" / "unified_dataset.csv",
        "neighbor_map": project_root / "data" / "neighbor_map.csv",
    }

    missing_files = [str(path) for path in data_paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError("Required data files were not found: {0}".format(", ".join(missing_files)))

    date_col = config["data"]["date_col"]
    df_timeseries = pd.read_csv(data_paths["timeseries"], parse_dates=[date_col])
    df_neighbors = pd.read_csv(data_paths["neighbor_map"])
    return {"timeseries": df_timeseries, "neighbor_map": df_neighbors}


def _validate_columns(df: pd.DataFrame, expected: List[str]) -> None:
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise DataPreparationError("Missing required columns: {0}".format(missing))


def _ensure_string_codes(df: pd.DataFrame, column: str) -> pd.Series:
    series = df[column]
    if pd.api.types.is_integer_dtype(series):
        return series.astype(str)
    return series.astype(str)


def _add_seasonal_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    df = df.copy()
    iso_calendar = df[date_col].dt.isocalendar()
    week_series = iso_calendar.week.astype(int)
    month_series = df[date_col].dt.month.astype(int)

    df["seasonal_sin_week"] = np.sin(2.0 * np.pi * week_series / 52.0)
    df["seasonal_cos_week"] = np.cos(2.0 * np.pi * week_series / 52.0)
    df["seasonal_sin_month"] = np.sin(2.0 * np.pi * month_series / 12.0)
    df["seasonal_cos_month"] = np.cos(2.0 * np.pi * month_series / 12.0)
    df["week_of_year"] = week_series
    df["month"] = month_series
    return df


def _add_seasonal_decomposition(df: pd.DataFrame, target_col: str, period: int = 52) -> pd.DataFrame:
    """
    Add seasonal decomposition features: trend, seasonal, and residual components.
    Uses simple moving average for trend and seasonal pattern extraction.
    """
    df = df.copy()
    target = df[target_col].values
    n = len(target)
    
    # Compute trend using centered moving average
    half_period = period // 2
    trend = np.full(n, np.nan)
    for i in range(half_period, n - half_period):
        trend[i] = np.nanmean(target[i - half_period:i + half_period + 1])
    
    # Fill edges with linear extrapolation
    valid_idx = ~np.isnan(trend)
    if np.sum(valid_idx) > 1:
        first_valid = np.where(valid_idx)[0][0]
        last_valid = np.where(valid_idx)[0][-1]
        # Forward fill at start
        trend[:first_valid] = trend[first_valid]
        # Backward fill at end
        trend[last_valid + 1:] = trend[last_valid]
    
    # Detrend
    detrended = target - trend
    
    # Compute seasonal component (average pattern across cycles)
    seasonal = np.full(n, np.nan)
    for week_in_cycle in range(period):
        indices = np.arange(week_in_cycle, n, period)
        if len(indices) > 0:
            seasonal_mean = np.nanmean(detrended[indices])
            seasonal[indices] = seasonal_mean
    
    # Residual
    residual = target - trend - seasonal
    
    # Add components to dataframe
    df["seasonal_component"] = seasonal
    df["trend_component"] = trend
    df["residual_component"] = residual
    
    # Lagged components (no leakage)
    df["seasonal_lag_1"] = df["seasonal_component"].shift(1)
    df["trend_lag_1"] = df["trend_component"].shift(1)
    
    # Trend differences (rate of change)
    df["trend_diff_1"] = df["trend_component"].diff(1)
    df["trend_diff_4"] = df["trend_component"].diff(4)
    
    # Fill NaN with 0
    decomp_cols = [
        "seasonal_component", "trend_component", "residual_component",
        "seasonal_lag_1", "trend_lag_1", "trend_diff_1", "trend_diff_4"
    ]
    df[decomp_cols] = df[decomp_cols].fillna(0)
    
    return df


def _pivot_neighbor_targets(
    df: pd.DataFrame,
    neighbor_map: pd.DataFrame,
    target_column: str,
    date_col: str,
    cd_mun: str,
) -> pd.DataFrame:
    neighbor_map = neighbor_map.copy()
    neighbor_map["cd_mun"] = _ensure_string_codes(neighbor_map, "cd_mun")
    neighbor_map["neighbor_cd_mun"] = _ensure_string_codes(neighbor_map, "neighbor_cd_mun")

    neighbors = (
        neighbor_map.loc[neighbor_map["cd_mun"] == cd_mun, "neighbor_cd_mun"]
        .dropna()
        .unique()
        .tolist()
    )
    if not neighbors:
        return pd.DataFrame()

    neighbor_df = df[df["cd_mun"].isin(neighbors)].copy()
    if neighbor_df.empty:
        return pd.DataFrame()

    pivot = (
        neighbor_df[["cd_mun", date_col, target_column]]
        .pivot(index=date_col, columns="cd_mun", values=target_column)
        .sort_index(axis=1)
    )
    pivot.columns = ["neighbor_{0}_{1}".format(code, target_column) for code in pivot.columns]
    pivot = pivot.sort_index()
    return pivot


def _apply_feature_engineering(
    df: pd.DataFrame,
    target_col: str,
    config: Dict,
) -> List[str]:
    engineered_columns: List[str] = []
    fe_cfg = config.get("feature_engineering", {}) or {}

    # Target lags
    raw_lags = fe_cfg.get("target_lags", [])
    lags = sorted({int(lag) for lag in raw_lags if int(lag) > 0})
    for lag in lags:
        col_name = "{0}_lag_{1}".format(target_col, lag)
        df[col_name] = df[target_col].shift(lag)
        engineered_columns.append(col_name)

    # Rolling aggregates
    rolling_windows = sorted({int(win) for win in fe_cfg.get("rolling_windows", []) if int(win) > 1})
    stats = [stat.lower() for stat in fe_cfg.get("rolling_statistics", ["mean", "max"])]
    for window in rolling_windows:
        rolling_series = df[target_col].rolling(window=window, min_periods=window)
        if "mean" in stats:
            col_name = "{0}_rollmean_{1}".format(target_col, window)
            df[col_name] = rolling_series.mean().shift(1)
            engineered_columns.append(col_name)
        if "max" in stats:
            col_name = "{0}_rollmax_{1}".format(target_col, window)
            df[col_name] = rolling_series.max().shift(1)
            engineered_columns.append(col_name)
        if "std" in stats:
            col_name = "{0}_rollstd_{1}".format(target_col, window)
            df[col_name] = rolling_series.std().shift(1)
            engineered_columns.append(col_name)

    # Growth rate features
    if fe_cfg.get("include_growth_rate", False):
        lag_1 = df[target_col].shift(1)
        diff_col = "{0}_diff_1".format(target_col)
        pct_col = "{0}_pct_change_1".format(target_col)
        logdiff_col = "{0}_log_diff_1".format(target_col)
        df[diff_col] = df[target_col] - lag_1
        denom = lag_1.fillna(0.0)
        safe_denom = np.where(np.abs(denom) > 1e-6, denom, np.sign(denom) * 1e-6)
        df[pct_col] = (df[target_col] - lag_1) / safe_denom
        log_input = lag_1.fillna(0.0).clip(lower=0.0)
        df[logdiff_col] = np.log1p(df[target_col]) - np.log1p(log_input)
        engineered_columns.extend([diff_col, pct_col, logdiff_col])

    # Feature lags (e.g., weather covariates)
    feature_lags = fe_cfg.get("feature_lags", {}) or {}
    for column, lag_list in feature_lags.items():
        if column not in df.columns:
            continue
        lag_values = sorted({int(lag) for lag in lag_list if int(lag) > 0})
        for lag in lag_values:
            col_name = "{0}_lag_{1}".format(column, lag)
            if col_name in df.columns:
                continue
            df[col_name] = df[column].shift(lag)
            engineered_columns.append(col_name)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return engineered_columns


def prepare_time_series(config: Dict) -> DataPreparationArtifacts:
    """Prepare the time series dataset based on the provided configuration."""

    config = _parse_config(config)
    raw_frames = _load_raw_frames(config)

    ts_df = raw_frames["timeseries"].copy()
    neighbor_df = raw_frames["neighbor_map"]

    data_cfg = config["data"]
    feature_cfg = config["features"]

    date_col = data_cfg["date_col"]
    target_col = data_cfg["target_variable"]
    cd_mun = str(data_cfg["cd_mun_to_train"])

    _validate_columns(ts_df, ["cd_mun", date_col, target_col])
    ts_df["cd_mun"] = _ensure_string_codes(ts_df, "cd_mun")

    df_main = ts_df.loc[ts_df["cd_mun"] == cd_mun].copy()
    if df_main.empty:
        raise DataPreparationError(
            "No rows found for municipality code {0}. Check the configuration and dataset.".format(cd_mun)
        )

    df_main.sort_values(by=date_col, inplace=True)

    exog_features = feature_cfg.get("exog_features", []) or []
    available_exog = [col for col in exog_features if col in df_main.columns]
    missing_exog = sorted(set(exog_features) - set(available_exog))
    if missing_exog:
        print("[data_loader] Warning: Missing exogenous features skipped: {0}".format(missing_exog))

    selected_columns = ['cd_mun', date_col, target_col] + available_exog
    df_features = df_main[selected_columns].copy()

    seasonal_columns: List[str] = []
    if feature_cfg.get("add_seasonal_features", False):
        df_features = _add_seasonal_features(df_features, date_col)
        seasonal_columns = [
            "seasonal_sin_week",
            "seasonal_cos_week",
            "seasonal_sin_month",
            "seasonal_cos_month",
            "week_of_year",
            "month",
        ]
        
        # Add seasonal decomposition features
        df_features = _add_seasonal_decomposition(df_features, target_col, period=52)
        seasonal_columns.extend([
            "seasonal_component",
            "trend_component",
            "residual_component",
            "seasonal_lag_1",
            "trend_lag_1",
            "trend_diff_1",
            "trend_diff_4",
        ])

    neighbor_columns: List[str] = []
    if feature_cfg.get("use_neighbor_effect", False):
        pivot = _pivot_neighbor_targets(ts_df, neighbor_df, target_col, date_col, cd_mun)
        if not pivot.empty:
            df_features = df_features.merge(pivot, left_on=date_col, right_index=True, how="left")
            neighbor_columns = [col for col in pivot.columns]
        else:
            print("[data_loader] Warning: No neighbor data available for the selected municipality.")

    df_features.sort_values(by=date_col, inplace=True)
    df_features.reset_index(drop=True, inplace=True)

    if neighbor_columns:
        na_ratio = df_features[neighbor_columns].isna().mean().max()
        if na_ratio > 0.0:
            print(
                "[data_loader] Info: Neighbor features contain {0:.2%} missing values. Forward/backward fill applied, residual NaNs set to 0.".format(na_ratio)
            )
            df_features[neighbor_columns] = (
                df_features[neighbor_columns]
                .fillna(method="ffill")
                .fillna(method="bfill")
                .fillna(0.0)
            )

    engineered_columns = _apply_feature_engineering(df_features, target_col, feature_cfg)

    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    if engineered_columns:
        df_features.dropna(subset=engineered_columns, inplace=True)
        df_features.reset_index(drop=True, inplace=True)

    all_feature_columns = available_exog + neighbor_columns + seasonal_columns + engineered_columns

    assert df_features[date_col].is_monotonic_increasing
    assert df_features[target_col].notna().all()

    feature_matrix = df_features[['cd_mun', date_col, target_col] + all_feature_columns].copy()

    print("[data_loader] Prepared dataset summary:")
    summary_lines = [
        "  - Municipality: {0}".format(cd_mun),
        "  - Rows: {0}".format(len(feature_matrix)),
        "  - Feature count: {0}".format(len(all_feature_columns)),
        "  - Engineered features: {0}".format(len(engineered_columns)),
        "  - Date range: {0} to {1}".format(
            feature_matrix[date_col].min().date(),
            feature_matrix[date_col].max().date(),
        ),
    ]
    for line in summary_lines:
        print(line)

    return DataPreparationArtifacts(
        data=feature_matrix,
        feature_columns=all_feature_columns,
        target_column=target_col,
        date_column=date_col,
        municipality_code=cd_mun,
        neighbor_columns=neighbor_columns,
        seasonal_columns=seasonal_columns,
        exogenous_columns=available_exog,
        engineered_columns=engineered_columns,
    )


__all__ = ["DataPreparationArtifacts", "DataPreparationError", "prepare_time_series"]

