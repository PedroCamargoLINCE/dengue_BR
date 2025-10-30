"""Configuration defaults for        "gru": {
            "use_validation": False,           # Set to False to disable val split and use train loss monitoring
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "optimizer": "adamw",
            "loss": "mae",
            "max_epochs": 200,
            "grad_clip": 1.0,
            "early_stopping": {"patience": 10, "min_delta": 0.0},
            "rlrop": {"factor": 0.5, "patience": 7, "cooldown": 0}casting system."""

def default_cfg() -> dict:
    """Return default configuration dictionary."""
    return {
        "target_col": "morbidity_rate",        # can switch to "mortality_rate"
        "train_window_weeks": 520,             # 10 years of training data (increased from 364)
        "step_weeks": 1,
        "horizon": 4,
        "lookback_L": 104,
        "val_weeks": 52,                       # internal val for GRU/TFT only (increased from 12)
        "lags_target": [1, 2, 3, 4, 8, 12, 16, 24, 52, 104],
        "lags_climate": [1, 2, 4, 8],          # causal lags for climate features
        "calendar_features": ["weekofyear_sin", "weekofyear_cos"],
        "static_cols": ["poverty_index", "urbanization_index", "demographic_density"],
        "climate_cols": ["min_humidity", "monthly_precipitation", "max_temperature"],
        "neighbor_map_path": "data/neighbor_map.csv",
        "dataset_path": "data/unified_dataset.csv",
        "models_dir": "models",
        "artifacts_dir": "artifacts",
        "figures_dir": "figures",
        "seed": 42,
        "catboost": {
            "loss": "MultiRMSE",
            "depth": 8,
            "learning_rate": 0.05,
            "iterations": 450,
            "early_stopping_rounds": 50
        },
        "gru": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "optimizer": "adamw",
            "loss": "rmse",
            "max_epochs": 200,
            "grad_clip": 1.0,
            "early_stopping": {"patience": 20, "min_delta": 0.0},
            "rlrop": {"factor": 0.5, "patience": 10, "cooldown": 0}
        },
        "tft": {
            "hidden_size": 64,
            "dropout": 0.2,
            "num_heads": 4,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "lr": 1e-3,
            "weight_decay": 1e-5,
            "optimizer": "adamw",
            "loss": "mae",
            "max_epochs": 200,
            "grad_clip": 1.0,
            "early_stopping": {"patience": 10, "min_delta": 0.0},
            "rlrop": {"factor": 0.5, "patience": 7, "cooldown": 0}
        }
    }