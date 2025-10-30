"""Configuration defaults for forecasting system."""

def default_cfg() -> dict:
    """Return default configuration dictionary."""
    return {
        "target_col": "morbidity_rate",        # can switch to "mortality_rate"
        "train_window_weeks": 520,             # 10 years of training data (increased from 364)
        "step_weeks": 1,                       # Changed to 4 for easier plotting (was 1)
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
            "depth": 6,
            "learning_rate": 0.05,
            "iterations": 550,
            "early_stopping_rounds": 50
        },
        "gru": {
            "use_validation": True,            # Set to True to enable validation split with val_MAE early stopping
            "target_scaling": "none",          # Changed from "maxabs" to "none" - let model learn in original scale
            "feature_scaler": "standard",      # Options: "none", "standard", "minmax", "robust"
            "hidden_size": 128,               # Increased significantly - paper used 1000, we use 512 (compromise for municipal level)
            "num_layers": 2,                  # Reduced from 3 to 2 - paper likely used 1 layer, we compromise with 2
            "dropout": 0.1,                   # Reduced slightly with fewer layers but more units per layer
            "lr": 1e-3,                       # Keep same learning rate
            "weight_decay": 1e-5,             # Keep same regularization
            "optimizer": "adamw",
            "loss": "mae",                    # Switch back to MAE - Huber was too conservative, over-predicting zeros
            "huber_beta": 0.01,               # Keep for potential future use
            "max_epochs": 750,                # Keep increased epochs for better convergence
            "grad_clip": 1.0,                 # Keep gradient clipping
            "early_stopping": {"patience": 25, "min_delta": 1e-6},  # More patience like paper approach
            "rlrop": {"factor": 0.8, "patience": 5, "cooldown": 10},  # Keep same LR scheduling
            # New activation and output options
            "output_activation": None,        # Remove Softplus constraint (was causing range issues)
            "post_process_clip": True,        # Enable post-processing clipping
            "min_output": 0.0,               # Minimum output value (clip negatives to 0)
            "max_output": 0.2,               # Maximum output value based on observed data (0.128 max observed + buffer)
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