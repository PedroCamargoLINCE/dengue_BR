# Batch Training Report: Brazilian Capital Cities

**Generated:** 2025-10-13 15:22:29

## Overview

- **Cities trained:** 26
- **Configurations tested:** 3
- **Total runs:** 78
- **Successful runs:** 69
- **Failed runs:** 9

## Configurations

### Config 1: Tweedie Loss (High Capacity)
- Loss: Tweedie (p=1.5)
- Architecture: 256 hidden × 3 layers
- Lookback: 104 weeks
- Dropout: 0.3

### Config 2: MAE Loss (Moderate Capacity)
- Loss: MAE
- Architecture: 128 hidden × 2 layers
- Lookback: 104 weeks
- Dropout: 0.2

### Config 3: MAE Loss (Simple)
- Loss: MAE
- Architecture: 64 hidden × 2 layers
- Lookback: 52 weeks
- Dropout: 0.1

## Results Summary


### Average Metrics by Configuration

**config_1_tweedie_high_horizon:**
- MAE: 0.4037 ± 0.7175
- RMSE: 0.6477 ± 1.3963
- R²: -0.9643 ± 1.9521
- sMAPE: 88.94% ± 20.45%

**config_2_mae_moderate_horizon:**
- MAE: 0.5303 ± 1.4162
- RMSE: 1.1761 ± 4.0025
- R²: -4.0221 ± 17.3316
- sMAPE: 86.75% ± 17.56%

**config_3_mae_simple_horizon:**
- MAE: 0.2872 ± 0.3306
- RMSE: 0.3997 ± 0.4828
- R²: -1.8926 ± 6.5724
- sMAPE: 98.77% ± 19.47%


## Output Directory

`C:\Users\pedro\OneDrive - Unesp\Documentos\GitHub\dengue4\artifacts\batch_capital_cities\20251013_144325`
