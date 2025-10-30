from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from .model import create_model_from_config

try:
    from scipy import stats
except ImportError:
    stats = None


class TweedieLoss(nn.Module):
    """Tweedie loss supporting 1 < power < 2 (compound Poisson)."""

    def __init__(self, power: float = 1.5, eps: float = 1e-6, reduction: str = "mean") -> None:
        super().__init__()
        if not (1.0 < power < 2.0):
            raise ValueError('Tweedie power must be in (1, 2) for compound Poisson')
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError('Unsupported reduction for TweedieLoss: {0}'.format(reduction))
        self.power = float(power)
        self.eps = float(eps)
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_clamped = torch.clamp(input, min=self.eps)
        target_clamped = torch.clamp(target, min=0.0)
        term1 = torch.pow(input_clamped, 2.0 - self.power) / (2.0 - self.power)
        term2 = target_clamped * torch.pow(input_clamped, 1.0 - self.power) / (1.0 - self.power)
        loss = term1 - term2
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()

class BaseTargetTransform:
    def fit(self, y: np.ndarray) -> "BaseTargetTransform":
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        return y

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        return y


class IdentityTransform(BaseTargetTransform):
    def fit(self, y: np.ndarray) -> "IdentityTransform":
        self._fitted = True
        return self


class Log1pTransform(BaseTargetTransform):
    def fit(self, y: np.ndarray) -> "Log1pTransform":
        y = np.asarray(y)
        if np.min(y) < -0.999999:
            raise ValueError("log1p transform requires target values >= -1.0")
        self._fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Log1pTransform must be fitted before transform is called.")
        return np.log1p(np.clip(np.asarray(y), a_min=-0.999999, a_max=None))

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if not getattr(self, "_fitted", False):
            raise RuntimeError("Log1pTransform must be fitted before inverse_transform is called.")
        values = np.asarray(y, dtype=np.float64)
        clipped = np.where(values < 0.0, 0.0, values)
        return np.expm1(clipped)


class BoxCoxTransform(BaseTargetTransform):
    def __init__(self) -> None:
        self.lambda_: Optional[float] = None
        self.shift_: float = 0.0
        self._fitted = False

    def fit(self, y: np.ndarray) -> "BoxCoxTransform":
        if stats is None:
            raise ImportError(
                "scipy is required for BoxCox transform. Install scipy or choose another target_transform."
            )
        y = np.asarray(y, dtype=np.float64)
        min_val = np.min(y)
        self.shift_ = 0.0
        if min_val <= 0:
            self.shift_ = abs(min_val) + 1e-6
        shifted = y + self.shift_
        transformed, lambda_opt = stats.boxcox(shifted)
        self.lambda_ = float(lambda_opt)
        self._fitted = True
        return self

    def transform(self, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BoxCoxTransform must be fitted before transform is called.")
        shifted = np.asarray(y, dtype=np.float64) + self.shift_
        return stats.boxcox(shifted, lmbda=self.lambda_)

    def inverse_transform(self, y: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BoxCoxTransform must be fitted before inverse_transform is called.")
        inverted = stats.inv_boxcox(np.asarray(y, dtype=np.float64), self.lambda_)
        return inverted - self.shift_

def build_target_transform(name: Optional[str]) -> BaseTargetTransform:
    if name is None:
        return IdentityTransform()
    key = str(name).lower()
    if key in {"none", "identity", ""}:
        return IdentityTransform()
    if key == "log1p":
        return Log1pTransform()
    if key == "boxcox":
        return BoxCoxTransform()
    raise ValueError(f"Unsupported target_transform {name}")


SCALER_MAP: Dict[str, type] = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "absmax": MaxAbsScaler,
}


def build_scaler(name: Optional[str]):
    if name is None:
        return None
    key = str(name).lower()
    if key in {"none", ""}:
        return None
    if key not in SCALER_MAP:
        raise ValueError(f"Unsupported scaler option {name}")
    return SCALER_MAP[key]()

class TimeSeriesWindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, weights: Optional[np.ndarray] = None) -> None:
        if X.ndim != 3:
            raise ValueError("Expected features with shape (samples, lookback, features)")
        if y.ndim != 2:
            raise ValueError("Expected targets with shape (samples, horizon)")
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.weights = None
        if weights is not None:
            if weights.shape != y.shape:
                raise ValueError("Weights must have the same shape as targets")
            self.weights = torch.from_numpy(weights.astype(np.float32))

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        if self.weights is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.weights[idx]


def make_sequences(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int,
    horizon: int,
    targets_original: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if len(features) <= lookback:
        raise ValueError("Not enough timesteps to assemble the requested lookback window")
    if len(features) <= horizon:
        raise ValueError("Not enough timesteps to cover the requested forecast horizon")
    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    y_original_list: List[np.ndarray] = []
    anchors: List[int] = []
    for anchor in range(lookback, len(features) - horizon + 1):
        X_list.append(features[anchor - lookback : anchor])
        y_list.append(targets[anchor : anchor + horizon])
        if targets_original is not None:
            y_original_list.append(targets_original[anchor : anchor + horizon])
        anchors.append(anchor)
    if not X_list:
        raise ValueError("Unable to build any training samples with the provided configuration")
    X_stack = np.stack(X_list, axis=0)
    y_stack = np.stack(y_list, axis=0)
    anchor_array = np.asarray(anchors, dtype=np.int64)
    if y_original_list:
        y_original_stack = np.stack(y_original_list, axis=0)
    else:
        y_original_stack = None
    return X_stack, y_stack, anchor_array, y_original_stack


def ensure_no_nan(array: np.ndarray, context: str) -> None:
    if np.isnan(array).any():
        raise ValueError(f"Detected NaN values in {context}")


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_preference: str) -> torch.device:
    if device_preference.lower() == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_preference.lower() == "cuda" and not torch.cuda.is_available():
        print("[training] Warning: CUDA requested but not available. Using CPU instead.")
    return torch.device("cpu")


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def build_loss(name: str, training_cfg: Dict, reduction: str = "mean") -> nn.Module:
    key = name.lower()
    if key == "mae":
        return nn.L1Loss(reduction=reduction)
    if key == "mse":
        return nn.MSELoss(reduction=reduction)
    if key == "huber":
        delta = training_cfg.get("huber_delta", 1.0)
        return nn.HuberLoss(delta=delta, reduction=reduction)
    if key == "tweedie":
        power = training_cfg.get("tweedie_power", 1.5)
        return TweedieLoss(power=power, reduction=reduction)
    raise ValueError(f"Unsupported loss function {name}")


def sanity_check_inverse(original: np.ndarray, recovered: np.ndarray, tolerance: float = 1e-5) -> None:
    if not np.allclose(original, recovered, atol=tolerance, equal_nan=True):
        raise ValueError("Inverse transformation check failed; recovered values diverge from originals")

@dataclass
class WindowDefinition:
    window_id: int
    train_start: int
    train_end: int
    val_start: Optional[int]
    forecast_start: int
    forecast_end: int


def generate_window_definitions(
    num_rows: int,
    wf_cfg: Dict,
    lookback: int,
    horizon: int,
) -> List[WindowDefinition]:
    train_window_weeks = int(round(float(wf_cfg["train_window_years"]) * 52))
    validation_window_weeks = int(wf_cfg.get("validation_window_weeks", 0))
    validation_split_pct = float(wf_cfg.get("validation_split_pct", 0.0))
    if validation_split_pct > 0.0:
        candidate = int(round(train_window_weeks * validation_split_pct))
        validation_window_weeks = max(validation_window_weeks, candidate) if validation_window_weeks else candidate
    use_validation = bool(wf_cfg.get("use_validation_set", True))
    test_window_weeks = int(wf_cfg["test_window_weeks"])
    step_weeks = int(wf_cfg["step_weeks"])

    if train_window_weeks < lookback + horizon:
        raise ValueError("Train window is too short compared to lookback and horizon")
    if step_weeks <= 0:
        raise ValueError("step_weeks must be positive")

    test_start = num_rows - test_window_weeks
    if test_start <= 0:
        raise ValueError("Not enough history to allocate the requested test window")
    test_end = num_rows - 1

    windows: List[WindowDefinition] = []
    window_id = 0
    forecast_start = test_start
    while forecast_start + horizon - 1 <= test_end:
        forecast_end = forecast_start + horizon - 1
        train_end = forecast_start - 1
        train_start = train_end - train_window_weeks + 1
        if train_start < 0:
            raise ValueError("Training window extends before the start of the dataset")

        val_start: Optional[int] = None
        if use_validation and validation_window_weeks > 0:
            candidate_val_start = train_end - validation_window_weeks + 1
            candidate_val_start = max(candidate_val_start, train_start + lookback)
            if candidate_val_start <= train_end - horizon + 1:
                val_start = candidate_val_start

        windows.append(
            WindowDefinition(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                val_start=val_start,
                forecast_start=forecast_start,
                forecast_end=forecast_end,
            )
        )
        forecast_start += step_weeks
        window_id += 1
        if forecast_start > test_end - horizon + 1:
            break

    if not windows:
        raise ValueError("No walk-forward windows could be generated with the supplied configuration")
    return windows

def fit_and_transform_window(
    features_window: np.ndarray,
    targets_window: np.ndarray,
    preprocessing_cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray, BaseTargetTransform, Optional[object], Optional[object], Optional[object]]:
    target_transform = build_target_transform(preprocessing_cfg.get("target_transform"))
    target_transform.fit(targets_window)
    transformed_targets = target_transform.transform(targets_window)
    sanity_check_inverse(targets_window, target_transform.inverse_transform(transformed_targets))

    target_scaler = build_scaler(preprocessing_cfg.get("target_scaler"))
    if target_scaler is not None:
        target_scaler.fit(transformed_targets.reshape(-1, 1))
        scaled_targets = target_scaler.transform(transformed_targets.reshape(-1, 1)).reshape(-1)
        sanity_check_inverse(
            transformed_targets,
            target_scaler.inverse_transform(scaled_targets.reshape(-1, 1)).reshape(-1),
        )
        if hasattr(target_scaler, "data_min_"):
            print(
                "[training] Target scaler applied. Sample range "
                f"{scaled_targets.min():.4f} .. {scaled_targets.max():.4f}"
            )
    else:
        scaled_targets = transformed_targets.astype(np.float64)

    window_target_scaler = build_scaler(preprocessing_cfg.get("target_window_scaler"))
    if window_target_scaler is not None:
        window_target_scaler.fit(scaled_targets.reshape(-1, 1))
        scaled_targets = window_target_scaler.transform(scaled_targets.reshape(-1, 1)).reshape(-1)
        ensure_no_nan(scaled_targets, "window-scaled target values")
    else:
        window_target_scaler = None
    ensure_no_nan(scaled_targets, "scaled target values")

    feature_scaler = build_scaler(preprocessing_cfg.get("exog_scaler"))
    if feature_scaler is not None:
        feature_scaler.fit(features_window)
        scaled_features = feature_scaler.transform(features_window)
        if isinstance(feature_scaler, MinMaxScaler):
            print(
                "[training] Feature MinMax scaling range "
                f"{scaled_features.min():.4f} .. {scaled_features.max():.4f}"
            )
    else:
        scaled_features = features_window.astype(np.float64)

    ensure_no_nan(scaled_targets, "scaled target values")
    ensure_no_nan(scaled_features, "scaled features")
    return (
        scaled_features.astype(np.float32),
        scaled_targets.astype(np.float32),
        target_transform,
        target_scaler,
        feature_scaler,
        window_target_scaler,
    )

def create_window_dataloaders(
    window_def: WindowDefinition,
    features_scaled: np.ndarray,
    targets_scaled: np.ndarray,
    targets_original: np.ndarray,
    lookback: int,
    horizon: int,
    batch_size: int,
    loss_cfg: Optional[Dict],
) -> Tuple[DataLoader, Optional[DataLoader], np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    X_all, y_all, anchor_indices, y_original = make_sequences(
        features_scaled, targets_scaled, lookback, horizon, targets_original=targets_original
    )
    global_anchor_indices = window_def.train_start + anchor_indices

    weights_all: Optional[np.ndarray] = None
    if loss_cfg and y_original is not None:
        weights_all = compute_sample_weights(y_original, loss_cfg)

    val_mask = np.zeros_like(global_anchor_indices, dtype=bool)
    if window_def.val_start is not None:
        val_mask = global_anchor_indices >= window_def.val_start
    train_mask = ~val_mask
    if train_mask.sum() == 0:
        raise ValueError("Training mask is empty; adjust validation window or training length")

    train_weights = weights_all[train_mask] if weights_all is not None else None
    train_dataset = TimeSeriesWindowDataset(X_all[train_mask], y_all[train_mask], train_weights)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    val_loader: Optional[DataLoader] = None
    val_size = int(val_mask.sum())
    if val_size > 0:
        val_weights = weights_all[val_mask] if weights_all is not None else None
        val_dataset = TimeSeriesWindowDataset(X_all[val_mask], y_all[val_mask], val_weights)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    print(
        "[training] Window {0}: train samples {1}, validation samples {2}".format(
            window_def.window_id, len(train_dataset), val_size
        )
    )
    return train_loader, val_loader, global_anchor_indices, X_all, y_all, weights_all

def compute_sample_weights(y_original: np.ndarray, cfg: Dict) -> np.ndarray:
    weights_type = str(cfg.get("type", "target_proportional")).lower()
    if weights_type not in {"target_proportional"}:
        raise ValueError('Unsupported loss_weighting type: {0}'.format(weights_type))
    alpha = float(cfg.get("alpha", 5.0))
    min_weight = float(cfg.get("min_weight", 1.0))
    max_weight = cfg.get("max_weight")
    clipped = np.maximum(y_original, 0.0)
    weights = min_weight + alpha * clipped
    if max_weight is not None:
        weights = np.minimum(weights, float(max_weight))
    return weights.astype(np.float32)


def reduce_loss_tensor(loss_tensor: torch.Tensor, weights: Optional[torch.Tensor]) -> torch.Tensor:
    if weights is not None:
        weights = weights.to(loss_tensor.device)
        weighted = loss_tensor * weights
        denom = torch.clamp(weights.sum(), min=1e-6)
        return weighted.sum() / denom
    if loss_tensor.dim() == 0:
        return loss_tensor
    return loss_tensor.mean()


def optimize_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    training_cfg: Dict,
    device: torch.device,
) -> List[Dict]:
    loss_mode = str(training_cfg.get("loss_function", "mae")).lower()
    criterion = None
    if loss_mode != "hybrid_mae_tweedie":
        criterion = build_loss(loss_mode, training_cfg, reduction="none")
    mae_loss_fn = None
    tweedie_loss_fn = None
    tweedie_weight = float(training_cfg.get("tweedie_weight", 0.2))
    if loss_mode == "hybrid_mae_tweedie":
        mae_loss_fn = nn.L1Loss(reduction="none")
        tweedie_loss_fn = TweedieLoss(power=training_cfg.get("tweedie_power", 1.5), reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=training_cfg["learning_rate"])
    scheduler = None
    if training_cfg.get("reduce_lr_patience"):
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=training_cfg.get("reduce_lr_factor", 0.1),
            patience=training_cfg.get("reduce_lr_patience", 5),
        )

    patience = training_cfg.get("early_stopping_patience", 10)
    min_delta = training_cfg.get("early_stopping_delta", 0.0)
    grad_clip = training_cfg.get("grad_clip", 1.0)
    epochs = int(training_cfg["epochs"])

    best_metric = math.inf
    best_state = deepcopy(model.state_dict())
    epochs_since_improve = 0
    history: List[Dict] = []

    progress = tqdm(range(1, epochs + 1), desc="Epochs", leave=False)
    for epoch in progress:
        model.train()
        train_losses: List[float] = []
        grad_norms: List[float] = []
        for batch in train_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                batch_X, batch_y, batch_w = batch
                batch_w = batch_w.to(device)
            else:
                batch_X, batch_y = batch
                batch_w = None
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_X)
            if loss_mode == "hybrid_mae_tweedie":
                mae_values = mae_loss_fn(preds, batch_y)
                tweedie_values = tweedie_loss_fn(preds, batch_y)
                loss_values = mae_values + tweedie_weight * tweedie_values
            else:
                loss_values = criterion(preds, batch_y)
            loss = reduce_loss_tensor(loss_values, batch_w)
            loss.backward()
            loss_scalar = float(loss.item())
            if math.isnan(loss_scalar):
                optimizer.zero_grad(set_to_none=True)
                continue
            grad_norm = None
            if grad_clip is not None:
                grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip)))
            optimizer.step()
            if grad_norm is not None and not math.isnan(grad_norm):
                grad_norms.append(grad_norm)
            train_losses.append(loss_scalar)

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        val_loss = float("nan")
        if val_loader is not None:
            model.eval()
            val_losses: List[float] = []
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        batch_X, batch_y, batch_w = batch
                        batch_w = batch_w.to(device)
                    else:
                        batch_X, batch_y = batch
                        batch_w = None
                    batch_X = batch_X.to(device)
                    batch_y = batch_y.to(device)
                    preds = model(batch_X)
                    if loss_mode == "hybrid_mae_tweedie":
                        mae_values = mae_loss_fn(preds, batch_y)
                        tweedie_values = tweedie_loss_fn(preds, batch_y)
                        loss_values = mae_values + tweedie_weight * tweedie_values
                    else:
                        loss_values = criterion(preds, batch_y)
                    loss = reduce_loss_tensor(loss_values, batch_w)
                    val_losses.append(float(loss.item()))
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        metric = val_loss if val_loader is not None else train_loss
        if scheduler is not None and not math.isnan(metric):
            scheduler.step(metric)

        progress.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": "-" if math.isnan(val_loss) else f"{val_loss:.4f}",
                "lr": f"{get_current_lr(optimizer):.6f}",
            }
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": None if math.isnan(val_loss) else val_loss,
                "lr": get_current_lr(optimizer),
                "grad_norm": float(np.mean(grad_norms)) if grad_norms else None,
            }
        )

        improved = not math.isnan(metric) and metric < (best_metric - min_delta)
        if improved:
            best_metric = metric
            best_state = deepcopy(model.state_dict())
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if patience and epochs_since_improve >= patience:
            print(
                f"[training] Early stopping triggered at epoch {epoch} with monitored metric {metric:.4f}"
            )
            break

    model.load_state_dict(best_state)
    return history

def forecast_with_model(
    model: nn.Module,
    features_scaled: np.ndarray,
    lookback: int,
    device: torch.device,
) -> np.ndarray:
    if lookback > len(features_scaled):
        raise ValueError("Lookback window longer than available history")
    context = features_scaled[-lookback:]
    context_tensor = torch.from_numpy(context.astype(np.float32)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        preds = model(context_tensor)
    return preds.squeeze(0).cpu().numpy()

def pretrain_model(artifacts_list: List, config: Dict) -> Optional[Dict[str, torch.Tensor]]:
    pre_cfg = config.get("pretraining") or {}
    if not pre_cfg.get("enabled"):
        return None
    if not artifacts_list:
        print("[pretrain] No artifacts supplied for pretraining. Skipping.")
        return None

    model_cfg = config["model"]
    training_cfg = config["training"]
    preprocessing_cfg = config["preprocessing"]
    wf_cfg = config["walk_forward"]

    lookback = int(model_cfg["lookback_weeks"])
    horizon = int(wf_cfg["horizon_weeks"])

    device = resolve_device(training_cfg.get("device", "cpu"))
    set_seed(training_cfg.get("seed"))

    X_batches: List[np.ndarray] = []
    y_batches: List[np.ndarray] = []
    weight_batches: List[np.ndarray] = []

    loss_cfg = pre_cfg.get("loss_weighting", training_cfg.get("loss_weighting"))

    for artifact in artifacts_list:
        df = artifact.data.reset_index(drop=True)
        feature_cols = artifact.feature_columns
        target_col = artifact.target_column
        if len(df) <= lookback + horizon:
            continue
        features = df[feature_cols].to_numpy(dtype=np.float32)
        targets = df[target_col].to_numpy(dtype=np.float32)
        (
            features_scaled,
            targets_scaled,
            _,
            _,
            _,
            _,
        ) = fit_and_transform_window(features, targets, preprocessing_cfg)
        try:
            X_seq, y_seq, _, y_original = make_sequences(
                features_scaled,
                targets_scaled,
                lookback,
                horizon,
                targets_original=targets,
            )
        except ValueError:
            continue
        if X_seq.size == 0:
            continue
        X_batches.append(X_seq)
        y_batches.append(y_seq)
        if loss_cfg and y_original is not None:
            weight_batches.append(compute_sample_weights(y_original, loss_cfg))

    if not X_batches:
        print("[pretrain] Unable to build sequences for pretraining.")
        return None

    X_all = np.concatenate(X_batches, axis=0)
    y_all = np.concatenate(y_batches, axis=0)
    weights_all = np.concatenate(weight_batches, axis=0) if weight_batches else None

    val_split = float(pre_cfg.get("val_split", 0.1))
    total_samples = X_all.shape[0]
    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    if 0.0 < val_split < 1.0 and total_samples > 1:
        split_idx = int(total_samples * (1.0 - val_split))
        split_idx = max(1, min(split_idx, total_samples - 1))
        train_idx = indices[:split_idx]
        val_idx = indices[split_idx:]
    else:
        train_idx = indices
        val_idx = np.array([], dtype=int)

    def _gather(data: np.ndarray, idx: np.ndarray) -> np.ndarray:
        if idx.size == 0:
            return np.empty((0,) + data.shape[1:], dtype=data.dtype)
        return data[idx]

    train_weights = _gather(weights_all, train_idx) if weights_all is not None else None
    val_weights = _gather(weights_all, val_idx) if weights_all is not None else None

    train_dataset = TimeSeriesWindowDataset(_gather(X_all, train_idx), _gather(y_all, train_idx), train_weights)
    batch_size = int(pre_cfg.get("batch_size", training_cfg.get("batch_size", 32)))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    val_loader = None
    if val_idx.size > 0:
        val_dataset = TimeSeriesWindowDataset(_gather(X_all, val_idx), _gather(y_all, val_idx), val_weights)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    pretrain_training_cfg = deepcopy(training_cfg)
    overrides = {
        "epochs": pre_cfg.get("epochs"),
        "learning_rate": pre_cfg.get("learning_rate"),
        "early_stopping_patience": pre_cfg.get("early_stopping_patience"),
        "reduce_lr_patience": pre_cfg.get("reduce_lr_patience"),
        "reduce_lr_factor": pre_cfg.get("reduce_lr_factor"),
        "grad_clip": pre_cfg.get("grad_clip"),
        "loss_weighting": loss_cfg,
        "batch_size": batch_size,
    }
    for key, value in overrides.items():
        if value is not None:
            pretrain_training_cfg[key] = value

    model = create_model_from_config(
        input_size=X_all.shape[2],
        horizon=horizon,
        model_cfg=model_cfg,
    ).to(device)

    print("[pretrain] Starting pretraining with {0} samples (val: {1}).".format(train_idx.size, val_idx.size))
    optimize_model(model, train_loader, val_loader, pretrain_training_cfg, device)

    state_dict = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    return state_dict


def run_walk_forward(
    artifacts,
    config: Dict,
    pretrained_state_dict: Optional[Dict[str, torch.Tensor]] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    df = artifacts.data.reset_index(drop=True)
    feature_cols = artifacts.feature_columns
    target_col = artifacts.target_column
    date_col = artifacts.date_column

    features = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.float32)
    dates = df[date_col].reset_index(drop=True)

    wf_cfg = config["walk_forward"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    preprocessing_cfg = config["preprocessing"]

    lookback = int(model_cfg["lookback_weeks"])
    horizon = int(wf_cfg["horizon_weeks"])

    windows = generate_window_definitions(len(df), wf_cfg, lookback, horizon)

    device = resolve_device(training_cfg.get("device", "cpu"))
    set_seed(training_cfg.get("seed"))

    prediction_frames: List[pd.DataFrame] = []
    window_histories: List[Dict] = []

    test_start = len(df) - int(wf_cfg["test_window_weeks"])
    test_end = len(df) - 1

    for window in windows:
        print(
            f"[training] Window {window.window_id}: train idx {window.train_start}..{window.train_end}, "
            f"forecast idx {window.forecast_start}..{window.forecast_end}"
        )

        window_features = features[window.train_start : window.train_end + 1].copy()
        window_targets = targets[window.train_start : window.train_end + 1].copy()

        (
            features_scaled,
            targets_scaled,
            target_transform,
            target_scaler,
            feature_scaler,
            window_target_scaler,
        ) = fit_and_transform_window(
            window_features,
            window_targets,
            preprocessing_cfg,
        )

        loss_cfg = training_cfg.get("loss_weighting")
        (
            train_loader,
            val_loader,
            anchor_indices,
            X_all,
            y_all,
            _weights_all,
        ) = create_window_dataloaders(
            window,
            features_scaled,
            targets_scaled,
            window_targets,
            lookback,
            horizon,
            int(training_cfg["batch_size"]),
            loss_cfg,
        )

        sample_feature_shape = train_loader.dataset[0][0].shape
        sample_target_shape = train_loader.dataset[0][1].shape
        print(
            f"[training] Window {window.window_id}: feature tensor {sample_feature_shape}, target tensor {sample_target_shape}"
        )

        model = create_model_from_config(
            input_size=features_scaled.shape[1],
            horizon=horizon,
            model_cfg=model_cfg,
        ).to(device)
        if pretrained_state_dict is not None:
            missing, unexpected = model.load_state_dict(pretrained_state_dict, strict=False)
            if missing or unexpected:
                print("[pretrain] State dict load warnings - missing: {0}, unexpected: {1}".format(missing, unexpected))

        history = optimize_model(model, train_loader, val_loader, training_cfg, device)

        window_histories.append(
            {
                "window_id": window.window_id,
                "train_range": (window.train_start, window.train_end),
                "val_start": window.val_start,
                "forecast_range": (window.forecast_start, window.forecast_end),
                "train_samples": len(train_loader.dataset),
                "val_samples": len(val_loader.dataset) if val_loader is not None else 0,
                "feature_scaler": feature_scaler.__class__.__name__ if feature_scaler is not None else "None",
                "target_scaler": target_scaler.__class__.__name__ if target_scaler is not None else "None",
                "window_target_scaler": window_target_scaler.__class__.__name__ if window_target_scaler is not None else "None",
                "loss_weighting": loss_cfg.copy() if isinstance(loss_cfg, dict) else loss_cfg,
                "history": history,
            }
        )

        preds_scaled = forecast_with_model(model, features_scaled, lookback, device)
        preds_intermediate = preds_scaled
        if window_target_scaler is not None:
            preds_intermediate = window_target_scaler.inverse_transform(
                preds_intermediate.reshape(-1, 1)
            ).reshape(-1)
        if target_scaler is not None:
            preds_transformed = target_scaler.inverse_transform(
                preds_intermediate.reshape(-1, 1)
            ).reshape(-1)
        else:
            preds_transformed = preds_intermediate
        preds_original = target_transform.inverse_transform(preds_transformed)
        ensure_no_nan(preds_original, "predictions after inverse transformations")

        forecast_slice = slice(window.forecast_start, window.forecast_end + 1)
        y_true = targets[forecast_slice]
        forecast_dates = dates.iloc[forecast_slice].reset_index(drop=True)

        mask = (forecast_dates.index + window.forecast_start >= test_start) & (
            forecast_dates.index + window.forecast_start <= test_end
        )
        if not mask.all():
            preds_original = preds_original[mask]
            y_true = y_true[mask]
            forecast_dates = forecast_dates[mask]

        pred_df = pd.DataFrame(
            {
                date_col: forecast_dates,
                "y_true": y_true,
                "y_pred": preds_original,
                "window_id": window.window_id,
                "train_start_date": dates.iloc[window.train_start],
                "train_end_date": dates.iloc[window.train_end],
                "val_start_date": dates.iloc[window.val_start] if window.val_start is not None else pd.NaT,
                "forecast_origin_date": dates.iloc[window.train_end],
            }
        )
        pred_df["horizon"] = np.arange(1, len(pred_df) + 1)
        prediction_frames.append(pred_df)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.sort_values(by=date_col, inplace=True)
    predictions.reset_index(drop=True, inplace=True)

    return predictions, window_histories


__all__ = [
    "pretrain_model",
    "run_walk_forward",
    "fit_and_transform_window",
    "create_window_dataloaders",
    "compute_sample_weights",
    "reduce_loss_tensor",
    "optimize_model",
    "forecast_with_model",
]
