from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

DEFAULT_SEED = 1337


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set seeds for reproducibility and print the seed once."""

    print(f"[debug] Using seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def preview_batch(dataloader: DataLoader, split: str, n: int = 5) -> None:
    """Print basic info about one batch from a dataloader."""

    batch = next(iter(dataloader))
    inputs, targets = batch[:2]
    print(
        f"[preview] split={split}, inputs shape={tuple(inputs.shape)}, targets shape={tuple(targets.shape)}"
    )
    print("[preview] first targets:", targets[:n])


def toggle_dropout(model: nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            if enabled:
                if hasattr(module, "_p_backup"):
                    module.p = module._p_backup
            else:
                module._p_backup = module.p
                module.p = 0.0


def freeze_batchnorm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d):
            module.eval()
            module.track_running_stats = False


def print_batchnorm_stats(model: nn.Module, tag: str) -> None:
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm1d):
            print(
                f"[bn stats] {tag} {name}: mean={module.running_mean.mean():.4f}, "
                f"var={module.running_var.mean():.4f}"
            )


def per_group_loss(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    group_index: int = 2,
) -> Dict[int, float]:
    model.eval()
    losses: Dict[int, list] = defaultdict(list)
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch[:2]
            groups = batch[group_index] if len(batch) > group_index else None
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs)
            batch_loss = criterion(preds, targets).detach().cpu().numpy()
            if groups is None:
                losses[0].extend(batch_loss.tolist())
            else:
                for g, l in zip(groups, batch_loss):
                    losses[int(g)].append(float(l))
    return {g: float(np.mean(v)) for g, v in losses.items()}
