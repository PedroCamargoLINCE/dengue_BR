"""Utility functions."""

import torch
import numpy as np
import random
import os


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def device() -> torch.device:
    """Get the appropriate device (GPU if available, else CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_device_info() -> str:
    """Get detailed device information."""
    device_info = f"Device: {device()}"
    if torch.cuda.is_available():
        device_info += f" ({torch.cuda.get_device_name()})"
        device_info += f"\nCUDA Version: {torch.version.cuda}"
        device_info += f"\nGPU Count: {torch.cuda.device_count()}"
    return device_info


def move_to_device(obj, device):
    """Move tensor or model to device."""
    if hasattr(obj, 'to'):
        return obj.to(device)
    return obj


def setup_logging():
    """Setup basic logging configuration."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
        ]
    )
    
    return logging.getLogger(__name__)


def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def save_model(model, path: str):
    """Save PyTorch model."""
    ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)


def load_model(model_class, model_kwargs: dict, path: str):
    """Load PyTorch model."""
    model = model_class(**model_kwargs)
    model.load_state_dict(torch.load(path, map_location=device()))
    return model


def get_model_size(model):
    """Get number of parameters in a PyTorch model."""
    return sum(p.numel() for p in model.parameters())


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def check_gpu_memory():
    """Check GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        cached = torch.cuda.memory_reserved() / 1024**3  # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB"
    else:
        return "GPU not available"


def clear_gpu_memory():
    """Clear GPU memory cache."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_gpu_info():
    """Print comprehensive GPU information."""
    if torch.cuda.is_available():
        print(f"🔧 GPU Information:")
        print(f"   Device: {torch.cuda.get_device_name()}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        print(f"   Current Device: {torch.cuda.current_device()}")
        print(f"   Memory: {check_gpu_memory()}")
    else:
        print("❌ CUDA not available - using CPU")