"""
Utility functions for Day 1: Gaussian Noise Basics
"""

import random
import time
from functools import wraps
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torchvision.utils as vutils
from PIL import Image


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name: Optional[str] = None) -> torch.device:
    """Get the appropriate device (cuda/cpu)."""
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device_name = "cpu"
    
    device = torch.device(device_name)
    print(f"Using device: {device}")
    return device


def save_image_grid(
    tensor: torch.Tensor,
    path: Union[str, Path],
    nrow: int = 8,
    normalize: bool = True,
    range: Optional[tuple] = None,
    pad_value: float = 0.0
) -> None:
    """Save a grid of images from tensor."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save using torchvision
    vutils.save_image(
        tensor,
        path,
        nrow=nrow,
        normalize=normalize,
        value_range=range,
        pad_value=pad_value
    )
    print(f"Saved image grid to {path}")


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image."""
    # Denormalize if needed and convert to [0, 255]
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # Remove batch dimension
    
    # Ensure values are in [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy and transpose to HWC
    if tensor.dim() == 3:  # CHW -> HWC
        np_array = tensor.permute(1, 2, 0).cpu().numpy()
    else:  # HW (grayscale)
        np_array = tensor.cpu().numpy()
    
    # Convert to [0, 255] uint8
    np_array = (np_array * 255).astype(np.uint8)
    
    # Convert to PIL
    if np_array.ndim == 3:
        return Image.fromarray(np_array)
    else:
        return Image.fromarray(np_array, mode='L')


def timeit(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def create_output_dirs(base_dir: Union[str, Path]) -> None:
    """Create output directory structure."""
    base_dir = Path(base_dir)
    dirs = [
        base_dir / "grids",
        base_dir / "animations", 
        base_dir / "logs"
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Created output directories under {base_dir}")


def get_memory_usage() -> str:
    """Get current GPU memory usage if available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "GPU not available"