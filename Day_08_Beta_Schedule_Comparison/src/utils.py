"""
Utility functions for seeding, device management, checkpoint I/O, timers, 
grid/gif savers, and configuration loading.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import os
import time
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from PIL import Image
import torchvision.utils as vutils
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Get appropriate device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


class Timer:
    """Simple timer context manager."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start_time
        if self.name:
            print(f"{self.name}: {elapsed:.3f}s")
        else:
            print(f"Elapsed: {elapsed:.3f}s")


class EMAModel:
    """Exponential Moving Average model wrapper."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update EMA parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply EMA parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def save_checkpoint(
    model: nn.Module,
    ema_model: Optional[EMAModel],
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    loss: float,
    filepath: Union[str, Path]
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'loss': loss,
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if ema_model is not None:
        checkpoint['ema_state_dict'] = ema_model.shadow
    
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    ema_model: Optional[EMAModel] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint."""
    if device is None:
        device = torch.device('cpu')
    
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if ema_model is not None and 'ema_state_dict' in checkpoint:
        ema_model.shadow = checkpoint['ema_state_dict']
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def save_image_grid(
    images: torch.Tensor,
    filepath: Union[str, Path],
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[tuple] = None
) -> None:
    """Save a grid of images."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if normalize and value_range is None:
        value_range = (-1, 1) if images.min() < 0 else (0, 1)
    
    grid = vutils.make_grid(
        images, 
        nrow=nrow, 
        normalize=normalize, 
        value_range=value_range,
        pad_value=1.0
    )
    
    vutils.save_image(grid, filepath)


def tensor_to_pil(tensor: torch.Tensor, normalize: bool = True) -> Image.Image:
    """Convert tensor to PIL Image."""
    if normalize:
        tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    if tensor.dim() == 4:  # Batch
        tensor = tensor[0]
    
    tensor = tensor.cpu()
    if tensor.shape[0] == 1:  # Grayscale
        tensor = tensor.squeeze(0)
        array = tensor.numpy()
        return Image.fromarray((array * 255).astype(np.uint8), mode='L')
    else:  # RGB
        array = tensor.permute(1, 2, 0).numpy()
        return Image.fromarray((array * 255).astype(np.uint8), mode='RGB')


def save_gif(
    frames: List[torch.Tensor],
    filepath: Union[str, Path],
    duration: int = 100,
    normalize: bool = True
) -> None:
    """Save list of tensors as animated GIF."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    pil_frames = []
    for frame in frames:
        pil_frame = tensor_to_pil(frame, normalize=normalize)
        pil_frames.append(pil_frame)
    
    pil_frames[0].save(
        filepath,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0
    )


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file with base config support."""
    config = OmegaConf.load(config_path)
    
    # Handle base config inheritance
    if '_base_' in config:
        base_path = Path(config_path).parent / config['_base_']
        base_config = OmegaConf.load(base_path)
        # Merge base config with current config (current takes precedence)
        config = OmegaConf.merge(base_config, config)
        # Remove the _base_ key
        del config['_base_']
    
    return OmegaConf.to_container(config, resolve=True)


def setup_logging_dir(run_dir: Union[str, Path]) -> Path:
    """Create and return logging directory structure."""
    run_dir = Path(run_dir)
    
    subdirs = ['ckpts', 'logs', 'curves', 'grids', 'animations']
    for subdir in subdirs:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return run_dir


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_model_size(model: nn.Module) -> float:
    """Compute model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024**2
    return size_mb


def save_metrics_csv(metrics: Dict[str, List[float]], filepath: Union[str, Path]) -> None:
    """Save metrics dictionary to CSV file."""
    import pandas as pd
    
    df = pd.DataFrame(metrics)
    df.to_csv(filepath, index=False)


def normalize_tensor(tensor: torch.Tensor, mode: str = "minus_one_one") -> torch.Tensor:
    """Normalize tensor to specified range."""
    if mode == "minus_one_one":
        return tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1]
    elif mode == "zero_one":
        return tensor  # Keep [0, 1]
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")


def denormalize_tensor(tensor: torch.Tensor, mode: str = "minus_one_one") -> torch.Tensor:
    """Denormalize tensor from specified range."""
    if mode == "minus_one_one":
        return (tensor + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    elif mode == "zero_one":
        return tensor  # Keep [0, 1]
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")
