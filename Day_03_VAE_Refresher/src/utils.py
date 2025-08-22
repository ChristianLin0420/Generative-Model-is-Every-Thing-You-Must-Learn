"""
Utility functions for VAE training and evaluation.
Includes: seed setting, device management, checkpoint I/O, image grid creation, logging.
"""

import random
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from PIL import Image
import matplotlib.pyplot as plt
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = "auto") -> torch.device:
    """Get the appropriate device for computation."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    
    console.print(f"Using device: {device}")
    if device.type == "cuda":
        console.print(f"GPU: {torch.cuda.get_device_name()}")
        console.print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    return device


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    filepath: str,
    **kwargs
) -> None:
    """Save model checkpoint with training state."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint and restore training state."""
    if device is None:
        device = torch.device("cpu")
    
    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint


def make_image_grid(
    images: torch.Tensor,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
    pad_value: float = 0.0
) -> np.ndarray:
    """Create an image grid from a batch of images."""
    if normalize and value_range is None:
        # Auto-detect value range
        value_range = (images.min().item(), images.max().item())
    
    grid = vutils.make_grid(
        images,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range,
        pad_value=pad_value,
        padding=2
    )
    
    # Convert to numpy array (HWC format)
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    
    # Ensure values are in [0, 1] range for display
    if grid_np.max() > 1.0 or grid_np.min() < 0.0:
        grid_np = np.clip(grid_np, 0.0, 1.0)
    
    return grid_np


def save_image_grid(
    images: torch.Tensor,
    filepath: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None
) -> None:
    """Save an image grid to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    grid_np = make_image_grid(images, nrow, normalize, value_range)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid_np)
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"Saved image grid: {filepath}")


def denormalize_images(
    images: torch.Tensor,
    mean: List[float],
    std: List[float]
) -> torch.Tensor:
    """Denormalize images from standard normalization."""
    mean = torch.tensor(mean).view(-1, 1, 1).to(images.device)
    std = torch.tensor(std).view(-1, 1, 1).to(images.device)
    return images * std + mean


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Setup rich logger with optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Console handler with rich formatting
    console_handler = RichHandler(
        console=console,
        show_time=True,
        show_path=True,
        markup=True
    )
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module) -> None:
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module) -> None:
        """Apply EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module) -> None:
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class MetricsTracker:
    """Track and compute running averages of metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics_dict: Dict[str, float], batch_size: int = 1):
        """Update metrics with new values."""
        for key, value in metrics_dict.items():
            if key not in self.metrics:
                self.metrics[key] = 0.0
                self.counts[key] = 0
            
            self.metrics[key] = (
                (self.metrics[key] * self.counts[key] + value * batch_size) /
                (self.counts[key] + batch_size)
            )
            self.counts[key] += batch_size
    
    def get_averages(self) -> Dict[str, float]:
        """Get current average values."""
        return self.metrics.copy()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}


def create_progress_bar(description: str = "Processing") -> Progress:
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TimeRemainingColumn(),
        console=console
    )