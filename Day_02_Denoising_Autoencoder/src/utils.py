"""
Utility functions for Day 2: Denoising Autoencoder
"""

import json
import random
import time
from functools import wraps
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torchvision.utils as vutils
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

console = Console()


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
    console.print(f"[green]Set seed to {seed}[/green]")


def get_device(device_name: Optional[str] = None) -> torch.device:
    """Get the appropriate device."""
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        console.print("[yellow]CUDA requested but not available, falling back to CPU[/yellow]")
        device_name = "cpu"
    
    device = torch.device(device_name)
    console.print(f"[green]Using device: {device}[/green]")
    return device


def create_output_dirs(base_dir: Union[str, Path]) -> None:
    """Create output directory structure."""
    base_dir = Path(base_dir)
    dirs = [
        base_dir / "ckpts",
        base_dir / "logs", 
        base_dir / "grids",
        base_dir / "panels",
        base_dir / "reports"
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    console.print(f"[green]Created output directories under {base_dir}[/green]")


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
    
    vutils.save_image(
        tensor,
        path,
        nrow=nrow,
        normalize=normalize,
        value_range=range,
        pad_value=pad_value
    )
    console.print(f"[blue]Saved image grid to {path}[/blue]")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: Union[str, Path],
    ema_model: Optional[torch.nn.Module] = None,
    scheduler: Optional = None,
    config: Optional[DictConfig] = None
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': OmegaConf.to_container(config) if config else None
    }
    
    if ema_model is not None:
        checkpoint['ema_state_dict'] = ema_model.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    console.print(f"[green]Saved checkpoint to {path}[/green]")


def load_checkpoint(
    path: Union[str, Path],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[torch.nn.Module] = None,
    scheduler: Optional = None,
    device: Optional[torch.device] = None
) -> Dict:
    """Load training checkpoint."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device or 'cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if ema_model is not None and 'ema_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    console.print(f"[green]Loaded checkpoint from {path}[/green]")
    return checkpoint


def save_metrics(metrics: Dict, path: Union[str, Path]) -> None:
    """Save metrics to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    console.print(f"[blue]Saved metrics to {path}[/blue]")


def load_config(config_path: str) -> DictConfig:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    console.print(f"[green]Loaded config from {config_path}[/green]")
    return config


def timeit(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        console.print(f"[dim]{func.__name__} took {end - start:.2f} seconds[/dim]")
        return result
    return wrapper


def print_model_summary(model: torch.nn.Module, input_shape: tuple = (1, 28, 28)) -> None:
    """Print model parameter summary."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    table = Table(title="Model Summary")
    table.add_column("Parameter", style="cyan")
    table.add_column("Count", style="magenta")
    
    table.add_row("Total Parameters", f"{total_params:,}")
    table.add_row("Trainable Parameters", f"{trainable_params:,}")
    table.add_row("Input Shape", str(input_shape))
    
    console.print(table)


def get_memory_usage() -> str:
    """Get current GPU memory usage if available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB"
    return "GPU not available"


class EMAModel:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, device: Optional[torch.device] = None):
        self.decay = decay
        self.device = device
        
        # Create a deep copy of the model instead of reconstructing
        import copy
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        
        if device is not None:
            self.ema_model.to(device)
        
        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
    
    def update(self, model: torch.nn.Module) -> None:
        """Update EMA model parameters."""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)
    
    def state_dict(self) -> Dict:
        """Get EMA model state dict."""
        return self.ema_model.state_dict()
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load EMA model state dict."""
        self.ema_model.load_state_dict(state_dict)
    
    def __call__(self, *args, **kwargs):
        """Forward pass through EMA model."""
        return self.ema_model(*args, **kwargs)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)