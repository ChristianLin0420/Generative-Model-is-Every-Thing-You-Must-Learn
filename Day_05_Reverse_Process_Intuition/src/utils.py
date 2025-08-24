"""Utility functions for DDPM implementation

Includes:
- Random seed management
- Device detection
- Checkpoint I/O
- Exponential Moving Average (EMA)
- Grid and animation helpers
"""

import os
import random
import torch
import numpy as np
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import imageio
from PIL import Image


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ema_model: Optional[torch.nn.Module] = None,
    **kwargs
) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
        **kwargs
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ema_model: Optional[torch.nn.Module] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint."""
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    if ema_model is not None and "ema_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_state_dict"])
    
    return checkpoint


class EMA:
    """Exponential Moving Average for model parameters."""
    
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: torch.nn.Module) -> None:
        """Update EMA parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model: torch.nn.Module) -> None:
        """Apply EMA parameters to model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: torch.nn.Module) -> None:
        """Restore original parameters."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state dict."""
        return self.shadow
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load EMA state dict."""
        self.shadow = state_dict


def save_image_grid(
    images: torch.Tensor,
    path: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[tuple] = None,
    pad_value: float = 0,
) -> None:
    """Save a grid of images."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    grid = make_grid(
        images, 
        nrow=nrow, 
        normalize=normalize, 
        value_range=value_range,
        pad_value=pad_value
    )
    
    # Convert to PIL and save
    grid_np = grid.permute(1, 2, 0).cpu().numpy()
    if normalize:
        grid_np = (grid_np * 255).astype(np.uint8)
    
    if grid_np.shape[2] == 1:
        grid_np = grid_np.squeeze(2)
        Image.fromarray(grid_np, 'L').save(path)
    else:
        Image.fromarray(grid_np, 'RGB').save(path)


def save_animation(
    image_sequences: List[torch.Tensor],
    path: str,
    fps: int = 10,
    duration: Optional[float] = None,
) -> None:
    """Save animation from sequence of image tensors."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    frames = []
    for images in image_sequences:
        # Convert to numpy
        if images.dim() == 4:  # Batch of images, take first
            img = images[0]
        else:
            img = images
        
        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)
        
        if img_np.shape[2] == 1:
            img_np = img_np.squeeze(2)
        
        frames.append(img_np)
    
    # Save as GIF or MP4
    if path.endswith('.gif'):
        imageio.mimsave(path, frames, fps=fps, duration=duration)
    elif path.endswith('.mp4'):
        imageio.mimsave(path, frames, fps=fps, quality=8)
    else:
        # Default to GIF
        path = path + '.gif'
        imageio.mimsave(path, frames, fps=fps, duration=duration)


def plot_loss_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    title: str = "Training Loss"
) -> None:
    """Plot and optionally save loss curves."""
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    
    if val_losses is not None:
        val_epochs = range(1, len(val_losses) + 1)
        plt.plot(val_epochs, val_losses, 'r-', label='Val Loss')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }