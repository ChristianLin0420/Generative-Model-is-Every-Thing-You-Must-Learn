"""Utility functions for the forward diffusion process."""

import random
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from typing import List, Optional, Union


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get torch device, fallback to CPU if CUDA not available."""
    if device_str.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    return torch.device("cpu")


def get_dtype(dtype_str: str = "float32") -> torch.dtype:
    """Get torch dtype from string."""
    dtype_map = {
        "float32": torch.float32,
        "float64": torch.float64,
        "half": torch.half,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str, torch.float32)


def normalize_to_neg_one_to_one(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [0, 1] to [-1, 1] range."""
    return tensor * 2.0 - 1.0


def unnormalize_to_zero_to_one(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from [-1, 1] to [0, 1] range."""
    return (tensor + 1.0) / 2.0


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image. Assumes tensor is in [0, 1] range."""
    if tensor.dim() == 4:  # Batch dimension
        tensor = tensor[0]
    if tensor.dim() == 3:  # CHW format
        tensor = tensor.permute(1, 2, 0)
    
    # Clamp to [0, 1] and convert to numpy
    tensor = torch.clamp(tensor, 0.0, 1.0)
    if tensor.shape[-1] == 1:  # Grayscale
        tensor = tensor.squeeze(-1)
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode='L')
    else:  # RGB
        array = (tensor.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode='RGB')


def save_image_grid(
    images: torch.Tensor,
    path: Union[str, Path],
    nrows: Optional[int] = None,
    normalize: bool = True,
    title: Optional[str] = None
) -> None:
    """Save a grid of images.
    
    Args:
        images: Tensor of shape (B, C, H, W)
        path: Path to save the image
        nrows: Number of rows in grid
        normalize: Whether to normalize from [-1,1] to [0,1]
        title: Optional title for the plot
    """
    if normalize:
        images = unnormalize_to_zero_to_one(images)
    
    batch_size = images.shape[0]
    if nrows is None:
        nrows = int(np.sqrt(batch_size))
    ncols = (batch_size + nrows - 1) // nrows
    
    # Create grid
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    if nrows == 1 and ncols == 1:
        axes = [axes]
    elif nrows == 1 or ncols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(batch_size):
        img = images[i]
        if img.shape[0] == 1:  # Grayscale
            img = img.squeeze(0)
            axes[i].imshow(img.cpu().numpy(), cmap='gray')
        else:  # RGB
            img = img.permute(1, 2, 0)
            axes[i].imshow(img.cpu().numpy())
        axes[i].axis('off')
    
    # Hide remaining axes
    for i in range(batch_size, len(axes)):
        axes[i].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_trajectory_grid(
    trajectories: torch.Tensor,
    timesteps: List[int],
    path: Union[str, Path],
    normalize: bool = True,
    title: Optional[str] = None
) -> None:
    """Save a grid showing trajectory of images across timesteps.
    
    Args:
        trajectories: Tensor of shape (B, T, C, H, W) where T is number of timesteps
        timesteps: List of timestep values for column labels
        path: Path to save the image
        normalize: Whether to normalize from [-1,1] to [0,1]
        title: Optional title for the plot
    """
    if normalize:
        trajectories = unnormalize_to_zero_to_one(trajectories)
    
    batch_size, T, C, H, W = trajectories.shape
    
    fig, axes = plt.subplots(batch_size, T, figsize=(T * 2, batch_size * 2))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        for t_idx, t in enumerate(timesteps):
            img = trajectories[i, t_idx]
            if C == 1:  # Grayscale
                img = img.squeeze(0)
                axes[i, t_idx].imshow(img.cpu().numpy(), cmap='gray')
            else:  # RGB
                img = img.permute(1, 2, 0)
                axes[i, t_idx].imshow(img.cpu().numpy())
            
            if i == 0:  # Add timestep labels to top row
                axes[i, t_idx].set_title(f't={t}', fontsize=10)
            axes[i, t_idx].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def save_animation(
    trajectory: torch.Tensor,
    path: Union[str, Path],
    duration: float = 0.1,
    normalize: bool = True,
    loop: int = 0
) -> None:
    """Save trajectory as animated GIF.
    
    Args:
        trajectory: Tensor of shape (T, C, H, W)
        path: Path to save the animation
        duration: Duration between frames in seconds
        normalize: Whether to normalize from [-1,1] to [0,1]
        loop: Number of loops (0 = infinite)
    """
    if normalize:
        trajectory = unnormalize_to_zero_to_one(trajectory)
    
    frames = []
    for t in range(trajectory.shape[0]):
        img = trajectory[t]
        if img.shape[0] == 1:  # Grayscale
            img = img.squeeze(0).cpu().numpy()
            frame = (img * 255).astype(np.uint8)
        else:  # RGB
            img = img.permute(1, 2, 0).cpu().numpy()
            frame = (img * 255).astype(np.uint8)
        frames.append(frame)
    
    # Save as GIF
    imageio.mimsave(
        path,
        frames,
        duration=duration,
        loop=loop
    )