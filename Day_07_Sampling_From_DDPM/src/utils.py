"""
Utility functions for DDPM sampling
- Seed management
- Device handling
- Checkpoint I/O
- Grid and animation savers
- Timer utilities
"""

import os
import time
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio
from typing import List, Optional, Union, Dict, Any
from pathlib import Path


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get appropriate device, with fallback logic."""
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        device = torch.device(device_str)
        # Test if device is actually available
        if device.type == 'cuda':
            torch.zeros(1).to(device)
        return device
    except (RuntimeError, AssertionError):
        print(f"Warning: {device_str} not available, falling back to CPU")
        return torch.device("cpu")


def save_checkpoint(checkpoint: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save checkpoint to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Union[str, Path], device: Optional[torch.device] = None) -> Dict[str, Any]:
    """Load checkpoint from disk."""
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def make_grid(images: torch.Tensor, nrow: int = 8, padding: int = 2, 
              normalize: bool = True, value_range: Optional[tuple] = None) -> torch.Tensor:
    """
    Create a grid of images.
    
    Args:
        images: Tensor of shape (N, C, H, W)
        nrow: Number of images per row
        padding: Padding between images
        normalize: Whether to normalize to [0, 1]
        value_range: Range for normalization
    
    Returns:
        Grid tensor of shape (C, grid_H, grid_W)
    """
    if normalize:
        if value_range is None:
            images = (images - images.min()) / (images.max() - images.min())
        else:
            min_val, max_val = value_range
            images = (images - min_val) / (max_val - min_val)
            images = torch.clamp(images, 0, 1)
    
    N, C, H, W = images.shape
    ncol = int(np.ceil(N / nrow))
    
    # Create grid tensor
    grid_h = ncol * H + (ncol + 1) * padding
    grid_w = nrow * W + (nrow + 1) * padding
    grid = torch.ones(C, grid_h, grid_w)
    
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        
        y_start = row * (H + padding) + padding
        y_end = y_start + H
        x_start = col * (W + padding) + padding
        x_end = x_start + W
        
        grid[:, y_start:y_end, x_start:x_end] = img
    
    return grid


def save_image_grid(images: torch.Tensor, path: Union[str, Path], 
                   nrow: int = 8, **kwargs) -> None:
    """Save a grid of images to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    grid = make_grid(images, nrow=nrow, **kwargs)
    
    # Convert to PIL format
    if grid.shape[0] == 1:  # Grayscale
        grid_np = grid.squeeze(0).numpy()
        img = Image.fromarray((grid_np * 255).astype(np.uint8), mode='L')
    else:  # RGB
        grid_np = grid.permute(1, 2, 0).numpy()
        img = Image.fromarray((grid_np * 255).astype(np.uint8), mode='RGB')
    
    img.save(path)


def save_animation(frames: List[torch.Tensor], path: Union[str, Path], 
                  fps: int = 10, duration: Optional[float] = None) -> None:
    """
    Save a list of image tensors as GIF or MP4 animation.
    
    Args:
        frames: List of tensors, each of shape (C, H, W)
        path: Output path (.gif or .mp4)
        fps: Frames per second
        duration: Duration per frame (overrides fps if provided)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert frames to numpy arrays
    np_frames = []
    for frame in frames:
        if frame.shape[0] == 1:  # Grayscale
            frame_np = frame.squeeze(0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
        else:  # RGB
            frame_np = frame.permute(1, 2, 0).numpy()
            frame_np = (frame_np * 255).astype(np.uint8)
        np_frames.append(frame_np)
    
    # Save animation
    if str(path).endswith('.gif'):
        if duration is None:
            duration = 1.0 / fps
        imageio.mimsave(path, np_frames, duration=duration)
    elif str(path).endswith('.mp4'):
        imageio.mimsave(path, np_frames, fps=fps)
    else:
        raise ValueError(f"Unsupported format: {path}. Use .gif or .mp4")


class Timer:
    """Simple timer utility for benchmarking."""
    
    def __init__(self):
        self.times = {}
        self.start_times = {}
    
    def start(self, name: str = "default") -> None:
        """Start timing."""
        self.start_times[name] = time.time()
    
    def stop(self, name: str = "default") -> float:
        """Stop timing and return elapsed time."""
        if name not in self.start_times:
            raise ValueError(f"Timer '{name}' was not started")
        
        elapsed = time.time() - self.start_times[name]
        if name not in self.times:
            self.times[name] = []
        self.times[name].append(elapsed)
        
        del self.start_times[name]
        return elapsed
    
    def get_average(self, name: str = "default") -> float:
        """Get average time for a named timer."""
        if name not in self.times or len(self.times[name]) == 0:
            return 0.0
        return np.mean(self.times[name])
    
    def get_total(self, name: str = "default") -> float:
        """Get total time for a named timer."""
        if name not in self.times:
            return 0.0
        return np.sum(self.times[name])
    
    def reset(self, name: Optional[str] = None) -> None:
        """Reset timer(s)."""
        if name is None:
            self.times.clear()
            self.start_times.clear()
        else:
            if name in self.times:
                del self.times[name]
            if name in self.start_times:
                del self.start_times[name]


def create_output_dirs(base_dir: Union[str, Path]) -> None:
    """Create all necessary output directories."""
    base_path = Path(base_dir)
    dirs = [
        base_path / "grids",
        base_path / "animations", 
        base_path / "curves",
        base_path / "logs",
        base_path / "reports"
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)


def format_time(seconds: float) -> str:
    """Format time duration as human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}h {m}m {s}s"


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
        }
    return {'allocated': 0.0, 'reserved': 0.0}


def log_system_info(device: torch.device) -> str:
    """Log system and device information."""
    info = f"Device: {device}\n"
    
    if device.type == 'cuda':
        info += f"CUDA Device: {torch.cuda.get_device_name(device)}\n"
        mem_info = get_memory_usage()
        info += f"GPU Memory: {mem_info['allocated']:.2f}GB allocated, {mem_info['reserved']:.2f}GB reserved\n"
    
    info += f"PyTorch Version: {torch.__version__}\n"
    info += f"Random Seed Set: {torch.initial_seed()}\n"
    
    return info
