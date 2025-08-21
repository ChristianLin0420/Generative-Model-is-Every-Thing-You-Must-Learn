"""
Visualization utilities for Day 1: Gaussian Noise Basics
"""

from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.utils as vutils
from PIL import Image
import imageio

from .utils import tensor_to_pil, save_image_grid


def make_progressive_grid(
    batch: torch.Tensor,
    sigmas: List[float],
    path: Union[str, Path],
    nrow: int = 8,
    normalize_range: tuple = (0, 1),
    add_labels: bool = True
) -> None:
    """
    Create a grid showing the same samples under increasing noise levels.
    
    Args:
        batch: Input batch [B, C, H, W]
        sigmas: List of noise levels
        path: Output path for the grid image
        nrow: Number of images per row
        normalize_range: Range for denormalization
        add_labels: Whether to add sigma labels
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    device = batch.device
    generator = torch.Generator(device=device).manual_seed(42)
    
    # Select subset of images if batch is too large
    num_images = min(nrow, batch.size(0))
    selected_batch = batch[:num_images]
    
    # Create grid data: [num_sigmas * num_images, C, H, W]
    grid_data = []
    
    for sigma in sigmas:
        from .noise import add_gaussian_noise
        
        # Add noise to all selected images
        if normalize_range == (0, 1):
            clip_range = (0, 1)
        elif normalize_range == (-1, 1):
            clip_range = (-1, 1)
        else:
            clip_range = None
            
        noisy_batch = add_gaussian_noise(
            selected_batch, sigma, clip_range, generator
        )
        grid_data.append(noisy_batch)
    
    # Concatenate all noise levels
    grid_tensor = torch.cat(grid_data, dim=0)
    
    # Denormalize for visualization
    if normalize_range == (-1, 1):
        grid_tensor = (grid_tensor + 1) / 2  # [-1, 1] -> [0, 1]
    
    # Save grid
    save_image_grid(
        grid_tensor,
        path,
        nrow=num_images,  # All images for one sigma level in one row
        normalize=False,  # Already normalized
        range=(0, 1)
    )
    
    # Add text labels if requested
    if add_labels:
        _add_sigma_labels_to_grid(path, sigmas, num_images)
    
    print(f"Saved progressive noise grid to {path}")


def make_animation(
    batch: torch.Tensor,
    sigmas: List[float],
    path: Union[str, Path],
    fps: int = 2,
    duration: Optional[float] = None,
    normalize_range: tuple = (0, 1),
    grid_size: Optional[tuple] = None
) -> None:
    """
    Create animation of progressive noising process.
    
    Args:
        batch: Input batch [B, C, H, W]
        sigmas: List of noise levels
        path: Output path (should end with .gif or .mp4)
        fps: Frames per second
        duration: Optional total duration (overrides fps)
        normalize_range: Range for denormalization
        grid_size: Optional (rows, cols) for grid layout
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    device = batch.device
    generator = torch.Generator(device=device).manual_seed(42)
    
    # Determine grid layout
    if grid_size is None:
        batch_size = batch.size(0)
        cols = min(4, int(np.sqrt(batch_size)))
        rows = (batch_size + cols - 1) // cols
        grid_size = (rows, cols)
    
    num_display = min(batch.size(0), grid_size[0] * grid_size[1])
    selected_batch = batch[:num_display]
    
    frames = []
    
    for i, sigma in enumerate(sigmas):
        from .noise import add_gaussian_noise
        
        # Add noise
        if normalize_range == (0, 1):
            clip_range = (0, 1)
        elif normalize_range == (-1, 1):
            clip_range = (-1, 1)
        else:
            clip_range = None
            
        noisy_batch = add_gaussian_noise(
            selected_batch, sigma, clip_range, generator
        )
        
        # Denormalize for visualization
        if normalize_range == (-1, 1):
            noisy_batch = (noisy_batch + 1) / 2
        
        # Create grid for this frame
        grid = vutils.make_grid(
            noisy_batch,
            nrow=grid_size[1],
            normalize=False,
            range=(0, 1),
            pad_value=1.0
        )
        
        # Convert to PIL Image
        frame = tensor_to_pil(grid)
        
        # Add text overlay with sigma value
        frame = _add_text_overlay(frame, f"σ = {sigma:.2f}")
        frames.append(frame)
    
    # Save animation
    if path.suffix.lower() == '.gif':
        if duration is not None:
            duration_per_frame = duration / len(frames) * 1000  # ms
        else:
            duration_per_frame = 1000 / fps
        
        imageio.mimsave(
            path, 
            frames,
            duration=duration_per_frame,
            loop=0
        )
    elif path.suffix.lower() == '.mp4':
        imageio.mimsave(path, frames, fps=fps)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")
    
    print(f"Saved animation to {path}")


def plot_metrics(
    csv_path: Union[str, Path],
    output_path: Union[str, Path],
    figsize: tuple = (12, 8),
    dpi: int = 100
) -> None:
    """
    Plot metrics from CSV log file.
    
    Args:
        csv_path: Path to CSV file with metrics
        output_path: Path for output plot
        figsize: Figure size
        dpi: Plot DPI
    """
    # Read data
    df = pd.read_csv(csv_path)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi)
    fig.suptitle('Noise Impact Analysis', fontsize=16)
    
    # Plot 1: SNR vs Sigma
    axes[0, 0].plot(df['sigma'], df['snr_db'], 'b-o', linewidth=2, markersize=4)
    axes[0, 0].set_xlabel('Noise Level (σ)')
    axes[0, 0].set_ylabel('SNR (dB)')
    axes[0, 0].set_title('Signal-to-Noise Ratio')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: MSE vs Sigma
    axes[0, 1].plot(df['sigma'], df['mse'], 'r-o', linewidth=2, markersize=4)
    axes[0, 1].set_xlabel('Noise Level (σ)')
    axes[0, 1].set_ylabel('Mean Squared Error')
    axes[0, 1].set_title('Reconstruction Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: PSNR vs Sigma
    if 'psnr' in df.columns:
        axes[1, 0].plot(df['sigma'], df['psnr'], 'g-o', linewidth=2, markersize=4)
        axes[1, 0].set_xlabel('Noise Level (σ)')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].set_title('Peak Signal-to-Noise Ratio')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: SSIM vs Sigma
    if 'ssim' in df.columns:
        axes[1, 1].plot(df['sigma'], df['ssim'], 'm-o', linewidth=2, markersize=4)
        axes[1, 1].set_xlabel('Noise Level (σ)')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].set_title('Structural Similarity Index')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metrics plot to {output_path}")


def plot_noise_schedules(
    schedules: dict,
    output_path: Union[str, Path],
    figsize: tuple = (10, 6),
    dpi: int = 100
) -> None:
    """
    Plot different noise schedules for comparison.
    
    Args:
        schedules: Dict of {name: sigma_list}
        output_path: Output path for plot
        figsize: Figure size
        dpi: Plot DPI
    """
    plt.figure(figsize=figsize, dpi=dpi)
    
    for name, sigmas in schedules.items():
        steps = range(len(sigmas))
        plt.plot(steps, sigmas, 'o-', label=name, linewidth=2, markersize=4)
    
    plt.xlabel('Step')
    plt.ylabel('Noise Level (σ)')
    plt.title('Noise Schedule Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"Saved schedule comparison to {output_path}")


def _add_sigma_labels_to_grid(
    image_path: Path,
    sigmas: List[float],
    num_images: int
) -> None:
    """Add sigma value labels to grid image."""
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Try to get a font, fallback to default
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        img_width, img_height = img.size
        row_height = img_height // len(sigmas)
        
        for i, sigma in enumerate(sigmas):
            y_pos = i * row_height + 5
            text = f"σ = {sigma:.2f}"
            draw.text((5, y_pos), text, fill=255, font=font)
        
        img.save(image_path)
    except Exception as e:
        print(f"Warning: Could not add labels to grid: {e}")


def _add_text_overlay(image: Image.Image, text: str) -> Image.Image:
    """Add text overlay to PIL Image."""
    try:
        from PIL import ImageDraw, ImageFont
        
        # Create a copy to avoid modifying original
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # Try to get a font
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # Add text at top-left corner
        draw.text((10, 10), text, fill=255, font=font)
        
        return img
    except Exception as e:
        print(f"Warning: Could not add text overlay: {e}")
        return image