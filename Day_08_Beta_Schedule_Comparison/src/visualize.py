"""
Visualization utilities for schedule plots, trajectory grids, and animations.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torchvision.utils as vutils

from .schedules import get_schedule, plot_all_schedules
from .sampler import DDPMSampler
from .utils import tensor_to_pil, save_gif, load_config


def plot_schedules(
    run_configs: List[Dict[str, Any]], 
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """
    Plot beta, alpha_bar, and SNR overlays for multiple runs.
    
    Args:
        run_configs: List of configuration dictionaries
        save_path: Path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, config in enumerate(run_configs):
        diffusion_config = config['diffusion']
        T = diffusion_config['T']
        schedule_name = diffusion_config['schedule']
        
        # Get schedule kwargs based on schedule type
        schedule_kwargs = {}
        if schedule_name in ['linear', 'quadratic']:
            if 'beta_min' in diffusion_config:
                schedule_kwargs['beta_min'] = diffusion_config['beta_min']
            if 'beta_max' in diffusion_config:
                schedule_kwargs['beta_max'] = diffusion_config['beta_max']
        elif schedule_name == 'cosine':
            if 'cosine_s' in diffusion_config:
                schedule_kwargs['s'] = diffusion_config['cosine_s']
        
        # Create schedule
        schedule = get_schedule(schedule_name, T, **schedule_kwargs)
        timesteps = np.arange(1, T + 1)
        color = colors[i % len(colors)]
        
        # Plot betas
        axes[0].plot(timesteps, schedule['betas'].numpy(),
                    label=schedule_name, color=color, linewidth=2)
        
        # Plot alpha_bars
        axes[1].plot(timesteps, schedule['alpha_bars'].numpy(),
                    label=schedule_name, color=color, linewidth=2)
        
        # Plot SNR
        axes[2].plot(timesteps, schedule['snr'].numpy(),
                    label=schedule_name, color=color, linewidth=2)
    
    # Format plots
    axes[0].set_title('Beta Schedule: β_t')
    axes[0].set_xlabel('Timestep t')
    axes[0].set_ylabel('β_t')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Cumulative Alpha: ᾱ_t')
    axes[1].set_xlabel('Timestep t')
    axes[1].set_ylabel('ᾱ_t')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Signal-to-Noise Ratio')
    axes[2].set_xlabel('Timestep t')
    axes[2].set_ylabel('SNR (log scale)')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Schedule comparison plot saved to {save_path}")
    
    plt.show()


def trajectory_grid(
    model: nn.Module,
    schedule: Dict[str, torch.Tensor],
    device: torch.device,
    num_samples: int = 8,
    num_timesteps: int = 10,
    save_path: Optional[str] = None
) -> torch.Tensor:
    """
    Create trajectory grid showing reverse diffusion process T→0.
    
    Args:
        model: Trained diffusion model
        schedule: Diffusion schedule
        device: Device
        num_samples: Number of samples to show
        num_timesteps: Number of timesteps to visualize
        save_path: Path to save grid
        
    Returns:
        Grid tensor [C, H_grid, W_grid]
    """
    model.eval()
    
    # Create sampler
    sampler = DDPMSampler(
        model,
        schedule['betas'],
        schedule['alphas'],
        schedule['alpha_bars']
    )
    
    # Determine image shape
    if hasattr(model, 'in_channels'):
        channels = model.in_channels
    else:
        channels = 1  # Default for MNIST
    
    if channels == 1:
        height, width = 28, 28  # MNIST
    else:
        height, width = 32, 32  # CIFAR
    
    shape = (num_samples, channels, height, width)
    
    # Sample trajectory
    with torch.no_grad():
        trajectory = sampler.sample(shape, device, return_trajectory=True)
    
    # Select timesteps to visualize
    T = trajectory.shape[1] - 1  # Remove initial noise
    if num_timesteps >= T:
        selected_timesteps = list(range(0, T + 1))
    else:
        selected_timesteps = np.linspace(0, T, num_timesteps, dtype=int)
    
    # Create grid
    grid_images = []
    for t_idx in selected_timesteps:
        grid_images.append(trajectory[:, t_idx])
    
    # Concatenate all timesteps
    all_images = torch.cat(grid_images, dim=0)  # [num_samples * num_timesteps, C, H, W]
    
    # Create grid
    grid = vutils.make_grid(
        all_images,
        nrow=num_samples,  # Each row is one timestep
        normalize=True,
        value_range=(-1, 1),
        pad_value=1.0
    )
    
    if save_path:
        vutils.save_image(grid, save_path)
        print(f"Trajectory grid saved to {save_path}")
    
    return grid


def multi_run_sample_panel(
    model_paths: List[str],
    config_paths: List[str],
    device: torch.device,
    num_samples: int = 16,
    save_path: Optional[str] = None
) -> torch.Tensor:
    """
    Create sample panel with rows=runs (linear/cosine/quadratic), cols=samples.
    
    Args:
        model_paths: Paths to trained model checkpoints
        config_paths: Paths to configuration files
        device: Device
        num_samples: Number of samples per run
        save_path: Path to save panel
        
    Returns:
        Panel grid tensor
    """
    from .models.unet_small import UNetSmall
    from .utils import load_checkpoint
    
    all_samples = []
    
    for model_path, config_path in zip(model_paths, config_paths):
        # Load config
        config = load_config(config_path)
        
        # Create model
        model_config = config['model']
        model = UNetSmall(
            in_channels=model_config['in_ch'],
            out_channels=model_config['in_ch'],
            base_channels=model_config['base_ch'],
            channel_multipliers=model_config['ch_mult'],
            time_embed_dim=model_config['time_embed_dim']
        ).to(device)
        
        # Load checkpoint
        load_checkpoint(model_path, model, device=device)
        
        # Get schedule
        diffusion_config = config['diffusion']
        schedule_kwargs = {}
        if 'beta_min' in diffusion_config:
            schedule_kwargs['beta_min'] = diffusion_config['beta_min']
        if 'beta_max' in diffusion_config:
            schedule_kwargs['beta_max'] = diffusion_config['beta_max']
        if 'cosine_s' in diffusion_config:
            schedule_kwargs['s'] = diffusion_config['cosine_s']
        
        schedule = get_schedule(
            diffusion_config['schedule'], 
            diffusion_config['T'], 
            **schedule_kwargs
        )
        
        # Move schedule to device
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        # Sample
        sampler = DDPMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars']
        )
        
        # Determine shape
        data_config = config['data']
        if data_config['dataset'].lower() == 'mnist':
            shape = (num_samples, 1, 28, 28)
        else:
            shape = (num_samples, 3, 32, 32)
        
        with torch.no_grad():
            samples = sampler.sample(shape, device)
        
        all_samples.append(samples)
    
    # Stack all samples
    all_samples = torch.cat(all_samples, dim=0)  # [num_runs * num_samples, C, H, W]
    
    # Create grid
    grid = vutils.make_grid(
        all_samples,
        nrow=num_samples,  # Each row is one run
        normalize=True,
        value_range=(-1, 1),
        pad_value=1.0
    )
    
    if save_path:
        vutils.save_image(grid, save_path)
        print(f"Multi-run sample panel saved to {save_path}")
    
    return grid


def create_reverse_animation(
    model: nn.Module,
    schedule: Dict[str, torch.Tensor],
    device: torch.device,
    save_path: str,
    num_frames: int = 50,
    duration: int = 100
) -> None:
    """
    Create animated GIF showing reverse diffusion process for a single sample.
    
    Args:
        model: Trained model
        schedule: Diffusion schedule
        device: Device
        save_path: Path to save GIF
        num_frames: Number of frames in animation
        duration: Duration per frame in ms
    """
    model.eval()
    
    # Create sampler
    sampler = DDPMSampler(
        model,
        schedule['betas'],
        schedule['alphas'],
        schedule['alpha_bars']
    )
    
    # Sample single trajectory
    if hasattr(model, 'in_channels'):
        channels = model.in_channels
    else:
        channels = 1
    
    if channels == 1:
        shape = (1, 1, 28, 28)
    else:
        shape = (1, 3, 32, 32)
    
    with torch.no_grad():
        trajectory = sampler.sample(shape, device, return_trajectory=True)
    
    # Remove batch dimension
    trajectory = trajectory.squeeze(0)  # [T+1, C, H, W]
    
    # Select frames
    T = trajectory.shape[0] - 1
    if num_frames >= T:
        frame_indices = list(range(T + 1))
    else:
        frame_indices = np.linspace(0, T, num_frames, dtype=int)
    
    # Convert to PIL frames
    frames = []
    for i in frame_indices:
        frame_tensor = trajectory[i]
        pil_frame = tensor_to_pil(frame_tensor, normalize=True)
        
        # Resize for better visibility
        pil_frame = pil_frame.resize((128, 128), Image.NEAREST)
        frames.append(pil_frame)
    
    # Save as GIF
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0
    )
    
    print(f"Reverse animation saved to {save_path}")


def plot_training_curves(
    run_dirs: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 4)
) -> None:
    """
    Plot training curves for multiple runs.
    
    Args:
        run_dirs: List of run directories containing metrics.csv
        save_path: Path to save plot
        figsize: Figure size
    """
    import pandas as pd
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, run_dir in enumerate(run_dirs):
        run_path = Path(run_dir)
        metrics_path = run_path / 'logs' / 'metrics.csv'
        
        if not metrics_path.exists():
            print(f"Warning: {metrics_path} not found")
            continue
        
        df = pd.read_csv(metrics_path)
        run_name = run_path.name
        color = colors[i % len(colors)]
        
        # Plot training loss
        axes[0].plot(df['epoch'], df['train_loss'], 
                    label=run_name, color=color, linewidth=2)
        
        # Plot learning rate
        axes[1].plot(df['epoch'], df['lr'], 
                    label=run_name, color=color, linewidth=2)
        
        # Plot epoch time
        axes[2].plot(df['epoch'], df['epoch_time'], 
                    label=run_name, color=color, linewidth=2)
    
    # Format plots
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Learning Rate')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('LR')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    axes[2].set_title('Epoch Time')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Time (s)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def test_visualization():
    """Test visualization functions."""
    # Test schedule plotting
    configs = [
        {'diffusion': {'T': 1000, 'schedule': 'linear'}},
        {'diffusion': {'T': 1000, 'schedule': 'cosine'}},
        {'diffusion': {'T': 1000, 'schedule': 'quadratic'}}
    ]
    
    plot_schedules(configs, save_path='test_schedules.png')
    print("Visualization test completed!")


if __name__ == "__main__":
    test_visualization()
