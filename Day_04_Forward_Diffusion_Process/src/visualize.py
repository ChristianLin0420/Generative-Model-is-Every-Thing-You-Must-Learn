"""Visualization functions for forward diffusion process."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Union
from PIL import Image

from .utils import unnormalize_to_zero_to_one, save_trajectory_grid, save_animation
from .forward import sample_trajectory, q_xt_given_x0


def plot_schedules(
    schedules_data: Dict[str, Dict[str, torch.Tensor]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> None:
    """Plot beta, alpha_bar, and SNR curves for different schedules.
    
    Args:
        schedules_data: Dictionary mapping schedule names to their statistics
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for idx, (schedule_name, stats) in enumerate(schedules_data.items()):
        color = colors[idx % len(colors)]
        timesteps = stats['timesteps'].cpu().numpy()
        
        # Plot beta
        axes[0].plot(
            timesteps, 
            stats['betas'].cpu().numpy(), 
            label=schedule_name,
            color=color,
            linewidth=2
        )
        
        # Plot alpha_bar
        axes[1].plot(
            timesteps, 
            stats['alpha_bars'].cpu().numpy(), 
            label=schedule_name,
            color=color,
            linewidth=2
        )
        
        # Plot SNR in dB
        axes[2].plot(
            timesteps, 
            stats['snr_db'].cpu().numpy(), 
            label=schedule_name,
            color=color,
            linewidth=2
        )
    
    # Formatting
    axes[0].set_title('Beta Schedule')
    axes[0].set_xlabel('Timestep t')
    axes[0].set_ylabel(r'$\beta_t$')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Cumulative Alpha')
    axes[1].set_xlabel('Timestep t')
    axes[1].set_ylabel(r'$\bar{\alpha}_t$')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].set_title('Signal-to-Noise Ratio')
    axes[2].set_xlabel('Timestep t')
    axes[2].set_ylabel('SNR (dB)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2].axhline(y=-5, color='red', linestyle='--', alpha=0.5, label='-5dB threshold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved schedule plots to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_snr_analysis(
    schedules_data: Dict[str, Dict[str, torch.Tensor]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6)
) -> None:
    """Plot detailed SNR analysis.
    
    Args:
        schedules_data: Dictionary mapping schedule names to their statistics
        save_path: Optional path to save the plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    thresholds = [-5, -10, -15, -20]
    
    for idx, (schedule_name, stats) in enumerate(schedules_data.items()):
        color = colors[idx % len(colors)]
        timesteps = stats['timesteps'].cpu().numpy()
        snr_values = stats['snr_db'].cpu().numpy()
        
        # Plot SNR curves
        axes[0].plot(timesteps, snr_values, label=schedule_name, color=color, linewidth=2)
        
        # Mark threshold crossings
        if 'snr_thresholds' in stats:
            for threshold in thresholds:
                if threshold in stats['snr_thresholds']:
                    crossing_t = stats['snr_thresholds'][threshold]
                    if crossing_t < len(timesteps):
                        axes[0].scatter(
                            crossing_t, threshold, 
                            color=color, s=50, zorder=5
                        )
    
    # Add threshold lines
    for threshold in thresholds:
        axes[0].axhline(y=threshold, color='gray', linestyle='--', alpha=0.5)
        axes[0].text(len(timesteps) * 0.02, threshold + 1, f'{threshold}dB', 
                    color='gray', fontsize=10)
    
    axes[0].set_title('SNR with Threshold Crossings')
    axes[0].set_xlabel('Timestep t')
    axes[0].set_ylabel('SNR (dB)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Bar plot of threshold crossing times
    schedule_names = list(schedules_data.keys())
    threshold_times = {threshold: [] for threshold in thresholds}
    
    for schedule_name in schedule_names:
        stats = schedules_data[schedule_name]
        if 'snr_thresholds' in stats:
            for threshold in thresholds:
                time = stats['snr_thresholds'].get(threshold, len(timesteps))
                threshold_times[threshold].append(time)
        else:
            for threshold in thresholds:
                threshold_times[threshold].append(len(timesteps))
    
    x = np.arange(len(schedule_names))
    width = 0.15
    
    for i, threshold in enumerate(thresholds):
        offset = (i - len(thresholds)/2) * width
        axes[1].bar(
            x + offset, threshold_times[threshold], width,
            label=f'{threshold}dB', alpha=0.7
        )
    
    axes[1].set_title('Time to SNR Thresholds')
    axes[1].set_xlabel('Schedule')
    axes[1].set_ylabel('Timestep')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(schedule_names)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved SNR analysis to {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_trajectory_grid(
    x0_batch: torch.Tensor,
    timesteps_to_show: List[int],
    alpha_bars: torch.Tensor,
    save_path: Optional[Path] = None,
    normalize: bool = True,
    title: Optional[str] = None
) -> torch.Tensor:
    """Create grid showing image degradation across timesteps.
    
    Args:
        x0_batch: Original images, shape (batch_size, channels, height, width)
        timesteps_to_show: List of timesteps to visualize
        alpha_bars: Alpha bar schedule
        save_path: Optional path to save the grid
        normalize: Whether input is in [-1,1] range
        title: Optional title for the plot
    
    Returns:
        Trajectory tensor for the selected timesteps
    """
    batch_size = min(16, x0_batch.shape[0])  # Limit to 16 images for visualization
    x0_subset = x0_batch[:batch_size]
    device = x0_subset.device
    
    # Create trajectory for selected timesteps
    trajectories = torch.zeros(
        (batch_size, len(timesteps_to_show)) + x0_subset.shape[1:],
        device=device
    )
    
    for t_idx, t in enumerate(timesteps_to_show):
        if t == 0:
            trajectories[:, t_idx] = x0_subset
        else:
            t_tensor = torch.full((batch_size,), t - 1, device=device)  # 0-indexed
            x_t, _ = q_xt_given_x0(x0_subset, t_tensor, alpha_bars)
            trajectories[:, t_idx] = x_t
    
    if save_path:
        save_trajectory_grid(
            trajectories, 
            timesteps_to_show, 
            save_path,
            normalize=normalize,
            title=title
        )
    
    return trajectories


def create_trajectory_animation(
    x0_image: torch.Tensor,
    T: int,
    alpha_bars: torch.Tensor,
    save_path: Optional[Path] = None,
    normalize: bool = True,
    duration: float = 0.1
) -> torch.Tensor:
    """Create animation of single image diffusion trajectory.
    
    Args:
        x0_image: Single original image, shape (1, channels, height, width)
        T: Total number of timesteps
        alpha_bars: Alpha bar schedule
        save_path: Optional path to save animation
        normalize: Whether input is in [-1,1] range
        duration: Duration between frames
    
    Returns:
        Full trajectory tensor
    """
    # Sample full trajectory using closed-form sampling
    trajectory = sample_trajectory(
        x0_image, T, None, alpha_bars,
        use_closed_form=True,
        fixed_noise=None  # Random noise for each step
    )
    
    # Remove batch dimension for animation: (1, T+1, C, H, W) -> (T+1, C, H, W)
    trajectory_single = trajectory.squeeze(0)
    
    if save_path:
        save_animation(
            trajectory_single,
            save_path,
            duration=duration,
            normalize=normalize
        )
    
    return trajectory


def plot_pixel_histograms(
    x0_batch: torch.Tensor,
    timesteps_to_show: List[int],
    alpha_bars: torch.Tensor,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """Plot histograms of pixel values at different timesteps.
    
    Args:
        x0_batch: Original images
        timesteps_to_show: Timesteps to show histograms for
        alpha_bars: Alpha bar schedule
        save_path: Optional path to save plot
        figsize: Figure size
    """
    device = x0_batch.device
    n_timesteps = len(timesteps_to_show)
    
    # Create subplots
    cols = min(4, n_timesteps)
    rows = (n_timesteps + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    for t_idx, t in enumerate(timesteps_to_show):
        if t == 0:
            x_t = x0_batch
        else:
            batch_size = x0_batch.shape[0]
            t_tensor = torch.full((batch_size,), t - 1, device=device)
            x_t, _ = q_xt_given_x0(x0_batch, t_tensor, alpha_bars)
        
        # Flatten all pixel values
        pixel_values = x_t.cpu().flatten().numpy()
        
        # Plot histogram
        axes[t_idx].hist(pixel_values, bins=50, alpha=0.7, density=True, color='skyblue')
        axes[t_idx].set_title(f't = {t}')
        axes[t_idx].set_xlabel('Pixel Value')
        axes[t_idx].set_ylabel('Density')
        axes[t_idx].grid(True, alpha=0.3)
        
        # Add standard Gaussian reference for later timesteps
        if t > 100:
            x_range = np.linspace(pixel_values.min(), pixel_values.max(), 100)
            gaussian = (1/np.sqrt(2*np.pi)) * np.exp(-0.5 * x_range**2)
            axes[t_idx].plot(x_range, gaussian, 'r--', alpha=0.8, label='N(0,1)')
            axes[t_idx].legend()
        
        # Add statistics
        mean_val = float(pixel_values.mean())
        std_val = float(pixel_values.std())
        axes[t_idx].text(
            0.05, 0.95, 
            f'μ={mean_val:.3f}\nσ={std_val:.3f}',
            transform=axes[t_idx].transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
    
    # Hide unused subplots
    for i in range(n_timesteps, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Pixel Value Distributions Across Timesteps', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved histograms to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_mse_and_kl_curves(
    stats: Dict[str, torch.Tensor],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> None:
    """Plot MSE and KL divergence curves.
    
    Args:
        stats: Statistics dictionary with MSE and KL data
        save_path: Optional path to save plot
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    timesteps = stats['timesteps'].cpu().numpy()
    
    # Plot MSE
    axes[0].plot(timesteps, stats['mse_to_x0'].cpu().numpy(), 'b-', linewidth=2)
    axes[0].set_title('MSE between x_t and x_0')
    axes[0].set_xlabel('Timestep t')
    axes[0].set_ylabel('MSE')
    axes[0].grid(True, alpha=0.3)
    
    # Add theoretical line: MSE should approach 1 - alpha_bar_t for unit variance data
    theoretical_mse = 1.0 - stats['alpha_bars'].cpu().numpy()
    axes[0].plot(timesteps, theoretical_mse, 'r--', alpha=0.7, label='1 - α̅_t')
    axes[0].legend()
    
    # Plot KL divergence
    axes[1].plot(timesteps, stats['kl_to_unit'].cpu().numpy(), 'g-', linewidth=2)
    axes[1].set_title('KL(q(x_t|x_0) || N(0,I))')
    axes[1].set_xlabel('Timestep t')
    axes[1].set_ylabel('KL Divergence')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved MSE/KL curves to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_class_conditional_mse(
    class_stats: Dict[int, Dict[str, torch.Tensor]],
    class_names: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """Plot per-class MSE degradation curves.
    
    Args:
        class_stats: Per-class statistics
        class_names: Optional class names for legend
        save_path: Optional path to save plot
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_stats)))
    
    for class_id, stats in class_stats.items():
        timesteps = torch.arange(len(stats['mse_to_x0']))
        label = class_names[class_id] if class_names else f'Class {class_id}'
        
        ax.plot(
            timesteps.numpy(),
            stats['mse_to_x0'].cpu().numpy(),
            color=colors[class_id],
            linewidth=2,
            label=label
        )
    
    ax.set_title('Per-Class MSE Degradation')
    ax.set_xlabel('Timestep t')
    ax.set_ylabel('MSE to x_0')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved class-conditional MSE to {save_path}")
    else:
        plt.show()
    
    plt.close()