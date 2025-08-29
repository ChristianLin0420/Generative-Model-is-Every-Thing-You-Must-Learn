"""
Beta schedules for DDPM: linear, cosine, and quadratic.
Each function returns a dictionary with betas, alphas, alpha_bars, and snr.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


def beta_linear(T: int, beta_min: float = 1e-4, beta_max: float = 0.02) -> Dict[str, torch.Tensor]:
    """
    Linear beta schedule: β_t grows linearly from beta_min to beta_max.
    
    Args:
        T: Number of diffusion timesteps
        beta_min: Minimum beta value at t=1
        beta_max: Maximum beta value at t=T
        
    Returns:
        Dictionary with 'betas', 'alphas', 'alpha_bars', 'snr'
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    
    betas = torch.linspace(beta_min, beta_max, T, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    snr = alpha_bars / (1.0 - alpha_bars)
    
    return {
        'betas': betas,
        'alphas': alphas, 
        'alpha_bars': alpha_bars,
        'snr': snr
    }


def beta_cosine(T: int, s: float = 0.008) -> Dict[str, torch.Tensor]:
    """
    Cosine beta schedule from "Improved Denoising Diffusion Probabilistic Models".
    Constructs via alpha_bar cosine schedule, then recovers beta.
    
    Args:
        T: Number of diffusion timesteps
        s: Small offset to prevent beta from being too small near t=0
        
    Returns:
        Dictionary with 'betas', 'alphas', 'alpha_bars', 'snr'
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    
    def alpha_bar_fn(t):
        return np.cos((t / T + s) / (1 + s) * np.pi / 2) ** 2
    
    alpha_bars = torch.zeros(T, dtype=torch.float32)
    for t in range(T):
        alpha_bars[t] = alpha_bar_fn(t + 1) / alpha_bar_fn(0)
    
    # Recover betas from alpha_bars
    betas = torch.zeros(T, dtype=torch.float32)
    betas[0] = 1 - alpha_bars[0]
    for t in range(1, T):
        betas[t] = 1 - alpha_bars[t] / alpha_bars[t-1]
    
    # Clamp betas to reasonable range
    betas = torch.clamp(betas, 0.0, 0.999)
    
    alphas = 1.0 - betas
    snr = alpha_bars / (1.0 - alpha_bars)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars,
        'snr': snr
    }


def beta_quadratic(T: int, beta_min: float = 1e-4, beta_max: float = 0.02) -> Dict[str, torch.Tensor]:
    """
    Quadratic beta schedule: β_t grows quadratically from beta_min to beta_max.
    
    Args:
        T: Number of diffusion timesteps
        beta_min: Minimum beta value at t=1
        beta_max: Maximum beta value at t=T
        
    Returns:
        Dictionary with 'betas', 'alphas', 'alpha_bars', 'snr'
    """
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    
    # Quadratic progression from 0 to 1
    t_norm = torch.linspace(0, 1, T, dtype=torch.float32)
    betas = beta_min + (beta_max - beta_min) * (t_norm ** 2)
    
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    snr = alpha_bars / (1.0 - alpha_bars)
    
    return {
        'betas': betas,
        'alphas': alphas,
        'alpha_bars': alpha_bars,
        'snr': snr
    }


def get_schedule(schedule_name: str, T: int, **kwargs) -> Dict[str, torch.Tensor]:
    """
    Get a beta schedule by name.
    
    Args:
        schedule_name: One of 'linear', 'cosine', 'quadratic'
        T: Number of timesteps
        **kwargs: Additional arguments for the schedule function
        
    Returns:
        Schedule dictionary
    """
    if schedule_name == 'linear':
        return beta_linear(T, **kwargs)
    elif schedule_name == 'cosine':
        return beta_cosine(T, **kwargs)
    elif schedule_name == 'quadratic':
        return beta_quadratic(T, **kwargs)
    else:
        raise ValueError(f"Unknown schedule: {schedule_name}")


def plot_all_schedules(
    T: int = 1000, 
    schedules: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
) -> None:
    """
    Plot beta, alpha_bar, and SNR for all schedules side by side.
    
    Args:
        T: Number of timesteps
        schedules: List of schedule names to plot
        save_path: Path to save the plot
        figsize: Figure size
    """
    if schedules is None:
        schedules = ['linear', 'cosine', 'quadratic']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    timesteps = np.arange(1, T + 1)
    
    colors = {'linear': 'blue', 'cosine': 'red', 'quadratic': 'green'}
    
    for schedule_name in schedules:
        schedule = get_schedule(schedule_name, T)
        color = colors.get(schedule_name, 'black')
        
        # Plot betas
        axes[0].plot(timesteps, schedule['betas'].numpy(), 
                    label=schedule_name, color=color, linewidth=2)
        
        # Plot alpha_bars
        axes[1].plot(timesteps, schedule['alpha_bars'].numpy(), 
                    label=schedule_name, color=color, linewidth=2)
        
        # Plot SNR (log scale)
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
        print(f"Schedule plots saved to {save_path}")
    
    plt.show()


def validate_schedule(schedule: Dict[str, torch.Tensor]) -> bool:
    """
    Validate that a schedule dictionary has proper properties:
    - betas ∈ (0, 1)
    - alpha_bars monotonically decreasing
    - All tensors have same length
    
    Args:
        schedule: Schedule dictionary
        
    Returns:
        True if valid, False otherwise
    """
    betas = schedule['betas']
    alpha_bars = schedule['alpha_bars']
    
    # Check beta range
    if not torch.all((betas > 0) & (betas < 1)):
        print("ERROR: betas not in (0, 1)")
        return False
    
    # Check alpha_bars monotonicity
    if not torch.all(alpha_bars[1:] <= alpha_bars[:-1]):
        print("ERROR: alpha_bars not monotonically decreasing")
        return False
    
    # Check tensor lengths
    T = len(betas)
    if not all(len(tensor) == T for tensor in schedule.values()):
        print("ERROR: schedule tensors have different lengths")
        return False
    
    return True
