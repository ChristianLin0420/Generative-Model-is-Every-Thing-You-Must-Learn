"""DDPM noise schedules and related computations."""

import torch
import numpy as np
from typing import Tuple, Union
import math


def make_beta_schedule(
    T: int, 
    schedule_type: str = "linear", 
    beta_start: float = 1e-4, 
    beta_end: float = 0.02,
    cosine_s: float = 0.008
) -> torch.Tensor:
    """Create beta schedule for diffusion process.
    
    Args:
        T: Number of diffusion timesteps
        schedule_type: Type of schedule ("linear", "cosine", "sigmoid")
        beta_start: Starting beta value for linear schedule
        beta_end: Ending beta value for linear schedule
        cosine_s: Small constant for cosine schedule to prevent beta from being too small
    
    Returns:
        Beta values of shape (T,)
    """
    if schedule_type == "linear":
        betas = torch.linspace(beta_start, beta_end, T)
    
    elif schedule_type == "cosine":
        # Cosine schedule as proposed in "Improved Denoising Diffusion Probabilistic Models"
        def alpha_bar_cosine(t):
            return math.cos((t + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
        
        timesteps = torch.linspace(0, T, T + 1) / T  # 0 to 1
        alpha_bars = torch.tensor([alpha_bar_cosine(t) for t in timesteps])
        
        # beta_t = 1 - alpha_t = 1 - alpha_bar_t / alpha_bar_{t-1}
        betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
        
    elif schedule_type == "sigmoid":
        # Sigmoid schedule - creates a smooth S-curve
        timesteps = torch.linspace(-6, 6, T)  # Range that gives good sigmoid shape
        sigmoid_vals = torch.sigmoid(timesteps)
        # Normalize to [beta_start, beta_end]
        betas = beta_start + (beta_end - beta_start) * sigmoid_vals
        
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    # Clamp to reasonable range to avoid numerical issues
    betas = torch.clamp(betas, min=1e-6, max=0.999)
    
    return betas


def compute_alpha_schedule(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute alpha and alpha_bar from beta schedule.
    
    Args:
        betas: Beta values of shape (T,)
    
    Returns:
        Tuple of (alphas, alpha_bars) both of shape (T,)
    """
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    return alphas, alpha_bars


def get_ddpm_schedule(
    T: int,
    schedule_type: str = "linear",
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get complete DDPM schedule.
    
    Args:
        T: Number of timesteps
        schedule_type: Type of schedule
        **kwargs: Additional arguments for make_beta_schedule
    
    Returns:
        Tuple of (betas, alphas, alpha_bars)
    """
    betas = make_beta_schedule(T, schedule_type, **kwargs)
    alphas, alpha_bars = compute_alpha_schedule(betas)
    
    return betas, alphas, alpha_bars


def get_schedule_stats(
    betas: torch.Tensor,
    alphas: torch.Tensor, 
    alpha_bars: torch.Tensor
) -> dict:
    """Get statistics about the schedule.
    
    Args:
        betas: Beta values
        alphas: Alpha values  
        alpha_bars: Alpha bar values
    
    Returns:
        Dictionary with schedule statistics
    """
    T = len(betas)
    
    stats = {
        "T": T,
        "beta_min": float(betas.min()),
        "beta_max": float(betas.max()),
        "alpha_bar_min": float(alpha_bars.min()),
        "alpha_bar_max": float(alpha_bars.max()),
        "alpha_bar_final": float(alpha_bars[-1]),
    }
    
    return stats