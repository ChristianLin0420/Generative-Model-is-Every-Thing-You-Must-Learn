"""Forward diffusion process implementation."""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Union, List
import math


def extract(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: torch.Size) -> torch.Tensor:
    """Extract values from array at given timesteps and reshape for broadcasting.
    
    Args:
        arr: Array to extract from, shape (T,)
        timesteps: Timesteps to extract, shape (batch_size,)
        broadcast_shape: Shape to broadcast to, typically (batch_size, 1, 1, 1)
    
    Returns:
        Extracted values reshaped for broadcasting
    """
    res = arr.gather(-1, timesteps)
    # Reshape for broadcasting: (batch_size,) -> (batch_size, 1, 1, 1)
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res


def q_xt_given_xtm1(
    x_tm1: torch.Tensor,
    t: torch.Tensor,
    betas: torch.Tensor,
    noise: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample from q(x_t | x_{t-1}).
    
    Args:
        x_tm1: x at timestep t-1, shape (batch_size, channels, height, width)
        t: Current timestep, shape (batch_size,)
        betas: Beta schedule, shape (T,)
        noise: Optional noise tensor, if None will be sampled
    
    Returns:
        Tuple of (x_t, noise_used)
    """
    batch_size = x_tm1.shape[0]
    device = x_tm1.device
    
    if noise is None:
        noise = torch.randn_like(x_tm1)
    
    # Extract beta_t for current timesteps
    beta_t = extract(betas, t, x_tm1.shape)
    alpha_t = 1.0 - beta_t
    
    # Sample: x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
    x_t = torch.sqrt(alpha_t) * x_tm1 + torch.sqrt(1.0 - alpha_t) * noise
    
    return x_t, noise


def q_xt_given_x0(
    x0: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor,
    noise: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample from q(x_t | x_0) - closed form.
    
    Args:
        x0: Original image, shape (batch_size, channels, height, width)
        t: Timestep, shape (batch_size,)
        alpha_bars: Alpha bar schedule, shape (T,)
        noise: Optional noise tensor, if None will be sampled
    
    Returns:
        Tuple of (x_t, noise_used)
    """
    if noise is None:
        noise = torch.randn_like(x0)
    
    # Extract alpha_bar_t for current timesteps
    alpha_bar_t = extract(alpha_bars, t, x0.shape)
    
    # Sample: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    
    return x_t, noise


def sample_trajectory(
    x0: torch.Tensor,
    T: int,
    betas: torch.Tensor,
    alpha_bars: torch.Tensor,
    use_closed_form: bool = False,
    fixed_noise: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Sample full trajectory x_0:T.
    
    Args:
        x0: Initial image, shape (batch_size, channels, height, width)
        T: Number of diffusion steps
        betas: Beta schedule
        alpha_bars: Alpha bar schedule  
        use_closed_form: If True, use q(x_t|x_0) for each step (independent noise)
                        If False, use iterative q(x_t|x_{t-1}) (sequential)
        fixed_noise: Optional fixed noise for reproducibility, shape (T, *x0.shape)
    
    Returns:
        Trajectory tensor of shape (batch_size, T+1, channels, height, width)
        Index 0 contains x_0, index t contains x_t
    """
    batch_size = x0.shape[0]
    device = x0.device
    trajectory = torch.zeros((batch_size, T + 1) + x0.shape[1:], device=device)
    trajectory[:, 0] = x0
    
    if use_closed_form:
        # Sample each x_t independently from x_0
        for t_val in range(1, T + 1):
            t_tensor = torch.full((batch_size,), t_val - 1, device=device)  # 0-indexed
            
            if fixed_noise is not None:
                noise = fixed_noise[t_val - 1]  # Use pre-generated noise
            else:
                noise = None
            
            x_t, _ = q_xt_given_x0(x0, t_tensor, alpha_bars, noise=noise)
            trajectory[:, t_val] = x_t
    else:
        # Sequential sampling: x_t from x_{t-1}
        x_current = x0
        for t_val in range(1, T + 1):
            t_tensor = torch.full((batch_size,), t_val - 1, device=device)  # 0-indexed
            
            if fixed_noise is not None:
                noise = fixed_noise[t_val - 1]
            else:
                noise = None
            
            x_current, _ = q_xt_given_xtm1(x_current, t_tensor, betas, noise=noise)
            trajectory[:, t_val] = x_current
    
    return trajectory


def snr(alpha_bars: torch.Tensor) -> torch.Tensor:
    """Compute Signal-to-Noise Ratio.
    
    Args:
        alpha_bars: Alpha bar values, shape (T,)
    
    Returns:
        SNR values, shape (T,)
    """
    return alpha_bars / (1.0 - alpha_bars)


def snr_db(alpha_bars: torch.Tensor) -> torch.Tensor:
    """Compute SNR in decibels.
    
    Args:
        alpha_bars: Alpha bar values, shape (T,)
    
    Returns:
        SNR in dB, shape (T,)
    """
    snr_linear = snr(alpha_bars)
    return 10.0 * torch.log10(snr_linear + 1e-8)  # Add small epsilon for numerical stability


def kl_divergence_to_unit_gaussian(
    x0: torch.Tensor,
    t: torch.Tensor,
    alpha_bars: torch.Tensor
) -> torch.Tensor:
    """Compute KL divergence between q(x_t|x_0) and N(0,I).
    
    For q(x_t|x_0) ~ N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    KL(q(x_t|x_0) || N(0,I)) = 0.5 * (alpha_bar_t * ||x_0||^2 + (1-alpha_bar_t) - 1 - log(1-alpha_bar_t))
    
    Args:
        x0: Original images, shape (batch_size, channels, height, width)
        t: Timesteps, shape (batch_size,)
        alpha_bars: Alpha bar schedule
    
    Returns:
        KL divergence per image, shape (batch_size,)
    """
    # Extract alpha_bar_t
    alpha_bar_t = extract(alpha_bars, t, x0.shape)
    
    # Compute ||x_0||^2 per image
    x0_norm_sq = torch.sum(x0.pow(2), dim=[1, 2, 3])  # (batch_size,)
    
    # KL formula
    variance_term = 1.0 - alpha_bar_t.squeeze()  # Remove extra dimensions
    
    kl = 0.5 * (
        alpha_bar_t.squeeze() * x0_norm_sq +
        variance_term * x0[0].numel() -  # Multiply by number of pixels
        x0[0].numel() -  # -D term
        x0[0].numel() * torch.log(variance_term + 1e-8)  # -log(variance) term
    )
    
    return kl


def compute_mse_to_x0(
    x_t: torch.Tensor,
    x0: torch.Tensor
) -> torch.Tensor:
    """Compute MSE between x_t and x_0.
    
    Args:
        x_t: Noisy images at timestep t, shape (batch_size, channels, height, width)
        x0: Original images, shape (batch_size, channels, height, width)
    
    Returns:
        MSE per image, shape (batch_size,)
    """
    mse = torch.sum((x_t - x0).pow(2), dim=[1, 2, 3])
    return mse / x0[0].numel()  # Normalize by number of pixels


def get_timesteps_for_snr_threshold(
    alpha_bars: torch.Tensor,
    snr_threshold_db: float
) -> torch.Tensor:
    """Find timesteps where SNR drops below threshold.
    
    Args:
        alpha_bars: Alpha bar schedule
        snr_threshold_db: SNR threshold in dB
    
    Returns:
        Timesteps where SNR < threshold
    """
    snr_values_db = snr_db(alpha_bars)
    return torch.where(snr_values_db < snr_threshold_db)[0]