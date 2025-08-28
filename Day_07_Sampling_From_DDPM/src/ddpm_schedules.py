"""
DDPM noise schedules and coefficient computation.
Matches Day 6 implementation exactly to avoid drift.

Key components:
- β_t, α_t, ᾱ_t schedule computation
- extract() helper for per-batch gathering
- Support for linear and cosine schedules
"""

import torch
import torch.nn.functional as F
import math
from typing import Union, Optional


def linear_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Linear schedule for β_t from Ho et al. (2020).
    
    Args:
        timesteps: Number of diffusion timesteps T
        beta_start: Starting β value
        beta_end: Ending β value
    
    Returns:
        β schedule of shape (T,)
    """
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule for β_t from Nichol & Dhariwal (2021).
    More stable than linear schedule.
    
    Args:
        timesteps: Number of diffusion timesteps T
        s: Small offset to prevent β_t from being too small at t=0
    
    Returns:
        β schedule of shape (T,)
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def quadratic_beta_schedule(timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """
    Quadratic schedule for β_t.
    
    Args:
        timesteps: Number of diffusion timesteps T
        beta_start: Starting β value  
        beta_end: Ending β value
    
    Returns:
        β schedule of shape (T,)
    """
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2


class DDPMSchedules:
    """
    DDPM noise schedule and coefficient computation.
    Precomputes all necessary coefficients for efficient sampling.
    """
    
    def __init__(self, timesteps: int = 1000, schedule: str = "cosine", 
                 beta_start: float = 0.0001, beta_end: float = 0.02):
        """
        Initialize DDPM schedules.
        
        Args:
            timesteps: Number of diffusion timesteps T
            schedule: Schedule type ("linear", "cosine", "quadratic")
            beta_start: Starting β value (for linear/quadratic)
            beta_end: Ending β value (for linear/quadratic)
        """
        self.timesteps = timesteps
        self.schedule = schedule
        
        # Compute β_t schedule
        if schedule == "linear":
            self.betas = linear_beta_schedule(timesteps, beta_start, beta_end)
        elif schedule == "cosine":
            self.betas = cosine_beta_schedule(timesteps)
        elif schedule == "quadratic":
            self.betas = quadratic_beta_schedule(timesteps, beta_start, beta_end)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Compute α_t = 1 - β_t
        self.alphas = 1. - self.betas
        
        # Compute ᾱ_t = ∏(α_s) for s=1 to t
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # ᾱ_0 = 1 by convention (no noise at t=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Precompute sampling coefficients
        self._precompute_sampling_coeffs()
        
        # Precompute posterior coefficients for q(x_{t-1} | x_t, x_0)
        self._precompute_posterior_coeffs()
    
    def _precompute_sampling_coeffs(self):
        """Precompute coefficients for ancestral sampling."""
        # For x_{t-1} = coeff1 * (x_t - coeff2 * ε) + σ_t * z
        self.sqrt_recip_alphas = torch.sqrt(1. / self.alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # σ_t options
        # Option 1: σ_t = β_t (original DDPM)
        self.posterior_variance_beta = self.betas
        
        # Option 2: σ_t = √((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t (posterior variance)  
        self.posterior_variance_posterior = (
            self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        
        # Log versions for numerical stability
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance_posterior, min=1e-20)
        )
        
    def _precompute_posterior_coeffs(self):
        """Precompute coefficients for posterior q(x_{t-1} | x_t, x_0)."""
        # μ_t(x_t, x_0) = coeff1 * x_0 + coeff2 * x_t
        self.posterior_mean_coeff1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        )
        self.posterior_mean_coeff2 = (
            (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        )
    
    def get_variance(self, t: torch.Tensor, variance_type: str = "posterior") -> torch.Tensor:
        """
        Get sampling variance σ_t for timestep t.
        
        Args:
            t: Timestep tensor of shape (batch_size,)
            variance_type: "beta" or "posterior"
        
        Returns:
            Variance tensor of shape (batch_size,)
        """
        if variance_type == "beta":
            return extract(self.posterior_variance_beta, t, (1,))
        elif variance_type == "posterior":
            return extract(self.posterior_variance_posterior, t, (1,))
        else:
            raise ValueError(f"Unknown variance type: {variance_type}")
    
    def to(self, device: torch.device):
        """Move all schedules to device."""
        attrs_to_move = [
            'betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
            'sqrt_recip_alphas', 'sqrt_one_minus_alphas_cumprod',
            'posterior_variance_beta', 'posterior_variance_posterior',
            'posterior_log_variance_clipped', 'posterior_mean_coeff1', 'posterior_mean_coeff2'
        ]
        
        for attr in attrs_to_move:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr).to(device))
        
        return self


def extract(buffer: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: tuple) -> torch.Tensor:
    """
    Extract values from buffer at given timesteps and reshape for broadcasting.
    
    This is a key helper function for gathering timestep-specific parameters
    during sampling and training.
    
    Args:
        buffer: Tensor containing values for each timestep, shape (T,) or (T, ...)
        timesteps: Timestep indices, shape (batch_size,) 
        broadcast_shape: Shape to broadcast to, typically (1,) for scalar per batch
                        or (1, 1, 1) for spatial broadcasting
    
    Returns:
        Extracted values reshaped for broadcasting, shape (batch_size, *broadcast_shape)
    
    Example:
        >>> buffer = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])  # T=5
        >>> timesteps = torch.tensor([0, 2, 4])  # batch_size=3
        >>> result = extract(buffer, timesteps, (1,))
        >>> print(result.shape)  # (3, 1)
        >>> print(result.squeeze())  # [0.1, 0.3, 0.5]
    """
    batch_size = timesteps.shape[0]
    extracted = buffer.gather(0, timesteps)
    
    # Reshape for broadcasting: (batch_size, *broadcast_shape)
    out_shape = (batch_size,) + broadcast_shape
    return extracted.reshape(out_shape)


def get_schedule_coefficients(timesteps: int = 1000, schedule: str = "cosine", 
                            device: Optional[torch.device] = None) -> DDPMSchedules:
    """
    Convenience function to create and return DDPM schedules.
    
    Args:
        timesteps: Number of diffusion timesteps
        schedule: Schedule type ("linear", "cosine", "quadratic")
        device: Device to move schedules to
    
    Returns:
        DDPMSchedules object with all precomputed coefficients
    """
    schedules = DDPMSchedules(timesteps, schedule)
    if device is not None:
        schedules = schedules.to(device)
    return schedules


# Alias for backward compatibility with Day 6
def make_ddpm_schedules(*args, **kwargs):
    """Alias for get_schedule_coefficients for Day 6 compatibility."""
    return get_schedule_coefficients(*args, **kwargs)


# Test functions for verification
def test_schedule_properties():
    """Test that schedules have expected properties."""
    schedules = DDPMSchedules(timesteps=1000, schedule="cosine")
    
    # Test shapes
    assert schedules.betas.shape == (1000,)
    assert schedules.alphas.shape == (1000,)
    assert schedules.alphas_cumprod.shape == (1000,)
    
    # Test value ranges
    assert torch.all(schedules.betas > 0) and torch.all(schedules.betas < 1)
    assert torch.all(schedules.alphas > 0) and torch.all(schedules.alphas < 1) 
    assert torch.all(schedules.alphas_cumprod > 0) and torch.all(schedules.alphas_cumprod <= 1)
    
    # Test that alphas_cumprod is decreasing
    assert torch.all(schedules.alphas_cumprod[1:] <= schedules.alphas_cumprod[:-1])
    
    print("✓ All schedule property tests passed")


def test_extract_function():
    """Test the extract function."""
    buffer = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
    timesteps = torch.tensor([0, 2, 4])
    
    # Test 1D broadcast
    result = extract(buffer, timesteps, (1,))
    expected = torch.tensor([[0.1], [0.3], [0.5]])
    assert torch.allclose(result, expected)
    
    # Test 3D broadcast
    result = extract(buffer, timesteps, (1, 1, 1))
    expected = torch.tensor([[[0.1]], [[0.3]], [[0.5]]])
    assert torch.allclose(result, expected)
    
    print("✓ Extract function tests passed")


if __name__ == "__main__":
    test_schedule_properties()
    test_extract_function()
    print("✓ All DDPM schedule tests passed!")
