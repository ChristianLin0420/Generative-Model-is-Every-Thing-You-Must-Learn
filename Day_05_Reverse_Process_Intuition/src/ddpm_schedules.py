"""DDPM Schedules and Noise Management

Implements beta schedules, alpha computations, and noise scheduling
for the forward and reverse diffusion processes.
"""

import torch
import numpy as np
from typing import Tuple, Union
import math


def make_beta_schedule(
    num_timesteps: int, 
    schedule: str = "linear",
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    cosine_s: float = 0.008
) -> torch.Tensor:
    """Create beta schedule for diffusion process.
    
    Args:
        num_timesteps: Total number of diffusion steps T
        schedule: Type of schedule ('linear', 'cosine', 'quadratic', 'sigmoid')
        beta_start: Starting beta value (for linear schedule)
        beta_end: Ending beta value (for linear schedule)
        cosine_s: Small offset for cosine schedule
    
    Returns:
        Beta values for timesteps [1, 2, ..., T]
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
    
    elif schedule == "cosine":
        # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
        timesteps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
        alphas_cumprod = torch.cos((timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clip(betas, 0.0001, 0.9999)
    
    elif schedule == "quadratic":
        # Quadratic schedule
        betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_timesteps, dtype=torch.float32) ** 2
        return betas
    
    elif schedule == "sigmoid":
        # Sigmoid schedule
        betas = torch.linspace(-6, 6, num_timesteps, dtype=torch.float32)
        betas = torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        return betas
    
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def compute_alpha_schedule(betas: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute alpha and alpha_bar (cumulative product) from betas.
    
    Args:
        betas: Beta values [β₁, β₂, ..., βₜ]
    
    Returns:
        alphas: α_t = 1 - β_t
        alpha_bar: ᾱ_t = ∏(α_s) for s=1 to t
    """
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """Extract values from tensor a at indices t and reshape for broadcasting.
    
    This is a key utility for indexing schedule parameters by timestep.
    
    Args:
        a: Tensor to extract from (length T)
        t: Timestep indices (batch_size,)
        x_shape: Shape to broadcast to (typically image shape)
    
    Returns:
        Values from a[t] reshaped for broadcasting with x_shape
    
    Example:
        >>> alphas = torch.tensor([0.99, 0.98, 0.97])
        >>> t = torch.tensor([0, 2, 1])  # batch of timesteps
        >>> x_shape = (3, 1, 32, 32)
        >>> result = extract(alphas, t, x_shape)
        >>> print(result.shape)  # torch.Size([3, 1, 1, 1])
        >>> print(result.squeeze())  # tensor([0.99, 0.97, 0.98])
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t).float()
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DDPMScheduler:
    """DDPM noise scheduler for forward and reverse processes."""
    
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon"
    ):
        """Initialize DDPM scheduler.
        
        Args:
            num_timesteps: Total diffusion steps T
            beta_schedule: Type of beta schedule
            beta_start: Starting beta value
            beta_end: Ending beta value
            prediction_type: What the model predicts ("epsilon", "x0", "v")
        """
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type
        
        # Create beta schedule
        self.betas = make_beta_schedule(
            num_timesteps, beta_schedule, beta_start, beta_end
        )
        
        # Compute alpha schedules
        self.alphas, self.alpha_bar = compute_alpha_schedule(self.betas)
        
        # Compute useful derived quantities
        self.alpha_bar_prev = torch.cat([torch.ones(1), self.alpha_bar[:-1]])
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bar)
        self.sqrt_recip_alpha_bar = torch.sqrt(1.0 / self.alpha_bar)
        self.sqrt_recipm1_alpha_bar = torch.sqrt(1.0 / self.alpha_bar - 1.0)
        
        # For posterior q(x_{t-1}|x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_bar_prev) / (1.0 - self.alpha_bar)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bar_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bar)
        )
    
    def add_noise(
        self, 
        x_start: torch.Tensor, 
        noise: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to clean images (forward process).
        
        Implements: x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        
        Args:
            x_start: Clean images [B, C, H, W]
            noise: Random noise ε ~ N(0, I) [B, C, H, W]  
            timesteps: Timesteps t [B]
        
        Returns:
            Noisy images x_t [B, C, H, W]
        """
        sqrt_alpha_bar_t = extract(self.sqrt_alpha_bar, timesteps, x_start.shape)
        sqrt_one_minus_alpha_bar_t = extract(
            self.sqrt_one_minus_alpha_bar, timesteps, x_start.shape
        )
        
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise
    
    def get_velocity(
        self, 
        x_start: torch.Tensor, 
        noise: torch.Tensor, 
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Compute velocity parameterization v_t.
        
        Implements: v_t = √ᾱ_t * ε - √(1-ᾱ_t) * x_0
        
        Args:
            x_start: Clean images [B, C, H, W]
            noise: Random noise ε [B, C, H, W]
            timesteps: Timesteps t [B]
        
        Returns:
            Velocity v_t [B, C, H, W]
        """
        sqrt_alpha_bar_t = extract(self.sqrt_alpha_bar, timesteps, x_start.shape)
        sqrt_one_minus_alpha_bar_t = extract(
            self.sqrt_one_minus_alpha_bar, timesteps, x_start.shape
        )
        
        return sqrt_alpha_bar_t * noise - sqrt_one_minus_alpha_bar_t * x_start
    
    def predict_start_from_noise(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise ε.
        
        Implements: x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
        
        Args:
            x_t: Noisy images [B, C, H, W]
            t: Timesteps [B]
            noise: Predicted noise [B, C, H, W]
        
        Returns:
            Predicted x_0 [B, C, H, W]
        """
        sqrt_recip_alpha_bar_t = extract(self.sqrt_recip_alpha_bar, t, x_t.shape)
        sqrt_recipm1_alpha_bar_t = extract(self.sqrt_recipm1_alpha_bar, t, x_t.shape)
        
        return sqrt_recip_alpha_bar_t * x_t - sqrt_recipm1_alpha_bar_t * noise
    
    def predict_noise_from_start(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        x_start: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise ε from x_t and x_0.
        
        Implements: ε = (x_t - √ᾱ_t * x_0) / √(1-ᾱ_t)
        
        Args:
            x_t: Noisy images [B, C, H, W]
            t: Timesteps [B]
            x_start: Clean images [B, C, H, W]
        
        Returns:
            Predicted noise [B, C, H, W]
        """
        sqrt_alpha_bar_t = extract(self.sqrt_alpha_bar, t, x_t.shape)
        sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alpha_bar, t, x_t.shape)
        
        return (x_t - sqrt_alpha_bar_t * x_start) / sqrt_one_minus_alpha_bar_t
    
    def q_posterior_mean_variance(
        self, 
        x_start: torch.Tensor, 
        x_t: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance of posterior q(x_{t-1}|x_t, x_0).
        
        Args:
            x_start: Clean images [B, C, H, W]
            x_t: Noisy images [B, C, H, W]
            t: Timesteps [B]
        
        Returns:
            Posterior mean and variance
        """
        posterior_mean_coef1_t = extract(self.posterior_mean_coef1, t, x_t.shape)
        posterior_mean_coef2_t = extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_variance_t = extract(self.posterior_variance, t, x_t.shape)
        
        posterior_mean = (
            posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        )
        
        return posterior_mean, posterior_variance_t
    
    def to(self, device: torch.device) -> 'DDPMScheduler':
        """Move scheduler tensors to device."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))
        return self