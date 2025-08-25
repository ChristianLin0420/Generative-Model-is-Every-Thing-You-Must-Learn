"""
DDPM noise schedules: β, α, ᾱ builders with linear/cosine schedules
Provides extract() helper for per-batch timestep indexing
"""

import torch
import numpy as np
from typing import Optional, Tuple


def linear_beta_schedule(num_timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Linear β schedule from Ho et al."""
    return torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)


def cosine_beta_schedule(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine β schedule from Nichol & Dhariwal.
    More stable for high-resolution images.
    """
    steps = num_timesteps + 1
    x = torch.linspace(0, num_timesteps, steps)
    alphas_cumprod = torch.cos(((x / num_timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class DDPMSchedules:
    """
    Container for DDPM noise schedules and helper functions.
    Computes β, α, ᾱ and provides indexing utilities.
    """
    
    def __init__(
        self, 
        num_timesteps: int = 1000, 
        schedule_type: str = "linear",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        cosine_s: float = 0.008,
        device: str = "cpu"
    ):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Build β schedule
        if schedule_type == "linear":
            betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        elif schedule_type == "cosine":
            betas = cosine_beta_schedule(num_timesteps, cosine_s)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
            
        self.betas = betas.to(device)
        
        # Compute α = 1 - β
        self.alphas = 1.0 - self.betas
        
        # Compute ᾱ = ∏α (cumulative product)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Shifted version for previous timestep (ᾱ_{t-1})
        self.alphas_cumprod_prev = torch.cat([
            torch.ones(1, device=device),
            self.alphas_cumprod[:-1]
        ])
        
        # Precompute useful constants for sampling
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Posterior variance: β_tilde = β_t * (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        # Clipped log variance (avoid -inf at t=0)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        # Posterior mean coefficients
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
        
    def extract(self, buf: torch.Tensor, t: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Extract values from buffer at timesteps t and reshape for broadcasting.
        
        Args:
            buf: 1D tensor of values (length num_timesteps)
            t: batch of timestep indices [B]
            shape: target shape for broadcasting (e.g., [B, C, H, W])
            
        Returns:
            Values indexed at timesteps t, reshaped for broadcasting
        """
        batch_size = t.shape[0]
        out = buf.gather(-1, t.to(buf.device)).float()
        # Reshape to [B, 1, 1, 1] for broadcasting with [B, C, H, W]
        return out.reshape(batch_size, *((1,) * (len(shape) - 1))).to(t.device)
        
    def q_sample(
        self, 
        x_start: torch.Tensor, 
        t: torch.Tensor, 
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward process: sample x_t from q(x_t | x_0)
        x_t = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        
    def q_posterior_mean_variance(
        self, 
        x_start: torch.Tensor, 
        x_t: torch.Tensor, 
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute posterior q(x_{t-1} | x_t, x_0) mean and variance
        """
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = self.extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
        
    def get_snr(self) -> torch.Tensor:
        """Signal-to-noise ratio: ᾱ / (1 - ᾱ)"""
        return self.alphas_cumprod / (1.0 - self.alphas_cumprod)
        
    def to(self, device: str):
        """Move all tensors to device"""
        self.device = device
        for attr in dir(self):
            if not attr.startswith('_') and hasattr(self, attr):
                val = getattr(self, attr)
                if isinstance(val, torch.Tensor):
                    setattr(self, attr, val.to(device))
        return self