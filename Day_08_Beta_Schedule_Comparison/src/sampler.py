"""
DDPM ancestral sampler and DDIM sampler with configurable eta parameter.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import numpy as np
from tqdm import tqdm


class DDPMSampler:
    """
    Ancestral DDPM sampler following the reverse diffusion process.
    """
    
    def __init__(
        self,
        model: nn.Module,
        betas: torch.Tensor,
        alphas: torch.Tensor,
        alpha_bars: torch.Tensor
    ):
        self.model = model
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.T = len(betas)
    
    def sample_step(
        self, 
        x_t: torch.Tensor, 
        t: torch.Tensor, 
        noise_pred: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single reverse diffusion step: p(x_{t-1} | x_t).
        
        Args:
            x_t: Noisy image at timestep t [B, C, H, W]
            t: Timestep [B] 
            noise_pred: Optional pre-computed noise prediction
            
        Returns:
            x_{t-1}: Less noisy image [B, C, H, W]
        """
        if noise_pred is None:
            noise_pred = self.model(x_t, t)
        
        # Get schedule values for timestep t
        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        
        # Compute mean of reverse distribution
        coeff1 = 1.0 / torch.sqrt(alpha_t)
        coeff2 = beta_t / torch.sqrt(1.0 - alpha_bar_t)
        mu_t = coeff1 * (x_t - coeff2 * noise_pred)
        
        # Add noise for t > 0
        if t[0] > 0:
            # Compute variance
            sigma_t = torch.sqrt(beta_t)
            noise = torch.randn_like(x_t)
            x_prev = mu_t + sigma_t * noise
        else:
            # No noise for final step
            x_prev = mu_t
        
        return x_prev
    
    def sample(
        self, 
        shape: Tuple[int, ...], 
        device: torch.device,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Sample images from pure noise using reverse diffusion.
        
        Args:
            shape: Shape of images to sample (B, C, H, W)
            device: Device to sample on
            return_trajectory: Whether to return full trajectory
            
        Returns:
            Sampled images [B, C, H, W] or trajectory if requested
        """
        self.model.eval()
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        if return_trajectory:
            trajectory = [x.clone()]
        
        # Reverse diffusion process
        for t in tqdm(range(self.T - 1, -1, -1), desc="Sampling"):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            with torch.no_grad():
                x = self.sample_step(x, t_tensor)
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return torch.stack(trajectory, dim=1)  # [B, T+1, C, H, W]
        else:
            return x


class DDIMSampler:
    """
    DDIM sampler with configurable eta parameter for deterministic/stochastic sampling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        betas: torch.Tensor,
        alphas: torch.Tensor,
        alpha_bars: torch.Tensor,
        eta: float = 0.0
    ):
        self.model = model
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.eta = eta
        self.T = len(betas)
    
    def sample_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        noise_pred: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        DDIM sampling step from t to t_prev.
        
        Args:
            x_t: Image at timestep t [B, C, H, W]
            t: Current timestep [B]
            t_prev: Previous timestep [B] 
            noise_pred: Optional pre-computed noise prediction
            
        Returns:
            x_{t_prev}: Image at previous timestep [B, C, H, W]
        """
        if noise_pred is None:
            noise_pred = self.model(x_t, t)
        
        # Get alpha_bar values
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)
        alpha_bar_prev = self.alpha_bars[t_prev].view(-1, 1, 1, 1)
        
        # Predict x_0 from noise prediction
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t
        
        # Compute direction to x_t
        sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
        sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)
        
        # Deterministic direction
        dir_to_xt = sqrt_one_minus_alpha_bar_prev * noise_pred
        
        # Add stochasticity if eta > 0
        if self.eta > 0 and t_prev[0] > 0:
            sigma = self.eta * torch.sqrt(
                (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * 
                (1.0 - alpha_bar_t / alpha_bar_prev)
            )
            noise = torch.randn_like(x_t)
            stochastic_noise = sigma.view(-1, 1, 1, 1) * noise
        else:
            stochastic_noise = 0.0
        
        # DDIM update
        x_prev = sqrt_alpha_bar_prev * x_0_pred + dir_to_xt + stochastic_noise
        
        return x_prev
    
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        num_steps: int = 50,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Sample using DDIM with specified number of steps.
        
        Args:
            shape: Shape of images to sample
            device: Device to sample on
            num_steps: Number of sampling steps (< T for acceleration)
            return_trajectory: Whether to return full trajectory
            
        Returns:
            Sampled images or trajectory
        """
        self.model.eval()
        
        # Create sampling schedule
        if num_steps >= self.T:
            timesteps = list(range(self.T - 1, -1, -1))
        else:
            # Uniform spacing
            timesteps = np.linspace(self.T - 1, 0, num_steps, dtype=int)
            timesteps = list(timesteps)
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        if return_trajectory:
            trajectory = [x.clone()]
        
        # DDIM sampling loop
        for i, t in enumerate(tqdm(timesteps, desc=f"DDIM Sampling (η={self.eta})")):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            # Get previous timestep
            if i == len(timesteps) - 1:
                t_prev = 0
            else:
                t_prev = timesteps[i + 1]
            
            t_prev_tensor = torch.full((shape[0],), t_prev, device=device, dtype=torch.long)
            
            with torch.no_grad():
                x = self.sample_step(x, t_tensor, t_prev_tensor)
            
            if return_trajectory:
                trajectory.append(x.clone())
        
        if return_trajectory:
            return torch.stack(trajectory, dim=1)  # [B, steps+1, C, H, W]
        else:
            return x


def test_samplers():
    """Test both samplers."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy model and schedule
    from .models.unet_small import UNetSmall
    from .schedules import beta_linear
    
    model = UNetSmall(in_channels=1, base_channels=32).to(device)
    schedule = beta_linear(100)  # Short schedule for testing
    
    betas = schedule['betas'].to(device)
    alphas = schedule['alphas'].to(device)
    alpha_bars = schedule['alpha_bars'].to(device)
    
    # Test DDPM sampler
    ddpm_sampler = DDPMSampler(model, betas, alphas, alpha_bars)
    
    with torch.no_grad():
        ddpm_samples = ddpm_sampler.sample((4, 1, 28, 28), device)
    
    print(f"DDPM samples shape: {ddpm_samples.shape}")
    print(f"DDPM samples range: [{ddpm_samples.min():.3f}, {ddmp_samples.max():.3f}]")
    
    # Test DDIM sampler
    ddim_sampler = DDIMSampler(model, betas, alphas, alpha_bars, eta=0.0)
    
    with torch.no_grad():
        ddim_samples = ddim_sampler.sample((4, 1, 28, 28), device, num_steps=20)
    
    print(f"DDIM samples shape: {ddim_samples.shape}")
    print(f"DDIM samples range: [{ddim_samples.min():.3f}, {ddim_samples.max():.3f}]")
    
    # Test trajectory sampling
    with torch.no_grad():
        trajectory = ddpm_sampler.sample((2, 1, 28, 28), device, return_trajectory=True)
    
    print(f"Trajectory shape: {trajectory.shape}")
    
    print("Sampler tests completed!")


if __name__ == "__main__":
    test_samplers()
