"""
DDPM Ancestral Sampler Implementation

Core implementation of the ancestral sampling procedure:
1. Start from pure noise x_T ~ N(0, I)  
2. Iteratively denoise using the learned model ε_θ(x_t, t)
3. Apply proper variance scheduling

Mathematical formulation:
ε̂ = ε_θ(x_t, t)
x̂_0 = (x_t - √(1-ᾱ_t) * ε̂) / √ᾱ_t  
x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε̂) + σ_t * z

With σ_t = β_t or posterior variance option.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Callable, Dict, Any
from tqdm import tqdm
import math

from .ddpm_schedules import DDPMSchedules, extract


class DDPMSampler:
    """
    Ancestral DDPM sampler with optional DDIM support.
    
    Implements the core sampling loop with proper variance handling,
    trajectory recording, and various sampling options.
    """
    
    def __init__(self, model: nn.Module, schedules: DDPMSchedules, 
                 device: Optional[torch.device] = None):
        """
        Initialize DDPM sampler.
        
        Args:
            model: Trained DDPM model that predicts noise ε_θ(x_t, t)
            schedules: DDPM schedules with precomputed coefficients  
            device: Device to run sampling on
        """
        self.model = model
        self.schedules = schedules
        self.device = device or next(model.parameters()).device
        self.T = schedules.timesteps
        
        # Move schedules to device
        self.schedules.to(self.device)
        
        # Sampling statistics
        self.last_sampling_info = {}
    
    @torch.no_grad()
    def ancestral_step(self, x_t: torch.Tensor, t: torch.Tensor, 
                      variance_type: str = "posterior") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single ancestral sampling step: x_t -> x_{t-1}.
        
        Args:
            x_t: Current noisy sample, shape (B, C, H, W)
            t: Current timestep, shape (B,) with values in [0, T-1]
            variance_type: "beta" or "posterior" variance
            
        Returns:
            Tuple of (x_{t-1}, x_0_pred, eps_pred)
            - x_{t-1}: Denoised sample at t-1
            - x_0_pred: Predicted clean image x_0
            - eps_pred: Predicted noise ε
        """
        batch_size = x_t.shape[0]
        
        # Predict noise ε_θ(x_t, t)
        eps_pred = self.model(x_t, t)
        
        # Extract coefficients for this timestep
        sqrt_recip_alpha_t = extract(self.schedules.sqrt_recip_alphas, t, (1, 1, 1))
        sqrt_one_minus_alpha_cumprod_t = extract(self.schedules.sqrt_one_minus_alphas_cumprod, t, (1, 1, 1))
        beta_t = extract(self.schedules.betas, t, (1, 1, 1))
        alpha_cumprod_t = extract(self.schedules.alphas_cumprod, t, (1, 1, 1))
        
        # Predict x_0 from x_t and predicted noise
        x_0_pred = (x_t - sqrt_one_minus_alpha_cumprod_t * eps_pred) / torch.sqrt(alpha_cumprod_t)
        
        # Clip x_0 to valid range (optional)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # Compute mean of x_{t-1}
        # μ_t = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε̂)
        mean = sqrt_recip_alpha_t * (x_t - beta_t / sqrt_one_minus_alpha_cumprod_t * eps_pred)
        
        # Add noise for t > 0
        if torch.any(t > 0):
            # Get variance σ_t
            if variance_type == "beta":
                variance = extract(self.schedules.betas, t, (1, 1, 1))
            elif variance_type == "posterior": 
                variance = extract(self.schedules.posterior_variance_posterior, t, (1, 1, 1))
            else:
                raise ValueError(f"Unknown variance type: {variance_type}")
            
            # Sample noise z ~ N(0, I)
            noise = torch.randn_like(x_t)
            
            # Set noise to zero for t=0 (no noise at final step)
            noise = torch.where((t == 0).view(-1, 1, 1, 1), 0., noise)
            
            # Add scaled noise: x_{t-1} = μ_t + σ_t * z  
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            # No noise for final step (t=0)
            x_t_minus_1 = mean
        
        return x_t_minus_1, x_0_pred, eps_pred
    
    @torch.no_grad()
    def ddim_step(self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor,
                  eta: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single DDIM sampling step for faster sampling.
        
        Args:
            x_t: Current sample, shape (B, C, H, W)
            t: Current timestep, shape (B,)
            t_prev: Previous timestep, shape (B,)  
            eta: DDIM interpolation parameter (0=deterministic, 1=DDPM)
            
        Returns:
            Tuple of (x_{t_prev}, x_0_pred, eps_pred)
        """
        # Predict noise
        eps_pred = self.model(x_t, t)
        
        # Extract coefficients
        alpha_cumprod_t = extract(self.schedules.alphas_cumprod, t, (1, 1, 1))
        alpha_cumprod_t_prev = extract(self.schedules.alphas_cumprod, t_prev, (1, 1, 1))
        
        # Predict x_0
        x_0_pred = (x_t - torch.sqrt(1 - alpha_cumprod_t) * eps_pred) / torch.sqrt(alpha_cumprod_t)
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        # DDIM sampling formula
        sqrt_alpha_cumprod_t_prev = torch.sqrt(alpha_cumprod_t_prev)
        sqrt_one_minus_alpha_cumprod_t_prev = torch.sqrt(1 - alpha_cumprod_t_prev)
        
        # Deterministic part
        x_t_prev = sqrt_alpha_cumprod_t_prev * x_0_pred + sqrt_one_minus_alpha_cumprod_t_prev * eps_pred
        
        # Add noise term for eta > 0 
        if eta > 0:
            sigma_t = eta * torch.sqrt((1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t)) * torch.sqrt(1 - alpha_cumprod_t / alpha_cumprod_t_prev)
            noise = torch.randn_like(x_t)
            x_t_prev = x_t_prev + sigma_t * noise
        
        return x_t_prev, x_0_pred, eps_pred
    
    @torch.no_grad()
    def sample(self, shape: Tuple[int, ...], 
               num_steps: Optional[int] = None,
               variance_type: str = "posterior",
               return_trajectory: bool = False,
               trajectory_steps: Optional[List[int]] = None,
               progress: bool = True,
               ddim: bool = False,
               ddim_eta: float = 0.0,
               clip_denoised: bool = True) -> Dict[str, torch.Tensor]:
        """
        Generate samples using ancestral sampling or DDIM.
        
        Args:
            shape: Shape of samples to generate (B, C, H, W)
            num_steps: Number of sampling steps (default: full T steps)
            variance_type: "beta" or "posterior" 
            return_trajectory: Whether to return intermediate states
            trajectory_steps: Specific steps to record (default: every 50 steps)
            progress: Whether to show progress bar
            ddim: Whether to use DDIM instead of ancestral sampling
            ddim_eta: DDIM eta parameter (0=deterministic)  
            clip_denoised: Whether to clip predicted x_0
            
        Returns:
            Dictionary containing:
            - 'samples': Final samples, shape (B, C, H, W)
            - 'trajectory': List of intermediate samples (if requested)
            - 'trajectory_steps': Timesteps corresponding to trajectory
            - 'x0_preds': List of x_0 predictions (if requested) 
            - 'eps_preds': List of noise predictions (if requested)
        """
        batch_size = shape[0]
        
        # Set up sampling schedule
        if num_steps is None or num_steps == self.T:
            # Full schedule
            timesteps = list(reversed(range(self.T)))
            full_sampling = True
        else:
            # Uniform subsampling for DDIM
            if ddim:
                step_size = self.T // num_steps
                timesteps = list(reversed(range(0, self.T, step_size)))
                full_sampling = False
            else:
                raise ValueError("Subsampling only supported for DDIM")
        
        # Initialize from pure noise
        x = torch.randn(shape, device=self.device)
        
        # Trajectory recording
        trajectory = []
        trajectory_timesteps = []
        x0_preds = []
        eps_preds = []
        
        if trajectory_steps is None:
            trajectory_steps = list(range(0, self.T, max(1, self.T // 20)))  # ~20 frames
        
        # Sampling loop
        iterator = tqdm(timesteps, desc="DDPM Sampling") if progress else timesteps
        
        for i, t_val in enumerate(iterator):
            t = torch.full((batch_size,), t_val, device=self.device, dtype=torch.long)
            
            if ddim and not full_sampling:
                # DDIM step
                t_prev_val = timesteps[i + 1] if i + 1 < len(timesteps) else 0
                t_prev = torch.full((batch_size,), t_prev_val, device=self.device, dtype=torch.long)
                x, x_0_pred, eps_pred = self.ddim_step(x, t, t_prev, ddim_eta)
            else:
                # Ancestral step
                x, x_0_pred, eps_pred = self.ancestral_step(x, t, variance_type)
            
            # Record trajectory at specified steps
            if return_trajectory and (t_val in trajectory_steps or t_val == 0):
                trajectory.append(x.clone())
                trajectory_timesteps.append(t_val)
                if return_trajectory:
                    x0_preds.append(x_0_pred.clone())
                    eps_preds.append(eps_pred.clone())
            
            # Update progress bar
            if progress and hasattr(iterator, 'set_postfix'):
                iterator.set_postfix({
                    't': t_val,
                    'x_mean': f"{x.mean().item():.3f}",
                    'x_std': f"{x.std().item():.3f}"
                })
        
        # Store sampling info
        self.last_sampling_info = {
            'num_steps': len(timesteps),
            'variance_type': variance_type,
            'ddim': ddim,
            'ddim_eta': ddim_eta,
            'final_mean': x.mean().item(),
            'final_std': x.std().item(),
        }
        
        # Prepare results
        results = {
            'samples': x,
            'trajectory': trajectory if return_trajectory else None,
            'trajectory_steps': trajectory_timesteps if return_trajectory else None,
            'x0_preds': x0_preds if return_trajectory else None,
            'eps_preds': eps_preds if return_trajectory else None,
            'sampling_info': self.last_sampling_info
        }
        
        return results
    
    @torch.no_grad()
    def sample_single_trajectory(self, shape: Tuple[int, ...],
                               record_every: int = 10,
                               **kwargs) -> Dict[str, torch.Tensor]:
        """
        Generate a single sample with full trajectory recording.
        Useful for animations and detailed analysis.
        
        Args:
            shape: Shape of single sample (1, C, H, W)
            record_every: Record every N steps
            **kwargs: Additional sampling arguments
            
        Returns:
            Full trajectory dictionary
        """
        if shape[0] != 1:
            raise ValueError("Single trajectory requires batch size of 1")
        
        # Create dense trajectory steps
        trajectory_steps = list(range(0, self.T, record_every)) + [0]
        trajectory_steps = sorted(set(trajectory_steps), reverse=True)
        
        return self.sample(
            shape=shape,
            return_trajectory=True, 
            trajectory_steps=trajectory_steps,
            **kwargs
        )
    
    def get_sampling_stats(self) -> Dict[str, Any]:
        """Get statistics from the last sampling run."""
        return self.last_sampling_info.copy()


def create_sampler(model: nn.Module, config: Dict[str, Any], 
                  device: Optional[torch.device] = None) -> DDPMSampler:
    """
    Create DDPM sampler from model and config.
    
    Args:
        model: Trained DDPM model
        config: Configuration dictionary with diffusion parameters
        device: Device to run on
        
    Returns:
        Configured DDPMSampler instance
    """
    # Extract diffusion config
    diffusion_config = config.get('diffusion', {})
    T = diffusion_config.get('T', 1000)
    schedule = diffusion_config.get('schedule', 'cosine')
    
    # Create schedules
    schedules = DDPMSchedules(timesteps=T, schedule=schedule)
    
    # Create sampler
    sampler = DDPMSampler(model, schedules, device)
    
    return sampler


# Utility functions for sampling workflows

def sample_grid(sampler: DDPMSampler, num_images: int, image_shape: Tuple[int, int, int],
               **sampling_kwargs) -> torch.Tensor:
    """Sample a grid of images."""
    shape = (num_images, *image_shape)
    results = sampler.sample(shape, **sampling_kwargs)
    return results['samples']


def sample_trajectory_animation(sampler: DDPMSampler, image_shape: Tuple[int, int, int],
                              record_every: int = 10, **sampling_kwargs) -> Dict[str, Any]:
    """Sample a single trajectory for animation."""
    shape = (1, *image_shape)
    return sampler.sample_single_trajectory(shape, record_every=record_every, **sampling_kwargs)


def compare_variance_schedules(sampler: DDPMSampler, shape: Tuple[int, ...],
                             variance_types: List[str] = ["beta", "posterior"]) -> Dict[str, torch.Tensor]:
    """Compare different variance schedules."""
    results = {}
    
    for var_type in variance_types:
        print(f"Sampling with {var_type} variance...")
        sample_result = sampler.sample(shape, variance_type=var_type, progress=False)
        results[var_type] = sample_result['samples']
    
    return results
