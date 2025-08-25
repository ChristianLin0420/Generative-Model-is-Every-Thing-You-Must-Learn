"""
DDPM and DDIM sampling from trained models
Supports ancestral sampling and deterministic DDIM with various eta values
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Callable, List, Union
import numpy as np
from tqdm import tqdm

from .ddpm_schedules import DDPMSchedules


class DDPMSampler:
    """
    DDPM sampling with both ancestral and DDIM methods
    """
    
    def __init__(self, schedules: DDPMSchedules):
        self.schedules = schedules
        
    def ddpm_sample_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        t_prev: Optional[torch.Tensor] = None,
        clip_denoised: bool = True,
        variance_type: str = "learned"
    ) -> torch.Tensor:
        """
        Single step of DDPM ancestral sampling
        
        Args:
            model: denoising model ε_θ(x_t, t)
            x: current sample x_t [B, C, H, W]
            t: current timestep [B]
            t_prev: previous timestep [B], defaults to t-1
            clip_denoised: whether to clip predicted x_0 to [-1, 1]
            variance_type: "fixed_small", "fixed_large", or "learned"
            
        Returns:
            x_{t-1}: previous timestep sample
        """
        batch_size = x.shape[0]
        
        if t_prev is None:
            t_prev = torch.clamp(t - 1, min=0)
            
        # Model prediction
        with torch.no_grad():
            eps_pred = model(x, t)
            
        # Predict x_0 from eps_pred
        sqrt_alphas_cumprod_t = self.schedules.extract(
            self.schedules.sqrt_alphas_cumprod, t, x.shape
        )
        sqrt_one_minus_alphas_cumprod_t = self.schedules.extract(
            self.schedules.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * eps_pred) / sqrt_alphas_cumprod_t
        
        # Clip predicted x_0 if requested
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
        # Compute posterior mean and variance
        posterior_mean, posterior_variance, posterior_log_variance_clipped = \
            self.schedules.q_posterior_mean_variance(pred_x0, x, t)
            
        # Add noise (except for t=0)
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        
        if variance_type == "fixed_small":
            variance = posterior_variance
        elif variance_type == "fixed_large":
            variance = self.schedules.extract(self.schedules.betas, t, x.shape)
        else:  # "learned" - use posterior variance for now
            variance = posterior_variance
            
        sample = posterior_mean + nonzero_mask * torch.sqrt(variance) * noise
        
        return sample
        
    def ddim_sample_step(
        self,
        model: nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0,
        clip_denoised: bool = True
    ) -> torch.Tensor:
        """
        Single step of DDIM sampling
        
        Args:
            model: denoising model
            x: current sample x_t
            t: current timestep
            t_prev: previous timestep
            eta: stochasticity parameter (0=deterministic, 1=DDPM)
            clip_denoised: whether to clip predicted x_0
            
        Returns:
            x_{t_prev}: sample at previous timestep
        """
        # Model prediction
        with torch.no_grad():
            eps_pred = model(x, t)
            
        # Extract schedule values
        alpha_bar_t = self.schedules.extract(self.schedules.alphas_cumprod, t, x.shape)
        alpha_bar_t_prev = self.schedules.extract(self.schedules.alphas_cumprod, t_prev, x.shape)
        
        # Predict x_0
        pred_x0 = (x - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)
        
        # Clip if requested
        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1, 1)
            
        # Compute direction pointing to x_t
        direction_to_xt = torch.sqrt(1 - alpha_bar_t_prev - eta**2 * (1 - alpha_bar_t)) * eps_pred
        
        # Compute x_{t-1}
        x_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0 + direction_to_xt
        
        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(x)
            variance = eta**2 * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
            x_prev = x_prev + torch.sqrt(variance) * noise
            
        return x_prev
        
    def ddpm_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        device: torch.device,
        clip_denoised: bool = True,
        progress: bool = True,
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Full DDPM ancestral sampling
        
        Args:
            model: trained denoising model
            shape: shape of samples to generate [B, C, H, W]
            device: device to run on
            clip_denoised: whether to clip predicted x_0
            progress: whether to show progress bar
            return_trajectory: whether to return full sampling trajectory
            
        Returns:
            samples: generated samples [B, C, H, W]
            trajectory: list of intermediate samples (if return_trajectory=True)
        """
        model.eval()
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        # Sampling loop
        timesteps = torch.arange(self.schedules.num_timesteps - 1, -1, -1, device=device)
        
        iterator = tqdm(timesteps) if progress else timesteps
        
        for i, t in enumerate(iterator):
            # Ensure t is a scalar (not tensor) for torch.full
            t = t.item() if torch.is_tensor(t) else t
            batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            
            x = self.ddpm_sample_step(
                model=model,
                x=x,
                t=batch_t,
                clip_denoised=clip_denoised
            )
            
            if return_trajectory:
                trajectory.append(x.clone())
                
            if progress:
                iterator.set_description(f"DDPM Sampling (t={t})")
                
        if return_trajectory:
            return x, trajectory
        return x
        
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        num_steps: int = 50,
        eta: float = 0.0,
        device: torch.device = None,
        clip_denoised: bool = True,
        progress: bool = True,
        return_trajectory: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        DDIM sampling with configurable number of steps
        
        Args:
            model: trained denoising model
            shape: shape of samples to generate
            num_steps: number of denoising steps (< num_timesteps)
            eta: stochasticity (0=deterministic, 1=DDPM-like)
            device: device to run on
            clip_denoised: whether to clip predicted x_0
            progress: whether to show progress bar
            return_trajectory: whether to return sampling trajectory
            
        Returns:
            samples: generated samples
            trajectory: sampling trajectory (if requested)
        """
        model.eval()
        
        if device is None:
            device = next(model.parameters()).device
            
        # Create timestep schedule
        # Use uniform spacing for DDIM
        timesteps = np.linspace(0, self.schedules.num_timesteps - 1, num_steps + 1).astype(int)
        timesteps = torch.from_numpy(timesteps[::-1].copy()).to(device)  # Reverse order
        
        # Start from pure noise
        x = torch.randn(shape, device=device)
        
        trajectory = [x.clone()] if return_trajectory else None
        
        # Sampling loop
        iterator = zip(timesteps[:-1], timesteps[1:])
        if progress:
            iterator = tqdm(list(iterator), desc="DDIM Sampling")
            
        for t, t_prev in iterator:
            # Ensure t and t_prev are scalars (not tensors) for torch.full
            t = t.item() if torch.is_tensor(t) else t
            t_prev = t_prev.item() if torch.is_tensor(t_prev) else t_prev
            
            batch_t = torch.full((shape[0],), t, device=device, dtype=torch.long)
            batch_t_prev = torch.full((shape[0],), t_prev, device=device, dtype=torch.long)
            
            x = self.ddim_sample_step(
                model=model,
                x=x,
                t=batch_t,
                t_prev=batch_t_prev,
                eta=eta,
                clip_denoised=clip_denoised
            )
            
            if return_trajectory:
                trajectory.append(x.clone())
                
        if return_trajectory:
            return x, trajectory
        return x
        
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        method: str = "ddim",
        num_steps: Optional[int] = None,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
        clip_denoised: bool = True,
        progress: bool = True,
        return_trajectory: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Unified sampling interface
        
        Args:
            model: trained denoising model
            shape: sample shape
            method: "ddpm" or "ddim"
            num_steps: number of steps (for DDIM), uses full timesteps for DDPM
            eta: stochasticity parameter (for DDIM)
            device: device to run on
            clip_denoised: whether to clip x_0 predictions
            progress: show progress bar
            return_trajectory: return intermediate samples
            
        Returns:
            samples and optionally trajectory
        """
        if device is None:
            device = next(model.parameters()).device
            
        if method.lower() == "ddpm":
            return self.ddpm_sample(
                model=model,
                shape=shape,
                device=device,
                clip_denoised=clip_denoised,
                progress=progress,
                return_trajectory=return_trajectory
            )
        elif method.lower() == "ddim":
            if num_steps is None:
                num_steps = 50  # Default DDIM steps
            return self.ddim_sample(
                model=model,
                shape=shape,
                num_steps=num_steps,
                eta=eta,
                device=device,
                clip_denoised=clip_denoised,
                progress=progress,
                return_trajectory=return_trajectory
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")
            
    def interpolate_samples(
        self,
        model: nn.Module,
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_interpolations: int = 8,
        method: str = "ddim",
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        Interpolate between two samples in latent space
        
        Args:
            model: trained model
            x1, x2: start and end samples [C, H, W] 
            num_interpolations: number of interpolation points
            method: sampling method
            num_steps: number of denoising steps
            
        Returns:
            interpolated_samples: [num_interpolations, C, H, W]
        """
        device = x1.device
        
        # Create interpolation weights
        alphas = torch.linspace(0, 1, num_interpolations, device=device)
        
        # Start with noise for each interpolation
        batch_shape = (num_interpolations, *x1.shape)
        noise = torch.randn(batch_shape, device=device)
        
        # Interpolate in noise space
        x1_batch = x1.unsqueeze(0).expand(num_interpolations, -1, -1, -1)
        x2_batch = x2.unsqueeze(0).expand(num_interpolations, -1, -1, -1)
        alphas = alphas.view(-1, 1, 1, 1)
        
        # Linear interpolation in noise space
        interpolated_noise = (1 - alphas) * x1_batch + alphas * x2_batch
        
        # This is a simplified interpolation - proper implementation would
        # require interpolating in the noise space and then denoising
        
        return interpolated_noise
        
    def sample_with_guidance(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        guidance_fn: Optional[Callable] = None,
        guidance_scale: float = 1.0,
        method: str = "ddim",
        num_steps: int = 50,
        device: Optional[torch.device] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sampling with classifier guidance (if guidance function provided)
        
        Args:
            model: denoising model
            shape: sample shape
            guidance_fn: optional guidance function
            guidance_scale: strength of guidance
            method: sampling method
            num_steps: number of steps
            device: device
            
        Returns:
            guided samples
        """
        # This is a placeholder for classifier guidance
        # Full implementation would require integrating gradients from guidance_fn
        
        return self.sample(
            model=model,
            shape=shape,
            method=method,
            num_steps=num_steps,
            device=device,
            **kwargs
        )
        
    def compute_likelihood(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        num_samples: int = 1000
    ) -> torch.Tensor:
        """
        Estimate log-likelihood using importance sampling
        This is computationally expensive and mainly for analysis
        """
        device = x0.device
        batch_size = x0.shape[0]
        
        # Sample timesteps
        t = torch.randint(0, self.schedules.num_timesteps, (num_samples * batch_size,), device=device)
        
        # Expand x0 for all samples
        x0_expanded = x0.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)
        x0_expanded = x0_expanded.reshape(-1, *x0.shape[1:])
        
        # Sample noise
        noise = torch.randn_like(x0_expanded)
        
        # Forward process
        x_t = self.schedules.q_sample(x0_expanded, t, noise)
        
        # Model prediction
        with torch.no_grad():
            eps_pred = model(x_t, t)
            
        # Compute negative log-likelihood (simplified)
        mse = torch.nn.functional.mse_loss(eps_pred, noise, reduction='none')
        mse = mse.flatten(1).mean(dim=1)  # Average over spatial dims
        
        # Reshape back to [batch_size, num_samples]
        mse = mse.view(batch_size, num_samples)
        
        # Estimate log-likelihood (negative because lower MSE = higher likelihood)
        log_likelihood = -mse.mean(dim=1)
        
        return log_likelihood