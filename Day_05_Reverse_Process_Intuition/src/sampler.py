"""Sampling algorithms for DDPM reverse process

Implements:
- DDPM ancestral sampling (stochastic)
- DDIM sampling (deterministic, faster)
- Various starting conditions (prior, teacher-forced)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Callable, Tuple
import numpy as np
from tqdm.auto import tqdm

from .ddpm_schedules import DDPMScheduler


class DDPMSampler:
    """DDPM ancestral sampler for reverse diffusion process.
    
    Implements the reverse process:
    x_{t-1} = μ_θ(x_t, t) + σ_t * z
    
    where μ_θ and σ_t are derived from the learned noise prediction.
    """
    
    def __init__(
        self,
        scheduler: DDPMScheduler,
        prediction_type: str = "epsilon",
        variance_type: str = "fixed_small",
        clip_denoised: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        """Initialize DDPM sampler.
        
        Args:
            scheduler: DDPM noise scheduler
            prediction_type: What model predicts ("epsilon", "x0", "v")
            variance_type: Posterior variance ("fixed_small", "fixed_large", "learned")
            clip_denoised: Whether to clip denoised predictions
            clip_range: Range to clip to if clip_denoised=True
        """
        self.scheduler = scheduler
        self.prediction_type = prediction_type
        self.variance_type = variance_type
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range
        
        # Precompute variance schedule
        if variance_type == "fixed_small":
            # Use β_t (small variance)
            self.posterior_variance = scheduler.betas
        elif variance_type == "fixed_large":
            # Use posterior variance (larger)
            self.posterior_variance = scheduler.posterior_variance
        else:
            # Learned variance not implemented in this version
            self.posterior_variance = scheduler.betas
        
        self.posterior_log_variance = torch.log(self.posterior_variance)
    
    def _predict_x0_from_model_output(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_output: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 from model output based on prediction type.
        
        Args:
            x_t: Current noisy images [B, C, H, W]
            t: Timesteps [B]
            model_output: Model predictions [B, C, H, W]
        
        Returns:
            Predicted x_0 [B, C, H, W]
        """
        if self.prediction_type == "epsilon":
            # x_0 = (x_t - √(1-ᾱ_t) * ε) / √ᾱ_t
            return self.scheduler.predict_start_from_noise(x_t, t, model_output)
        
        elif self.prediction_type == "x0":
            # Direct x_0 prediction
            return model_output
        
        elif self.prediction_type == "v":
            # Convert v-prediction to x_0
            # v = √ᾱ_t * ε - √(1-ᾱ_t) * x_0
            # x_0 = (√ᾱ_t * ε - v) / √(1-ᾱ_t)
            sqrt_alpha_bar_t = self.scheduler.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar_t = self.scheduler.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
            
            # First predict epsilon from v
            epsilon = (model_output + sqrt_one_minus_alpha_bar_t * x_t) / sqrt_alpha_bar_t
            
            # Then predict x_0 from epsilon
            return self.scheduler.predict_start_from_noise(x_t, t, epsilon)
        
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
    
    def _get_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get posterior mean and variance for q(x_{t-1}|x_t, x_0).
        
        Args:
            x_start: Predicted x_0 [B, C, H, W]
            x_t: Current x_t [B, C, H, W]
            t: Timesteps [B]
        
        Returns:
            Posterior mean and variance
        """
        posterior_mean, posterior_variance = self.scheduler.q_posterior_mean_variance(x_start, x_t, t)
        return posterior_mean, posterior_variance
    
    def p_sample_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        generator: Optional[torch.Generator] = None
    ) -> Dict[str, torch.Tensor]:
        """Single reverse diffusion step: x_t -> x_{t-1}.
        
        Args:
            model: Denoising model
            x_t: Current noisy images [B, C, H, W]
            t: Current timesteps [B]
            generator: Random number generator
        
        Returns:
            Dictionary with sampling results
        """
        batch_size = x_t.shape[0]
        device = x_t.device
        
        # Predict model output
        with torch.no_grad():
            model_output = model(x_t, t)
        
        # Predict x_0
        pred_x0 = self._predict_x0_from_model_output(x_t, t, model_output)
        
        # Clip denoised prediction if requested
        if self.clip_denoised:
            pred_x0 = torch.clamp(pred_x0, self.clip_range[0], self.clip_range[1])
        
        # Get posterior mean and variance
        posterior_mean, posterior_variance = self._get_posterior_mean_variance(pred_x0, x_t, t)
        
        # Add noise for t > 0
        if t[0] > 0:  # Assuming all timesteps in batch are the same
            if generator is not None:
                noise = torch.randn(x_t.shape, generator=generator, device=x_t.device, dtype=x_t.dtype)
            else:
                noise = torch.randn_like(x_t)
            posterior_std = torch.sqrt(posterior_variance)
            x_prev = posterior_mean + posterior_std * noise
        else:
            # No noise for final step
            x_prev = posterior_mean
        
        return {
            "x_prev": x_prev,
            "pred_x0": pred_x0,
            "posterior_mean": posterior_mean,
            "posterior_variance": posterior_variance
        }
    
    def p_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        return_trajectory: bool = False,
        progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Full reverse diffusion sampling loop.
        
        Args:
            model: Denoising model
            shape: Shape of images to generate [B, C, H, W]
            num_inference_steps: Number of denoising steps (defaults to full schedule)
            generator: Random number generator
            device: Device to use
            return_trajectory: Whether to return full trajectory
            progress: Whether to show progress bar
        
        Returns:
            Dictionary with sampling results
        """
        if device is None:
            device = next(model.parameters()).device
        
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        
        # Create timestep schedule
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps,
            dtype=torch.long, device=device
        )
        
        # Start from pure noise
        if generator is not None:
            x_t = torch.randn(shape, generator=generator, device=device)
        else:
            x_t = torch.randn(shape, device=device)
        
        trajectory = [x_t.clone()] if return_trajectory else None
        
        # Reverse diffusion loop
        iterator = tqdm(timesteps, desc="Sampling", disable=not progress)
        for t in iterator:
            # Create batch of timesteps
            t_batch = t.repeat(shape[0]).to(device)
            
            # Single denoising step
            step_output = self.p_sample_step(model, x_t, t_batch, generator)
            x_t = step_output["x_prev"]
            
            if return_trajectory:
                trajectory.append(x_t.clone())
            
            # Update progress bar
            if progress:
                iterator.set_postfix({"timestep": t.item()})
        
        result = {"images": x_t}
        if return_trajectory:
            result["trajectory"] = trajectory
        
        return result
    
    def p_sample_from_noise(
        self,
        model: nn.Module,
        x_T: torch.Tensor,
        num_inference_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        return_trajectory: bool = False,
        progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Sample from given initial noise.
        
        Args:
            model: Denoising model
            x_T: Initial noise [B, C, H, W]
            num_inference_steps: Number of denoising steps
            generator: Random number generator
            return_trajectory: Whether to return full trajectory
            progress: Whether to show progress bar
        
        Returns:
            Dictionary with sampling results
        """
        device = x_T.device
        
        if num_inference_steps is None:
            num_inference_steps = self.scheduler.num_timesteps
        
        # Create timestep schedule
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps,
            dtype=torch.long, device=device
        )
        
        x_t = x_T.clone()
        trajectory = [x_t.clone()] if return_trajectory else None
        
        # Reverse diffusion loop
        iterator = tqdm(timesteps, desc="Sampling", disable=not progress)
        for t in iterator:
            # Create batch of timesteps
            t_batch = t.repeat(x_T.shape[0]).to(device)
            
            # Single denoising step
            step_output = self.p_sample_step(model, x_t, t_batch, generator)
            x_t = step_output["x_prev"]
            
            if return_trajectory:
                trajectory.append(x_t.clone())
        
        result = {"images": x_t}
        if return_trajectory:
            result["trajectory"] = trajectory
        
        return result


class DDIMSampler:
    """DDIM sampler for faster deterministic sampling.
    
    Implements the DDIM sampling process which allows for faster
    generation with fewer steps while maintaining high quality.
    """
    
    def __init__(
        self,
        scheduler: DDPMScheduler,
        prediction_type: str = "epsilon",
        clip_denoised: bool = True,
        clip_range: Tuple[float, float] = (-1.0, 1.0)
    ):
        """Initialize DDIM sampler.
        
        Args:
            scheduler: DDPM noise scheduler
            prediction_type: What model predicts ("epsilon", "x0", "v")
            clip_denoised: Whether to clip denoised predictions
            clip_range: Range to clip to if clip_denoised=True
        """
        self.scheduler = scheduler
        self.prediction_type = prediction_type
        self.clip_denoised = clip_denoised
        self.clip_range = clip_range
    
    def ddim_step(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eta: float = 0.0
    ) -> torch.Tensor:
        """Single DDIM step.
        
        Args:
            model: Denoising model
            x_t: Current images [B, C, H, W]
            t: Current timesteps [B]
            t_prev: Previous timesteps [B]
            eta: Stochasticity parameter (0=deterministic, 1=DDPM)
        
        Returns:
            x_{t_prev} [B, C, H, W]
        """
        # Predict noise
        with torch.no_grad():
            model_output = model(x_t, t)
        
        # Get alpha values
        alpha_bar_t = self.scheduler.alpha_bar[t].view(-1, 1, 1, 1)
        alpha_bar_t_prev = self.scheduler.alpha_bar[t_prev].view(-1, 1, 1, 1)
        
        # Handle prediction type
        if self.prediction_type == "epsilon":
            epsilon = model_output
        elif self.prediction_type == "x0":
            # Convert x0 prediction to epsilon
            epsilon = self.scheduler.predict_noise_from_start(x_t, t, model_output)
        elif self.prediction_type == "v":
            # Convert v prediction to epsilon
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            epsilon = (model_output + sqrt_one_minus_alpha_bar_t * x_t) / sqrt_alpha_bar_t
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
        
        # Predict x_0
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * epsilon) / torch.sqrt(alpha_bar_t)
        
        # Clip if requested
        if self.clip_denoised:
            pred_x0 = torch.clamp(pred_x0, self.clip_range[0], self.clip_range[1])
        
        # DDIM formula
        sqrt_alpha_bar_t_prev = torch.sqrt(alpha_bar_t_prev)
        sqrt_one_minus_alpha_bar_t_prev = torch.sqrt(1 - alpha_bar_t_prev)
        
        # Deterministic part
        x_prev_det = sqrt_alpha_bar_t_prev * pred_x0 + sqrt_one_minus_alpha_bar_t_prev * epsilon
        
        # Stochastic part (eta > 0)
        if eta > 0:
            # Compute posterior variance
            alpha_t = self.scheduler.alphas[t].view(-1, 1, 1, 1)
            beta_t = 1 - alpha_t
            
            sigma_t = eta * torch.sqrt(beta_t * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t))
            noise = torch.randn_like(x_t)
            x_prev = x_prev_det + sigma_t * noise
        else:
            x_prev = x_prev_det
        
        return x_prev
    
    def ddim_sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        num_inference_steps: int = 50,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        device: Optional[torch.device] = None,
        return_trajectory: bool = False,
        progress: bool = True
    ) -> Dict[str, torch.Tensor]:
        """DDIM sampling loop.
        
        Args:
            model: Denoising model
            shape: Shape of images to generate [B, C, H, W]
            num_inference_steps: Number of denoising steps
            eta: Stochasticity (0=deterministic, 1=stochastic like DDPM)
            generator: Random number generator
            device: Device to use
            return_trajectory: Whether to return full trajectory
            progress: Whether to show progress bar
        
        Returns:
            Dictionary with sampling results
        """
        if device is None:
            device = next(model.parameters()).device
        
        # Create timestep schedule (uniform spacing)
        timesteps = torch.linspace(
            self.scheduler.num_timesteps - 1, 0, num_inference_steps + 1,
            dtype=torch.long, device=device
        )
        
        # Start from noise
        if generator is not None:
            x_t = torch.randn(shape, generator=generator, device=device)
        else:
            x_t = torch.randn(shape, device=device)
        trajectory = [x_t.clone()] if return_trajectory else None
        
        # DDIM loop
        iterator = tqdm(range(num_inference_steps), desc="DDIM Sampling", disable=not progress)
        for i in iterator:
            t = timesteps[i]
            t_prev = timesteps[i + 1] if i < num_inference_steps - 1 else torch.tensor(0, device=device)
            
            # Create batch tensors
            t_batch = t.repeat(shape[0])
            t_prev_batch = t_prev.repeat(shape[0])
            
            # DDIM step
            x_t = self.ddim_step(model, x_t, t_batch, t_prev_batch, eta)
            
            if return_trajectory:
                trajectory.append(x_t.clone())
        
        result = {"images": x_t}
        if return_trajectory:
            result["trajectory"] = trajectory
        
        return result


def reconstruct_from_noise(
    model: nn.Module,
    scheduler: DDPMScheduler,
    x_start: torch.Tensor,
    t_start: int,
    sampler_type: str = "ddpm",
    **sampler_kwargs
) -> Dict[str, torch.Tensor]:
    """Reconstruct images starting from a specific timestep (teacher-forced).
    
    Useful for evaluation and understanding model behavior.
    
    Args:
        model: Denoising model
        scheduler: DDPM scheduler
        x_start: Clean images [B, C, H, W]
        t_start: Starting timestep for reconstruction
        sampler_type: Type of sampler ("ddpm" or "ddim")
        **sampler_kwargs: Additional sampler arguments
    
    Returns:
        Dictionary with reconstruction results
    """
    device = x_start.device
    batch_size = x_start.shape[0]
    
    # Add noise to get x_t
    timesteps = torch.full((batch_size,), t_start, device=device, dtype=torch.long)
    noise = torch.randn_like(x_start)
    x_t = scheduler.add_noise(x_start, noise, timesteps)
    
    # Sample from x_t
    if sampler_type == "ddpm":
        sampler = DDPMSampler(scheduler)
        # Only sample from t_start to 0
        steps_remaining = t_start + 1
        result = sampler.p_sample_from_noise(
            model, x_t, num_inference_steps=steps_remaining, **sampler_kwargs
        )
    elif sampler_type == "ddim":
        sampler = DDIMSampler(scheduler)
        # Adjust number of steps proportionally
        total_steps = sampler_kwargs.get("num_inference_steps", 50)
        remaining_steps = int(total_steps * (t_start + 1) / scheduler.num_timesteps)
        result = sampler.ddim_sample(
            model, x_t.shape, num_inference_steps=remaining_steps,
            **{k: v for k, v in sampler_kwargs.items() if k != "num_inference_steps"}
        )
    else:
        raise ValueError(f"Unknown sampler_type: {sampler_type}")
    
    result.update({
        "x_start": x_start,
        "x_t": x_t,
        "t_start": t_start,
        "noise": noise
    })
    
    return result


def test_samplers():
    """Test sampling functionality."""
    from .ddpm_schedules import DDPMScheduler
    from .models.unet_tiny import UNetTiny
    
    # Create components
    scheduler = DDPMScheduler(num_timesteps=100)
    model = UNetTiny(in_channels=3, out_channels=3, model_channels=32)
    
    # Test DDPM sampler
    ddpm_sampler = DDPMSampler(scheduler)
    result = ddpm_sampler.p_sample_loop(model, (2, 3, 32, 32), num_inference_steps=10, progress=False)
    print(f"DDPM result shape: {result['images'].shape}")
    
    # Test DDIM sampler
    ddim_sampler = DDIMSampler(scheduler)
    result = ddim_sampler.ddim_sample(model, (2, 3, 32, 32), num_inference_steps=10, progress=False)
    print(f"DDIM result shape: {result['images'].shape}")
    
    # Test reconstruction
    x_start = torch.randn(2, 3, 32, 32)
    result = reconstruct_from_noise(model, scheduler, x_start, t_start=50, sampler_type="ddpm", progress=False)
    print(f"Reconstruction shape: {result['images'].shape}")
    
    print("Sampler tests passed!")


if __name__ == "__main__":
    test_samplers()