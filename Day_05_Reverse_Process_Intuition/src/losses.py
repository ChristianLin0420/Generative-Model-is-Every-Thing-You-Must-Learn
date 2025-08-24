"""Loss functions for DDPM training

Implements various loss formulations:
- Epsilon prediction (standard DDPM)
- X0 prediction (direct denoising)
- V-prediction (velocity parameterization)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import numpy as np

from .ddpm_schedules import DDPMScheduler


class DDPMLoss(nn.Module):
    """DDPM Loss with multiple parameterizations.
    
    Supports:
    - Epsilon prediction: ||ε - ε_θ(x_t, t)||²
    - X0 prediction: ||x_0 - x_θ(x_t, t)||²
    - V prediction: ||v - v_θ(x_t, t)||²
    """
    
    def __init__(
        self,
        scheduler: DDPMScheduler,
        prediction_type: str = "epsilon",
        loss_type: str = "l2",
        weight_schedule: Optional[str] = None,
        min_snr_gamma: Optional[float] = None
    ):
        """Initialize DDPM loss.
        
        Args:
            scheduler: DDPM scheduler for noise scheduling
            prediction_type: What model predicts ("epsilon", "x0", "v")
            loss_type: Loss function type ("l2", "l1", "huber")
            weight_schedule: Weighting schedule ("uniform", "snr", "snr_trunc")
            min_snr_gamma: Minimum SNR for SNR truncation
        """
        super().__init__()
        
        self.scheduler = scheduler
        self.prediction_type = prediction_type
        self.loss_type = loss_type
        self.weight_schedule = weight_schedule
        self.min_snr_gamma = min_snr_gamma
        
        # Validate parameters
        assert prediction_type in ["epsilon", "x0", "v"], f"Unknown prediction_type: {prediction_type}"
        assert loss_type in ["l2", "l1", "huber"], f"Unknown loss_type: {loss_type}"
        
        if weight_schedule is not None:
            assert weight_schedule in ["uniform", "snr", "snr_trunc"], f"Unknown weight_schedule: {weight_schedule}"
    
    def _get_loss_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Get loss weights based on weighting schedule.
        
        Args:
            timesteps: Batch of timesteps [B]
        
        Returns:
            Loss weights [B]
        """
        if self.weight_schedule is None or self.weight_schedule == "uniform":
            return torch.ones_like(timesteps, dtype=torch.float32)
        
        # Compute SNR (Signal-to-Noise Ratio)
        alpha_bar_t = self.scheduler.alpha_bar[timesteps]
        snr = alpha_bar_t / (1 - alpha_bar_t)
        
        if self.weight_schedule == "snr":
            # Weight by SNR
            return snr
        
        elif self.weight_schedule == "snr_trunc":
            # Truncated SNR weighting (Min-SNR-γ)
            if self.min_snr_gamma is None:
                raise ValueError("min_snr_gamma must be specified for snr_trunc weighting")
            return torch.clamp(snr, max=self.min_snr_gamma)
        
        else:
            return torch.ones_like(timesteps, dtype=torch.float32)
    
    def _compute_base_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute base loss function.
        
        Args:
            pred: Predicted values
            target: Target values
        
        Returns:
            Loss values (unreduced)
        """
        if self.loss_type == "l2":
            return F.mse_loss(pred, target, reduction="none")
        elif self.loss_type == "l1":
            return F.l1_loss(pred, target, reduction="none")
        elif self.loss_type == "huber":
            return F.huber_loss(pred, target, reduction="none", delta=0.1)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    def forward(
        self,
        model_output: torch.Tensor,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        return_dict: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Compute DDPM loss.
        
        Args:
            model_output: Model predictions [B, C, H, W]
            x_start: Clean images [B, C, H, W]
            noise: Added noise [B, C, H, W]
            timesteps: Timesteps [B]
            return_dict: Whether to return detailed info
        
        Returns:
            Loss value or dictionary with detailed information
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Move scheduler to correct device
        if self.scheduler.betas.device != device:
            self.scheduler = self.scheduler.to(device)
        
        # Create noisy images x_t
        x_t = self.scheduler.add_noise(x_start, noise, timesteps)
        
        # Determine target based on prediction type
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = x_start
        elif self.prediction_type == "v":
            target = self.scheduler.get_velocity(x_start, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction_type: {self.prediction_type}")
        
        # Compute base loss
        loss_tensor = self._compute_base_loss(model_output, target)
        
        # Reduce spatial dimensions
        loss_tensor = loss_tensor.mean(dim=[1, 2, 3])  # [B]
        
        # Apply temporal weighting
        weights = self._get_loss_weights(timesteps)
        weighted_loss = loss_tensor * weights
        
        # Final loss
        loss = weighted_loss.mean()
        
        if return_dict:
            # Compute additional metrics for monitoring
            with torch.no_grad():
                # MSE for all parameterizations (for comparison)
                if self.prediction_type == "epsilon":
                    epsilon_mse = F.mse_loss(model_output, noise).item()
                    x0_pred = self.scheduler.predict_start_from_noise(x_t, timesteps, model_output)
                    x0_mse = F.mse_loss(x0_pred, x_start).item()
                elif self.prediction_type == "x0":
                    x0_mse = F.mse_loss(model_output, x_start).item()
                    epsilon_pred = self.scheduler.predict_noise_from_start(x_t, timesteps, model_output)
                    epsilon_mse = F.mse_loss(epsilon_pred, noise).item()
                else:  # v-prediction
                    v_target = self.scheduler.get_velocity(x_start, noise, timesteps)
                    v_mse = F.mse_loss(model_output, v_target).item()
                    # Convert v to epsilon and x0 for comparison
                    sqrt_alpha_bar_t = self.scheduler.sqrt_alpha_bar[timesteps]
                    sqrt_one_minus_alpha_bar_t = self.scheduler.sqrt_one_minus_alpha_bar[timesteps]
                    
                    # Reshape for broadcasting
                    sqrt_alpha_bar_t = sqrt_alpha_bar_t.view(-1, 1, 1, 1)
                    sqrt_one_minus_alpha_bar_t = sqrt_one_minus_alpha_bar_t.view(-1, 1, 1, 1)
                    
                    # v = sqrt_alpha_bar * epsilon - sqrt(1 - alpha_bar) * x0
                    # Solve for epsilon: epsilon = (v + sqrt(1 - alpha_bar) * x0) / sqrt_alpha_bar
                    epsilon_pred = (model_output + sqrt_one_minus_alpha_bar_t * x_start) / sqrt_alpha_bar_t
                    epsilon_mse = F.mse_loss(epsilon_pred, noise).item()
                    
                    # Solve for x0: x0 = (sqrt_alpha_bar * epsilon - v) / sqrt(1 - alpha_bar)
                    x0_pred = (sqrt_alpha_bar_t * noise - model_output) / sqrt_one_minus_alpha_bar_t
                    x0_mse = F.mse_loss(x0_pred, x_start).item()
                    v_mse = F.mse_loss(model_output, v_target).item()
            
            return {
                "loss": loss,
                "loss_dict": {
                    "total_loss": loss.item(),
                    "epsilon_mse": epsilon_mse,
                    "x0_mse": x0_mse,
                    "v_mse": v_mse if self.prediction_type == "v" else 0.0,
                    "mean_timestep": timesteps.float().mean().item(),
                    "mean_weight": weights.mean().item() if weights is not None else 1.0
                }
            }
        else:
            return loss


class SimpleDDPMLoss(nn.Module):
    """Simplified DDPM loss for educational purposes.
    
    Only implements epsilon prediction with L2 loss.
    This is the "simple" loss from the original DDPM paper.
    """
    
    def __init__(self, scheduler: DDPMScheduler):
        super().__init__()
        self.scheduler = scheduler
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Compute simple DDPM loss: ||ε - ε_θ(x_t, t)||²
        
        Args:
            noise_pred: Predicted noise [B, C, H, W]
            x_start: Clean images [B, C, H, W]
            noise: Ground truth noise [B, C, H, W]
            timesteps: Timesteps [B]
        
        Returns:
            Loss value
        """
        # Simple L2 loss between predicted and true noise
        return F.mse_loss(noise_pred, noise)


def compute_training_sample(
    scheduler: DDPMScheduler,
    x_start: torch.Tensor,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a training sample for DDPM.
    
    Args:
        scheduler: DDPM scheduler
        x_start: Clean images [B, C, H, W]
        device: Device to use
    
    Returns:
        x_t: Noisy images [B, C, H, W]
        noise: Added noise [B, C, H, W]
        timesteps: Random timesteps [B]
        targets: Training targets based on prediction type
    """
    batch_size = x_start.shape[0]
    
    # Sample random timesteps
    timesteps = torch.randint(
        0, scheduler.num_timesteps,
        (batch_size,), device=device, dtype=torch.long
    )
    
    # Sample noise
    noise = torch.randn_like(x_start)
    
    # Create noisy images
    x_t = scheduler.add_noise(x_start, noise, timesteps)
    
    return x_t, noise, timesteps


def test_losses():
    """Test loss functions."""
    from .ddpm_schedules import DDPMScheduler
    
    # Create scheduler and sample data
    scheduler = DDPMScheduler(num_timesteps=100)
    x_start = torch.randn(4, 3, 32, 32)
    noise = torch.randn_like(x_start)
    timesteps = torch.randint(0, 100, (4,))
    
    # Test epsilon prediction
    epsilon_pred = torch.randn_like(noise)
    
    # Test simple loss
    simple_loss = SimpleDDPMLoss(scheduler)
    loss_val = simple_loss(epsilon_pred, x_start, noise, timesteps)
    print(f"Simple loss: {loss_val.item():.6f}")
    
    # Test full DDPM loss
    ddpm_loss = DDPMLoss(scheduler, prediction_type="epsilon", loss_type="l2")
    result = ddpm_loss(epsilon_pred, x_start, noise, timesteps, return_dict=True)
    print(f"DDPM loss: {result['loss'].item():.6f}")
    print(f"Loss dict: {result['loss_dict']}")
    
    # Test different prediction types
    for pred_type in ["epsilon", "x0", "v"]:
        ddpm_loss = DDPMLoss(scheduler, prediction_type=pred_type)
        loss_val = ddpm_loss(epsilon_pred, x_start, noise, timesteps)
        print(f"{pred_type} prediction loss: {loss_val.item():.6f}")
    
    print("Loss tests passed!")


if __name__ == "__main__":
    test_losses()