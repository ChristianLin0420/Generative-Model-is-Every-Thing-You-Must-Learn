"""
DDPM loss functions with support for epsilon-prediction, x0-prediction, and v-parameterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DDPMLoss(nn.Module):
    """
    DDPM loss for epsilon-prediction objective.
    
    L_simple = E[||ε - ε_θ(x_t, t)||²]
    
    Where:
    - ε is the noise added to x_0
    - ε_θ is the predicted noise from the model
    - x_t is the noisy image at timestep t
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute DDPM loss.
        
        Args:
            noise_pred: Predicted noise from model [B, C, H, W]
            noise_true: True noise added to x_0 [B, C, H, W]
            mask: Optional mask for loss weighting [B, C, H, W] or [B, 1, 1, 1]
            
        Returns:
            Loss scalar
        """
        # MSE loss between predicted and true noise
        loss = F.mse_loss(noise_pred, noise_true, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
        
        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class WeightedDDPMLoss(nn.Module):
    """
    DDPM loss with SNR weighting as in "Perception Prioritized Training of Diffusion Models".
    
    The loss is weighted by SNR(t) to balance learning across timesteps.
    """
    
    def __init__(self, weighting: str = 'snr', reduction: str = 'mean'):
        super().__init__()
        self.weighting = weighting
        self.reduction = reduction
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_true: torch.Tensor,
        alpha_bars: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute weighted DDPM loss.
        
        Args:
            noise_pred: Predicted noise [B, C, H, W]
            noise_true: True noise [B, C, H, W]
            alpha_bars: Alpha bar values for batch timesteps [B]
            mask: Optional mask [B, C, H, W] or [B, 1, 1, 1]
            
        Returns:
            Weighted loss scalar
        """
        # Base MSE loss
        loss = F.mse_loss(noise_pred, noise_true, reduction='none')
        
        # Compute weights based on alpha_bars
        if self.weighting == 'snr':
            # SNR weighting: w(t) = SNR(t) = ᾱ_t / (1 - ᾱ_t)
            snr = alpha_bars / (1 - alpha_bars)
            weights = snr.view(-1, 1, 1, 1)  # [B, 1, 1, 1]
        elif self.weighting == 'snr_sqrt':
            # Square root SNR weighting
            snr = alpha_bars / (1 - alpha_bars)
            weights = torch.sqrt(snr).view(-1, 1, 1, 1)
        elif self.weighting == 'none':
            weights = 1.0
        else:
            raise ValueError(f"Unknown weighting: {self.weighting}")
        
        # Apply weighting
        loss = loss * weights
        
        # Apply mask if provided
        if mask is not None:
            loss = loss * mask
        
        # Reduce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class VLBLoss(nn.Module):
    """
    Variational Lower Bound (VLB) loss for DDPM.
    
    This is the full VLB objective, more expensive but theoretically grounded.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_true: torch.Tensor,
        alpha_bars: torch.Tensor,
        betas: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute VLB loss.
        
        Args:
            noise_pred: Predicted noise [B, C, H, W]
            noise_true: True noise [B, C, H, W]
            alpha_bars: Alpha bar values [B]
            betas: Beta values [B]
            
        Returns:
            VLB loss scalar
        """
        # This is a simplified version - full VLB requires more careful implementation
        # For now, use weighted MSE as approximation
        loss = F.mse_loss(noise_pred, noise_true, reduction='none')
        
        # Weight by 1/2β_t as in DDPM paper
        weights = 1.0 / (2.0 * betas.view(-1, 1, 1, 1))
        loss = loss * weights
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


def compute_forward_process(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    alpha_bars: torch.Tensor,
    noise: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute forward diffusion process: q(x_t | x_0).
    
    x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
    
    Args:
        x0: Clean images [B, C, H, W]
        timesteps: Timestep indices [B]
        alpha_bars: Alpha bar schedule [T]
        noise: Optional pre-sampled noise [B, C, H, W]
        
    Returns:
        Tuple of (noisy_images, noise)
    """
    if noise is None:
        noise = torch.randn_like(x0)
    
    # Get alpha_bar values for the batch timesteps
    alpha_bar_t = alpha_bars[timesteps].view(-1, 1, 1, 1)  # [B, 1, 1, 1]
    
    # Compute noisy images
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
    
    noisy_images = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    
    return noisy_images, noise


def get_loss_fn(loss_type: str = 'ddpm', **kwargs) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: Loss function type
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function
    """
    if loss_type == 'ddpm':
        return DDPMLoss(**kwargs)
    elif loss_type == 'weighted_ddpm':
        return WeightedDDPMLoss(**kwargs)
    elif loss_type == 'vlb':
        return VLBLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def test_losses():
    """Test loss functions with dummy data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    batch_size = 4
    channels = 1
    height = 28
    width = 28
    
    noise_pred = torch.randn(batch_size, channels, height, width, device=device)
    noise_true = torch.randn(batch_size, channels, height, width, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    
    # Create dummy schedules
    T = 1000
    betas = torch.linspace(1e-4, 0.02, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    
    # Test DDPM loss
    ddpm_loss = DDPMLoss()
    loss1 = ddpm_loss(noise_pred, noise_true)
    print(f"DDPM Loss: {loss1.item():.4f}")
    
    # Test weighted DDPM loss
    weighted_loss = WeightedDDPMLoss(weighting='snr')
    batch_alpha_bars = alpha_bars[timesteps]
    loss2 = weighted_loss(noise_pred, noise_true, batch_alpha_bars)
    print(f"Weighted DDPM Loss: {loss2.item():.4f}")
    
    # Test VLB loss
    vlb_loss = VLBLoss()
    batch_betas = betas[timesteps]
    loss3 = vlb_loss(noise_pred, noise_true, batch_alpha_bars, batch_betas)
    print(f"VLB Loss: {loss3.item():.4f}")
    
    # Test forward process
    x0 = torch.randn(batch_size, channels, height, width, device=device)
    noisy_images, noise = compute_forward_process(x0, timesteps, alpha_bars)
    print(f"Forward process - Input range: [{x0.min():.3f}, {x0.max():.3f}]")
    print(f"Forward process - Noisy range: [{noisy_images.min():.3f}, {noisy_images.max():.3f}]")
    
    print("Loss tests completed!")


if __name__ == "__main__":
    test_losses()
