"""
Loss functions for VAE training.
Includes reconstruction losses (BCE, L2, Charbonnier), KL divergence, and beta-annealing schedulers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Dict, Any


def reconstruction_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    loss_type: str = "bce",
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute reconstruction loss between target and reconstructed images.
    
    Args:
        x_hat: Reconstructed images
        x: Target images
        loss_type: Type of loss ('bce', 'l2', 'charbonnier')
        reduction: How to reduce the loss ('mean', 'sum', 'none')
    
    Returns:
        Reconstruction loss tensor
    """
    
    if loss_type == "bce":
        # Binary Cross-Entropy (for [0,1] normalized images)
        # x_hat should be logits, x should be in [0,1]
        loss = F.binary_cross_entropy_with_logits(x_hat, x, reduction='none')
        
    elif loss_type == "l2":
        # Mean Squared Error
        loss = F.mse_loss(x_hat, x, reduction='none')
        
    elif loss_type == "charbonnier":
        # Charbonnier loss (smooth L1-like loss)
        epsilon = 1e-6
        diff = x_hat - x
        loss = torch.sqrt(diff * diff + epsilon * epsilon)
        
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Reduce spatial dimensions but keep batch dimension
    loss = loss.view(loss.size(0), -1).sum(dim=1)
    
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def kl_divergence_gaussian(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Compute KL divergence between N(mu, exp(logvar)) and N(0, I).
    
    KL(q(z|x) || p(z)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    
    Args:
        mu: Mean of the latent distribution [batch_size, latent_dim]
        logvar: Log variance of the latent distribution [batch_size, latent_dim]
        reduction: How to reduce the loss ('mean', 'sum', 'none')
    
    Returns:
        KL divergence tensor
    """
    
    # KL divergence per sample (sum over latent dimensions)
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    if reduction == "mean":
        return kl_per_sample.mean()
    elif reduction == "sum":
        return kl_per_sample.sum()
    elif reduction == "none":
        return kl_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


def elbo_loss(
    x_hat: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
    recon_loss_type: str = "bce"
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Compute the Evidence Lower Bound (ELBO) loss for VAE.
    
    ELBO = E[log p(x|z)] - beta * KL(q(z|x) || p(z))
    Loss = -ELBO = recon_loss + beta * kl_loss
    
    Args:
        x_hat: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL term (beta-VAE)
        recon_loss_type: Type of reconstruction loss
    
    Returns:
        Total loss, dictionary of individual loss components
    """
    
    # Compute individual losses
    recon_loss = reconstruction_loss(x_hat, x, recon_loss_type, reduction="mean")
    kl_loss = kl_divergence_gaussian(mu, logvar, reduction="mean")
    
    # Total loss
    total_loss = recon_loss + beta * kl_loss
    
    # Loss components for logging
    loss_dict = {
        "total_loss": total_loss,
        "recon_loss": recon_loss,
        "kl_loss": kl_loss,
        "beta": torch.tensor(beta, device=total_loss.device),
        "weighted_kl": beta * kl_loss
    }
    
    return total_loss, loss_dict


class BetaScheduler:
    """
    Beta annealing scheduler for beta-VAE training.
    Gradually increases beta from 0 to target value during training.
    """
    
    def __init__(
        self,
        schedule_type: str = "linear",
        max_beta: float = 1.0,
        warmup_epochs: int = 10,
        cycle_length: Optional[int] = None,
        min_beta: float = 0.0
    ):
        """
        Initialize beta scheduler.
        
        Args:
            schedule_type: Type of schedule ('none', 'linear', 'cyclical')
            max_beta: Maximum beta value
            warmup_epochs: Number of epochs for warmup
            cycle_length: Length of cycles for cyclical annealing
            min_beta: Minimum beta value
        """
        self.schedule_type = schedule_type
        self.max_beta = max_beta
        self.min_beta = min_beta
        self.warmup_epochs = warmup_epochs
        self.cycle_length = cycle_length or (warmup_epochs * 4)
        
    def get_beta(self, epoch: int) -> float:
        """Get beta value for current epoch."""
        
        if self.schedule_type == "none":
            return self.max_beta
            
        elif self.schedule_type == "linear":
            if epoch < self.warmup_epochs:
                # Linear warmup from min_beta to max_beta
                progress = epoch / self.warmup_epochs
                return self.min_beta + (self.max_beta - self.min_beta) * progress
            else:
                return self.max_beta
                
        elif self.schedule_type == "cyclical":
            # Cyclical annealing
            cycle_progress = (epoch % self.cycle_length) / self.cycle_length
            
            if cycle_progress < 0.5:
                # First half of cycle: increase from min to max
                progress = cycle_progress * 2
                return self.min_beta + (self.max_beta - self.min_beta) * progress
            else:
                # Second half of cycle: decrease from max to min
                progress = (cycle_progress - 0.5) * 2
                return self.max_beta - (self.max_beta - self.min_beta) * progress
                
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


def compute_iwae_bound(
    model: nn.Module,
    x: torch.Tensor,
    num_samples: int = 64,
    recon_loss_type: str = "bce"
) -> torch.Tensor:
    """
    Compute the Importance Weighted Autoencoder (IWAE) bound.
    
    This provides a tighter bound on the log-likelihood than standard ELBO.
    
    Args:
        model: VAE model
        x: Input images [batch_size, channels, height, width]
        num_samples: Number of importance samples (K in IWAE paper)
        recon_loss_type: Type of reconstruction loss
    
    Returns:
        IWAE bound (higher is better)
    """
    model.eval()
    batch_size = x.size(0)
    
    with torch.no_grad():
        # Encode to get posterior parameters
        mu, logvar = model.encode(x)  # [batch_size, latent_dim]
        
        # Expand for multiple samples
        mu = mu.unsqueeze(1).expand(-1, num_samples, -1)  # [batch_size, K, latent_dim]
        logvar = logvar.unsqueeze(1).expand(-1, num_samples, -1)
        
        # Sample from posterior
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps  # [batch_size, K, latent_dim]
        
        # Reshape for decoder
        z_flat = z.reshape(-1, z.size(-1))  # [batch_size * K, latent_dim]
        x_expanded = x.unsqueeze(1).expand(-1, num_samples, -1, -1, -1)
        x_flat = x_expanded.contiguous().reshape(-1, *x.shape[1:])  # [batch_size * K, ...]
        
        # Decode
        x_hat = model.decode(z_flat)  # [batch_size * K, ...]
        
        # Compute log probabilities
        # Log p(x|z) - reconstruction likelihood
        if recon_loss_type == "bce":
            log_p_x_given_z = -F.binary_cross_entropy_with_logits(
                x_hat, x_flat, reduction='none'
            ).reshape(x_flat.size(0), -1).sum(dim=1)
        elif recon_loss_type == "l2":
            # Assume Gaussian likelihood with unit variance
            log_p_x_given_z = -0.5 * F.mse_loss(
                x_hat, x_flat, reduction='none'
            ).reshape(x_flat.size(0), -1).sum(dim=1)
        else:
            raise ValueError(f"IWAE not implemented for {recon_loss_type}")
        
        # Log p(z) - prior likelihood (standard normal)
        log_p_z = -0.5 * (z_flat.pow(2) + math.log(2 * math.pi)).sum(dim=1)
        
        # Log q(z|x) - posterior likelihood
        log_q_z_given_x = -0.5 * (
            logvar.reshape(-1, logvar.size(-1)).sum(dim=1) +
            ((z_flat - mu.reshape(-1, mu.size(-1))).pow(2) / 
             torch.exp(logvar.reshape(-1, logvar.size(-1)))).sum(dim=1) +
            z.size(-1) * math.log(2 * math.pi)
        )
        
        # Log importance weights
        log_w = log_p_x_given_z + log_p_z - log_q_z_given_x
        log_w = log_w.reshape(batch_size, num_samples)  # [batch_size, K]
        
        # IWAE bound: log(1/K * sum(exp(log_w)))
        iwae_bound = torch.logsumexp(log_w, dim=1) - math.log(num_samples)
        
    return iwae_bound.mean()


def free_bits_kl(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.5,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    KL divergence with free bits to prevent posterior collapse.
    
    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        free_bits: Minimum number of bits per latent dimension
        reduction: How to reduce the loss
    
    Returns:
        KL loss with free bits
    """
    
    # KL per latent dimension
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # [batch, latent_dim]
    
    # Apply free bits threshold
    kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
    
    # Sum over latent dimensions
    kl_per_sample = kl_per_dim.sum(dim=1)
    
    if reduction == "mean":
        return kl_per_sample.mean()
    elif reduction == "sum":
        return kl_per_sample.sum()
    elif reduction == "none":
        return kl_per_sample
    else:
        raise ValueError(f"Unknown reduction: {reduction}")


class AdaptiveBetaScheduler:
    """
    Adaptive beta scheduler that adjusts based on KL divergence magnitude.
    Helps prevent posterior collapse by maintaining a target KL level.
    """
    
    def __init__(
        self,
        target_kl: float = 3.0,
        tolerance: float = 0.5,
        update_rate: float = 0.01,
        min_beta: float = 0.0,
        max_beta: float = 2.0
    ):
        self.target_kl = target_kl
        self.tolerance = tolerance
        self.update_rate = update_rate
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.beta = 1.0
        
    def update(self, current_kl: float) -> float:
        """Update beta based on current KL divergence."""
        
        if current_kl < self.target_kl - self.tolerance:
            # KL too low, decrease beta
            self.beta = max(self.min_beta, self.beta - self.update_rate)
        elif current_kl > self.target_kl + self.tolerance:
            # KL too high, increase beta
            self.beta = min(self.max_beta, self.beta + self.update_rate)
        
        return self.beta