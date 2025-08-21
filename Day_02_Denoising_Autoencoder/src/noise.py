"""
Noise generation utilities for Day 2: Denoising Autoencoder
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch


def add_gaussian_noise(
    x: torch.Tensor,
    sigma: float,
    clip_range: Optional[Tuple[float, float]] = None,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Add Gaussian noise to input tensor.
    
    Args:
        x: Input tensor
        sigma: Standard deviation of noise
        clip_range: Optional clipping range (min, max)
        generator: Optional random generator for reproducibility
    
    Returns:
        Noisy tensor
    """
    if sigma == 0.0:
        return x.clone()
    
    # Generate noise with same shape and device as input
    if generator is not None:
        # Use generator with explicit shape and device
        noise = torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype) * sigma
    else:
        noise = torch.randn_like(x) * sigma
    
    noisy_x = x + noise
    
    # Apply clipping if specified
    if clip_range is not None:
        noisy_x = torch.clamp(noisy_x, clip_range[0], clip_range[1])
    
    return noisy_x


def sigma_schedule(
    kind: str,
    num_levels: int,
    min_sigma: float = 0.01,
    max_sigma: float = 1.0
) -> List[float]:
    """
    Generate noise schedule (sequence of sigma values).
    
    Args:
        kind: Type of schedule ('linear', 'cosine')
        num_levels: Number of noise levels
        min_sigma: Minimum sigma value
        max_sigma: Maximum sigma value
    
    Returns:
        List of sigma values
    """
    if kind == "linear":
        return list(np.linspace(min_sigma, max_sigma, num_levels))
    
    elif kind == "cosine":
        # Cosine schedule - slower progression at start
        t = np.linspace(0, math.pi/2, num_levels)
        cosine_vals = np.cos(t)
        # Invert and scale to [min_sigma, max_sigma]
        normalized = (1 - cosine_vals)
        return list(min_sigma + normalized * (max_sigma - min_sigma))
    
    else:
        raise ValueError(f"Unknown schedule kind: {kind}")


def get_train_test_mismatch_sigmas(
    train_range: Tuple[float, float] = (0.1, 0.5),
    test_range: Tuple[float, float] = (0.1, 1.0),
    num_train: int = 4,
    num_test: int = 6
) -> Tuple[List[float], List[float]]:
    """
    Generate train/test sigma lists with deliberate mismatch for robustness testing.
    
    Args:
        train_range: (min, max) sigma values for training
        test_range: (min, max) sigma values for testing (can exceed training range)
        num_train: Number of training sigma levels
        num_test: Number of test sigma levels
    
    Returns:
        (train_sigmas, test_sigmas)
    """
    train_sigmas = list(np.linspace(train_range[0], train_range[1], num_train))
    test_sigmas = list(np.linspace(test_range[0], test_range[1], num_test))
    
    return train_sigmas, test_sigmas


def batch_add_different_noise(
    batch: torch.Tensor,
    sigmas: Union[List[float], torch.Tensor],
    clip_range: Optional[Tuple[float, float]] = None,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Add different noise levels to each sample in batch.
    
    Args:
        batch: Input batch [B, C, H, W]
        sigmas: Noise levels for each sample [B] or list of length B
        clip_range: Optional clipping range
        generator: Optional random generator
    
    Returns:
        Batch with different noise levels applied
    """
    batch_size = batch.size(0)
    
    if isinstance(sigmas, list):
        sigmas = torch.tensor(sigmas, device=batch.device)
    
    if len(sigmas) != batch_size:
        # Cycle through sigmas if not enough provided
        sigmas = sigmas.repeat((batch_size + len(sigmas) - 1) // len(sigmas))[:batch_size]
    
    noisy_batch = torch.zeros_like(batch)
    
    for i in range(batch_size):
        noisy_batch[i] = add_gaussian_noise(
            batch[i:i+1], 
            sigmas[i].item(),
            clip_range,
            generator
        ).squeeze(0)
    
    return noisy_batch


def compute_noise_statistics(
    clean: torch.Tensor,
    noisy: torch.Tensor
) -> dict:
    """
    Compute statistics about applied noise.
    
    Args:
        clean: Clean images
        noisy: Noisy images
    
    Returns:
        Dictionary with noise statistics
    """
    noise = noisy - clean
    
    stats = {
        'noise_mean': noise.mean().item(),
        'noise_std': noise.std().item(),
        'noise_min': noise.min().item(),
        'noise_max': noise.max().item(),
        'signal_mean': clean.mean().item(),
        'signal_std': clean.std().item(),
        'empirical_snr_db': 20 * torch.log10(clean.std() / noise.std()).item() if noise.std() > 0 else float('inf')
    }
    
    return stats


class AdaptiveNoiseScheduler:
    """Adaptive noise scheduler that adjusts based on training progress."""
    
    def __init__(
        self,
        initial_sigmas: List[float],
        adaptation_type: str = "curriculum",  # curriculum | random | fixed
        curriculum_epochs: int = 10
    ):
        self.initial_sigmas = initial_sigmas
        self.current_sigmas = initial_sigmas.copy()
        self.adaptation_type = adaptation_type
        self.curriculum_epochs = curriculum_epochs
        self.epoch = 0
    
    def step(self, epoch: int) -> List[float]:
        """Update noise schedule based on training progress."""
        self.epoch = epoch
        
        if self.adaptation_type == "curriculum":
            # Start with lower noise, gradually increase
            progress = min(epoch / self.curriculum_epochs, 1.0)
            max_sigma = max(self.initial_sigmas)
            self.current_sigmas = [
                sigma * (0.5 + 0.5 * progress) for sigma in self.initial_sigmas
            ]
            
        elif self.adaptation_type == "random":
            # Randomly permute sigma order each epoch
            self.current_sigmas = self.initial_sigmas.copy()
            np.random.shuffle(self.current_sigmas)
            
        # "fixed" keeps initial sigmas unchanged
        
        return self.current_sigmas
    
    def get_current_sigmas(self) -> List[float]:
        """Get current sigma values."""
        return self.current_sigmas


def create_sigma_curriculum(
    easy_range: Tuple[float, float] = (0.05, 0.2),
    hard_range: Tuple[float, float] = (0.3, 0.8),
    num_levels: int = 4,
    curriculum_schedule: str = "linear"
) -> dict:
    """
    Create curriculum learning schedule for noise levels.
    
    Args:
        easy_range: Initial easy noise range
        hard_range: Final hard noise range  
        num_levels: Number of discrete noise levels
        curriculum_schedule: How to progress from easy to hard
    
    Returns:
        Dictionary with curriculum configuration
    """
    curriculum = {
        'easy_sigmas': list(np.linspace(easy_range[0], easy_range[1], num_levels)),
        'hard_sigmas': list(np.linspace(hard_range[0], hard_range[1], num_levels)),
        'schedule': curriculum_schedule,
        'num_levels': num_levels
    }
    
    return curriculum