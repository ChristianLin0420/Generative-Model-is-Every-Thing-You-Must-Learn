"""
Core noise functionality for Day 1: Gaussian Noise Basics
"""

import math
from typing import List, Optional, Union

import numpy as np
import torch


def add_gaussian_noise(
    x: torch.Tensor,
    sigma: float,
    clip_range: Optional[tuple] = None,
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
    
    # Generate noise with same shape as input
    noise = torch.randn_like(x, generator=generator) * sigma
    noisy_x = x + noise
    
    # Apply clipping if specified
    if clip_range is not None:
        noisy_x = torch.clamp(noisy_x, clip_range[0], clip_range[1])
    
    return noisy_x


def sigma_schedule(
    kind: str,
    num_levels: int,
    min_sigma: float = 0.0,
    max_sigma: float = 1.0
) -> List[float]:
    """
    Generate noise schedule (sequence of sigma values).
    
    Args:
        kind: Type of schedule ('linear', 'cosine', 'custom')
        num_levels: Number of noise levels
        min_sigma: Minimum sigma value
        max_sigma: Maximum sigma value
    
    Returns:
        List of sigma values
    """
    if kind == "linear":
        return list(np.linspace(min_sigma, max_sigma, num_levels))
    
    elif kind == "cosine":
        # Cosine schedule from 0 to pi/2
        t = np.linspace(0, math.pi/2, num_levels)
        cosine_vals = np.cos(t)
        # Invert and scale to [min_sigma, max_sigma]
        normalized = (1 - cosine_vals)  # Now goes from 0 to 1
        return list(min_sigma + normalized * (max_sigma - min_sigma))
    
    elif kind == "custom":
        # Custom schedule with more noise at the end
        t = np.linspace(0, 1, num_levels)
        custom_vals = t ** 2  # Quadratic progression
        return list(min_sigma + custom_vals * (max_sigma - min_sigma))
    
    else:
        raise ValueError(f"Unknown schedule kind: {kind}")


def compute_snr(sigma: float, data_std: float = 1.0) -> float:
    """
    Compute Signal-to-Noise Ratio.
    
    Args:
        sigma: Noise standard deviation
        data_std: Data standard deviation
    
    Returns:
        SNR in dB
    """
    if sigma == 0.0:
        return float('inf')
    
    snr_linear = (data_std ** 2) / (sigma ** 2)
    snr_db = 10 * math.log10(snr_linear)
    return snr_db


def batch_add_noise(
    batch: torch.Tensor,
    sigmas: List[float],
    clip_range: Optional[tuple] = None,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Add different noise levels to each image in batch.
    
    Args:
        batch: Input batch [B, C, H, W]
        sigmas: List of sigma values (length should match batch size)
        clip_range: Optional clipping range
        generator: Optional random generator
    
    Returns:
        Batch with different noise levels applied
    """
    if len(sigmas) != batch.size(0):
        raise ValueError(f"Number of sigmas ({len(sigmas)}) must match batch size ({batch.size(0)})")
    
    noisy_batch = []
    for i, sigma in enumerate(sigmas):
        noisy_img = add_gaussian_noise(batch[i:i+1], sigma, clip_range, generator)
        noisy_batch.append(noisy_img)
    
    return torch.cat(noisy_batch, dim=0)


def progressive_noise_sequence(
    image: torch.Tensor,
    sigmas: List[float],
    clip_range: Optional[tuple] = None,
    generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """
    Create sequence of progressively noisier versions of same image.
    
    Args:
        image: Single image tensor [C, H, W]
        sigmas: List of noise levels
        clip_range: Optional clipping range
        generator: Optional random generator
    
    Returns:
        Tensor of shape [len(sigmas), C, H, W] with progressive noise
    """
    sequence = []
    for sigma in sigmas:
        noisy_img = add_gaussian_noise(image.unsqueeze(0), sigma, clip_range, generator)
        sequence.append(noisy_img)
    
    return torch.cat(sequence, dim=0)


def analyze_noise_impact(
    original: torch.Tensor,
    noisy: torch.Tensor
) -> dict:
    """
    Analyze the impact of noise on the image.
    
    Args:
        original: Original image tensor
        noisy: Noisy image tensor
    
    Returns:
        Dictionary with analysis results
    """
    diff = noisy - original
    
    results = {
        'mse': torch.mean((noisy - original) ** 2).item(),
        'mae': torch.mean(torch.abs(noisy - original)).item(),
        'noise_std': torch.std(diff).item(),
        'noise_mean': torch.mean(diff).item(),
        'original_std': torch.std(original).item(),
        'noisy_std': torch.std(noisy).item(),
        'snr_empirical': 20 * torch.log10(torch.std(original) / torch.std(diff)).item()
    }
    
    return results


class NoiseScheduler:
    """Class to manage different noise schedules."""
    
    def __init__(
        self,
        schedule_type: str = "linear",
        num_levels: int = 10,
        min_sigma: float = 0.0,
        max_sigma: float = 1.0
    ):
        self.schedule_type = schedule_type
        self.num_levels = num_levels
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        
        self.sigmas = sigma_schedule(schedule_type, num_levels, min_sigma, max_sigma)
    
    def __len__(self) -> int:
        return len(self.sigmas)
    
    def __getitem__(self, idx: int) -> float:
        return self.sigmas[idx]
    
    def get_sigmas(self) -> List[float]:
        return self.sigmas.copy()
    
    def get_snrs(self, data_std: float = 1.0) -> List[float]:
        return [compute_snr(sigma, data_std) for sigma in self.sigmas]