"""
Evaluation metrics for Day 1: Gaussian Noise Basics
"""

import math
from typing import Optional, Union
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_skimage


def mse(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Mean Squared Error between two tensors.
    
    Args:
        x: First tensor
        y: Second tensor
    
    Returns:
        MSE value
    """
    return torch.mean((x - y) ** 2).item()


def psnr(
    x: torch.Tensor, 
    y: torch.Tensor, 
    max_val: float = 1.0
) -> float:
    """
    Compute Peak Signal-to-Noise Ratio.
    
    Args:
        x: First tensor (ground truth)
        y: Second tensor (noisy/reconstructed)
        max_val: Maximum possible pixel value
    
    Returns:
        PSNR in dB
    """
    mse_val = torch.mean((x - y) ** 2)
    if mse_val == 0:
        return float('inf')
    
    psnr_val = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse_val)
    return psnr_val.item()


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    channel_axis: int = 1  # For tensors: BCHW format
) -> float:
    """
    Compute Structural Similarity Index Measure using scikit-image.
    
    Args:
        x: First tensor
        y: Second tensor
        data_range: Range of the data (max - min)
        channel_axis: Channel axis (1 for BCHW format)
    
    Returns:
        SSIM value
    """
    # Convert to numpy and handle batches
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    if x_np.ndim == 4:  # Batch of images
        ssim_vals = []
        for i in range(x_np.shape[0]):
            # For multi-channel images, we need to transpose to HWC
            if x_np.shape[1] == 1:  # Grayscale
                x_img = x_np[i, 0]  # Remove channel dim for grayscale
                y_img = y_np[i, 0]
                ssim_val = ssim_skimage(x_img, y_img, data_range=data_range)
            else:  # Multi-channel
                x_img = np.transpose(x_np[i], (1, 2, 0))  # CHW -> HWC
                y_img = np.transpose(y_np[i], (1, 2, 0))
                ssim_val = ssim_skimage(
                    x_img, y_img, 
                    data_range=data_range, 
                    multichannel=True,
                    channel_axis=-1
                )
            ssim_vals.append(ssim_val)
        return np.mean(ssim_vals)
    
    elif x_np.ndim == 3:  # Single multi-channel image
        if x_np.shape[0] == 1:  # Grayscale
            return ssim_skimage(x_np[0], y_np[0], data_range=data_range)
        else:  # Multi-channel
            x_img = np.transpose(x_np, (1, 2, 0))  # CHW -> HWC
            y_img = np.transpose(y_np, (1, 2, 0))
            return ssim_skimage(
                x_img, y_img,
                data_range=data_range,
                multichannel=True,
                channel_axis=-1
            )
    
    elif x_np.ndim == 2:  # Single grayscale image
        return ssim_skimage(x_np, y_np, data_range=data_range)
    
    else:
        raise ValueError(f"Unsupported tensor shape: {x.shape}")


def batch_metrics(
    original: torch.Tensor,
    noisy: torch.Tensor,
    max_val: float = 1.0
) -> dict:
    """
    Compute all metrics for a batch of images.
    
    Args:
        original: Original images [B, C, H, W]
        noisy: Noisy images [B, C, H, W]
        max_val: Maximum pixel value
    
    Returns:
        Dictionary with metric values
    """
    return {
        'mse': mse(original, noisy),
        'psnr': psnr(original, noisy, max_val),
        'ssim': ssim(original, noisy, data_range=max_val)
    }


def noise_degradation_metrics(
    original: torch.Tensor,
    sigma: float,
    noisy: Optional[torch.Tensor] = None,
    data_std: float = 1.0,
    max_val: float = 1.0
) -> dict:
    """
    Compute comprehensive metrics for noise degradation analysis.
    
    Args:
        original: Original clean images
        sigma: Noise standard deviation used
        noisy: Noisy images (if None, will be computed)
        data_std: Standard deviation of original data
        max_val: Maximum pixel value
    
    Returns:
        Dictionary with all metrics
    """
    if noisy is None:
        from .noise import add_gaussian_noise
        generator = torch.Generator(device=original.device).manual_seed(42)
        noisy = add_gaussian_noise(original, sigma, generator=generator)
    
    # Basic metrics
    metrics = batch_metrics(original, noisy, max_val)
    
    # Add sigma and theoretical SNR
    metrics['sigma'] = sigma
    metrics['snr_db'] = compute_snr_db(sigma, data_std)
    
    # Add empirical noise analysis
    noise = noisy - original
    metrics['noise_std_empirical'] = torch.std(noise).item()
    metrics['noise_mean'] = torch.mean(noise).item()
    metrics['signal_std'] = torch.std(original).item()
    
    # Empirical SNR
    if torch.std(noise).item() > 0:
        metrics['snr_empirical_db'] = 20 * torch.log10(
            torch.std(original) / torch.std(noise)
        ).item()
    else:
        metrics['snr_empirical_db'] = float('inf')
    
    return metrics


def compute_snr_db(sigma: float, data_std: float = 1.0) -> float:
    """
    Compute theoretical Signal-to-Noise Ratio in dB.
    
    Args:
        sigma: Noise standard deviation
        data_std: Data standard deviation
    
    Returns:
        SNR in dB
    """
    if sigma == 0.0:
        return float('inf')
    
    snr_linear = (data_std ** 2) / (sigma ** 2)
    return 10 * math.log10(snr_linear)


class MetricsTracker:
    """Class to track metrics across multiple evaluations."""
    
    def __init__(self):
        self.metrics_history = []
    
    def add_measurement(
        self,
        original: torch.Tensor,
        noisy: torch.Tensor,
        sigma: float,
        max_val: float = 1.0,
        data_std: float = 1.0
    ) -> dict:
        """Add a new measurement."""
        metrics = noise_degradation_metrics(original, sigma, noisy, data_std, max_val)
        self.metrics_history.append(metrics)
        return metrics
    
    def get_summary(self) -> dict:
        """Get summary statistics across all measurements."""
        if not self.metrics_history:
            return {}
        
        import pandas as pd
        df = pd.DataFrame(self.metrics_history)
        
        summary = {}
        for col in df.columns:
            if col != 'sigma':  # Don't summarize sigma values
                try:
                    summary[f'{col}_mean'] = df[col].mean()
                    summary[f'{col}_std'] = df[col].std()
                    summary[f'{col}_min'] = df[col].min()
                    summary[f'{col}_max'] = df[col].max()
                except:
                    pass  # Skip non-numeric columns
        
        return summary
    
    def to_dataframe(self):
        """Convert metrics history to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.metrics_history)
    
    def save_csv(self, path: Union[str, Path]) -> None:
        """Save metrics to CSV file."""
        import pandas as pd
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(path, index=False)
        print(f"Saved metrics to {path}")
    
    def clear(self) -> None:
        """Clear metrics history."""
        self.metrics_history.clear()