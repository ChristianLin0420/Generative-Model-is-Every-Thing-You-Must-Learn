"""
Evaluation metrics for Day 2: Denoising Autoencoder
"""

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import lpips
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_skimage


class MetricsCalculator:
    """Comprehensive metrics calculator for image reconstruction evaluation."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        
        # Initialize LPIPS (perceptual similarity)
        try:
            self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_available = True
        except Exception as e:
            print(f"LPIPS not available: {e}")
            self.lpips_fn = None
            self.lpips_available = False
    
    def mse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Squared Error."""
        return torch.mean((pred - target) ** 2).item()
    
    def psnr(self, pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
        """Peak Signal-to-Noise Ratio."""
        mse_val = torch.mean((pred - target) ** 2)
        if mse_val == 0:
            return float('inf')
        
        psnr_val = 20 * torch.log10(torch.tensor(max_val)) - 10 * torch.log10(mse_val)
        return psnr_val.item()
    
    def ssim(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        data_range: float = 1.0,
        window_size: int = 11,
        channel_axis: int = 1
    ) -> float:
        """Structural Similarity Index Measure."""
        # Convert to numpy and handle batches
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        
        if pred_np.ndim == 4:  # Batch of images [B, C, H, W]
            ssim_vals = []
            for i in range(pred_np.shape[0]):
                if pred_np.shape[1] == 1:  # Grayscale
                    pred_img = pred_np[i, 0]
                    target_img = target_np[i, 0]
                    ssim_val = ssim_skimage(
                        pred_img, target_img,
                        data_range=data_range,
                        win_size=window_size
                    )
                else:  # Multi-channel (RGB)
                    pred_img = np.transpose(pred_np[i], (1, 2, 0))  # CHW -> HWC
                    target_img = np.transpose(target_np[i], (1, 2, 0))
                    ssim_val = ssim_skimage(
                        pred_img, target_img,
                        data_range=data_range,
                        win_size=window_size,
                        multichannel=True,
                        channel_axis=-1
                    )
                ssim_vals.append(ssim_val)
            return np.mean(ssim_vals)
        
        elif pred_np.ndim == 3:  # Single image [C, H, W]
            if pred_np.shape[0] == 1:  # Grayscale
                return ssim_skimage(pred_np[0], target_np[0], data_range=data_range)
            else:  # Multi-channel
                pred_img = np.transpose(pred_np, (1, 2, 0))
                target_img = np.transpose(target_np, (1, 2, 0))
                return ssim_skimage(
                    pred_img, target_img,
                    data_range=data_range,
                    multichannel=True,
                    channel_axis=-1
                )
        
        else:  # 2D grayscale
            return ssim_skimage(pred_np, target_np, data_range=data_range)
    
    def lpips(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Learned Perceptual Image Patch Similarity."""
        if not self.lpips_available:
            return 0.0
        
        # Ensure tensors are in [-1, 1] range for LPIPS
        pred_norm = 2 * pred - 1
        target_norm = 2 * target - 1
        
        # Handle grayscale by replicating to 3 channels
        if pred.size(1) == 1:
            pred_norm = pred_norm.repeat(1, 3, 1, 1)
            target_norm = target_norm.repeat(1, 3, 1, 1)
        
        with torch.no_grad():
            lpips_score = self.lpips_fn(pred_norm, target_norm)
        
        return lpips_score.mean().item()
    
    def mae(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Mean Absolute Error."""
        return torch.mean(torch.abs(pred - target)).item()
    
    def compute_all_metrics(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        max_val: float = 1.0,
        compute_lpips: bool = False
    ) -> Dict[str, float]:
        """Compute all available metrics."""
        metrics = {
            'mse': self.mse(pred, target),
            'mae': self.mae(pred, target),
            'psnr': self.psnr(pred, target, max_val),
            'ssim': self.ssim(pred, target, data_range=max_val)
        }
        
        if compute_lpips and self.lpips_available:
            metrics['lpips'] = self.lpips(pred, target)
        
        return metrics


class EpochMetrics:
    """Track metrics across an epoch."""
    
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {name: [] for name in self.metric_names}
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for name, value in kwargs.items():
            if name in self.metrics:
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.metrics[name].append(value)
    
    def get_averages(self) -> Dict[str, float]:
        """Get average values for all metrics."""
        averages = {}
        for name, values in self.metrics.items():
            if values:
                averages[name] = np.mean(values)
            else:
                averages[name] = 0.0
        return averages
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get comprehensive summary statistics."""
        summary = {}
        for name, values in self.metrics.items():
            if values:
                values = np.array(values)
                summary[name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
            else:
                summary[name] = {
                    'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0
                }
        return summary


class MetricsLogger:
    """Log and save metrics to files."""
    
    def __init__(self, log_dir: Union[str, Path]):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_metrics = []
        self.val_metrics = []
    
    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float] = None
    ):
        """Log metrics for an epoch."""
        train_entry = {'epoch': epoch, **train_metrics}
        self.train_metrics.append(train_entry)
        
        if val_metrics:
            val_entry = {'epoch': epoch, **val_metrics}
            self.val_metrics.append(val_entry)
    
    def save_csv(self):
        """Save metrics to CSV files."""
        # Training metrics
        if self.train_metrics:
            train_csv = self.log_dir / 'train_metrics.csv'
            with open(train_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.train_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.train_metrics)
        
        # Validation metrics
        if self.val_metrics:
            val_csv = self.log_dir / 'val_metrics.csv'
            with open(val_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.val_metrics[0].keys())
                writer.writeheader()
                writer.writerows(self.val_metrics)
    
    def get_best_epoch(self, metric: str = 'psnr', mode: str = 'max') -> Dict:
        """Get best epoch based on validation metric."""
        if not self.val_metrics:
            return {}
        
        if mode == 'max':
            best = max(self.val_metrics, key=lambda x: x.get(metric, -float('inf')))
        else:
            best = min(self.val_metrics, key=lambda x: x.get(metric, float('inf')))
        
        return best


def compute_reconstruction_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int = None,
    compute_lpips: bool = False
) -> Dict[str, float]:
    """
    Compute reconstruction metrics over entire dataset.
    
    Args:
        model: Trained model
        dataloader: Data loader with (clean, noisy, sigma) tuples
        device: Computation device
        max_batches: Limit number of batches for faster evaluation
        compute_lpips: Whether to compute LPIPS (expensive)
    
    Returns:
        Dictionary with averaged metrics
    """
    model.eval()
    metrics_calc = MetricsCalculator(device)
    
    all_metrics = {
        'mse': [], 'mae': [], 'psnr': [], 'ssim': []
    }
    if compute_lpips:
        all_metrics['lpips'] = []
    
    with torch.no_grad():
        for batch_idx, (clean, noisy, _) in enumerate(dataloader):
            if max_batches and batch_idx >= max_batches:
                break
            
            clean = clean.to(device)
            noisy = noisy.to(device)
            
            # Forward pass
            recon = model(noisy)
            
            # Compute metrics
            batch_metrics = metrics_calc.compute_all_metrics(
                recon, clean, max_val=1.0, compute_lpips=compute_lpips
            )
            
            # Accumulate
            for key, value in batch_metrics.items():
                all_metrics[key].append(value)
    
    # Average all metrics
    avg_metrics = {}
    for key, values in all_metrics.items():
        if values:
            avg_metrics[key] = np.mean(values)
        else:
            avg_metrics[key] = 0.0
    
    return avg_metrics


def compute_sigma_wise_metrics(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    test_sigmas: List[float],
    max_samples_per_sigma: int = 100
) -> Dict[float, Dict[str, float]]:
    """
    Compute metrics for each noise level separately.
    
    Returns:
        Dictionary mapping sigma -> metrics dict
    """
    model.eval()
    metrics_calc = MetricsCalculator(device)
    
    # Group data by sigma
    sigma_data = {sigma: {'clean': [], 'noisy': [], 'recon': []} for sigma in test_sigmas}
    
    with torch.no_grad():
        for clean, noisy, sigmas in dataloader:
            clean = clean.to(device)
            noisy = noisy.to(device)
            recon = model(noisy)
            
            # Group by sigma value
            for i, sigma in enumerate(sigmas):
                sigma_val = sigma.item()
                if sigma_val in sigma_data:
                    if len(sigma_data[sigma_val]['clean']) < max_samples_per_sigma:
                        sigma_data[sigma_val]['clean'].append(clean[i:i+1])
                        sigma_data[sigma_val]['recon'].append(recon[i:i+1])
    
    # Compute metrics for each sigma
    sigma_metrics = {}
    for sigma, data in sigma_data.items():
        if data['clean']:
            clean_batch = torch.cat(data['clean'], dim=0)
            recon_batch = torch.cat(data['recon'], dim=0)
            
            metrics = metrics_calc.compute_all_metrics(recon_batch, clean_batch)
            sigma_metrics[sigma] = metrics
    
    return sigma_metrics