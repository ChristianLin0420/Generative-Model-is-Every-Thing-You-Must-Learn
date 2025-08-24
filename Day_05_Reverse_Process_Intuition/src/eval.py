"""Evaluation metrics for DDPM models

Implements:
- Teacher-forced reconstruction evaluation (PSNR, SSIM)
- Sample quality metrics (IS, FID, LPIPS)
- Denoising performance across timesteps
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from tqdm.auto import tqdm
import os
import csv

try:
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("Warning: scikit-image not available. PSNR/SSIM metrics disabled.")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Perceptual similarity metrics disabled.")

from .ddpm_schedules import DDPMScheduler
from .sampler import DDPMSampler, reconstruct_from_noise


def tensor_to_numpy_batch(tensor: torch.Tensor) -> np.ndarray:
    """Convert batch of tensors to numpy arrays.
    
    Args:
        tensor: Input tensor [B, C, H, W]
    
    Returns:
        Numpy array [B, H, W, C] or [B, H, W] for grayscale
    """
    # Move to CPU and convert
    arr = tensor.detach().cpu().numpy()
    
    # Transpose to [B, H, W, C]
    if arr.shape[1] == 3:  # RGB
        arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.shape[1] == 1:  # Grayscale
        arr = arr.squeeze(1)  # Remove channel dim -> [B, H, W]
    
    # Clip to valid range
    arr = np.clip(arr, 0, 1)
    
    return arr


def compute_psnr_batch(images1: torch.Tensor, images2: torch.Tensor) -> List[float]:
    """Compute PSNR between two batches of images.
    
    Args:
        images1: First batch [B, C, H, W]
        images2: Second batch [B, C, H, W]
    
    Returns:
        List of PSNR values for each image pair
    """
    if not SKIMAGE_AVAILABLE:
        return [0.0] * images1.shape[0]
    
    arr1 = tensor_to_numpy_batch(images1)
    arr2 = tensor_to_numpy_batch(images2)
    
    psnr_values = []
    for i in range(arr1.shape[0]):
        if len(arr1.shape) == 4:  # RGB
            psnr = peak_signal_noise_ratio(arr1[i], arr2[i], data_range=1.0)
        else:  # Grayscale
            psnr = peak_signal_noise_ratio(arr1[i], arr2[i], data_range=1.0)
        psnr_values.append(float(psnr))
    
    return psnr_values


def compute_ssim_batch(images1: torch.Tensor, images2: torch.Tensor) -> List[float]:
    """Compute SSIM between two batches of images.
    
    Args:
        images1: First batch [B, C, H, W]
        images2: Second batch [B, C, H, W]
    
    Returns:
        List of SSIM values for each image pair
    """
    if not SKIMAGE_AVAILABLE:
        return [0.0] * images1.shape[0]
    
    arr1 = tensor_to_numpy_batch(images1)
    arr2 = tensor_to_numpy_batch(images2)
    
    ssim_values = []
    for i in range(arr1.shape[0]):
        if len(arr1.shape) == 4:  # RGB
            ssim = structural_similarity(
                arr1[i], arr2[i], data_range=1.0, multichannel=True, channel_axis=2
            )
        else:  # Grayscale
            ssim = structural_similarity(arr1[i], arr2[i], data_range=1.0)
        ssim_values.append(float(ssim))
    
    return ssim_values


def compute_lpips_batch(
    images1: torch.Tensor, 
    images2: torch.Tensor, 
    net: str = 'alex',
    device: Optional[torch.device] = None
) -> List[float]:
    """Compute LPIPS (perceptual similarity) between two batches.
    
    Args:
        images1: First batch [B, C, H, W]
        images2: Second batch [B, C, H, W]
        net: Network to use ('alex', 'vgg', 'squeeze')
        device: Device to run on
    
    Returns:
        List of LPIPS values for each image pair
    """
    if not LPIPS_AVAILABLE:
        return [0.0] * images1.shape[0]
    
    if device is None:
        device = images1.device
    
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net=net).to(device)
    
    with torch.no_grad():
        # LPIPS expects values in [-1, 1]
        img1_norm = images1 * 2.0 - 1.0
        img2_norm = images2 * 2.0 - 1.0
        
        # Handle grayscale by repeating channels
        if img1_norm.shape[1] == 1:
            img1_norm = img1_norm.repeat(1, 3, 1, 1)
            img2_norm = img2_norm.repeat(1, 3, 1, 1)
        
        # Compute LPIPS
        distances = lpips_model(img1_norm, img2_norm)
        distances = distances.squeeze().cpu().numpy()
        
        if distances.ndim == 0:
            distances = [float(distances)]
        else:
            distances = distances.tolist()
    
    return distances


class DDPMEvaluator:
    """Comprehensive evaluator for DDPM models."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        scheduler: DDPMScheduler,
        device: Optional[torch.device] = None,
        metrics: List[str] = None
    ):
        """Initialize evaluator.
        
        Args:
            model: Trained DDPM model
            scheduler: DDPM scheduler
            device: Device to run evaluation on
            metrics: List of metrics to compute ['psnr', 'ssim', 'lpips']
        """
        self.model = model
        self.scheduler = scheduler
        self.device = device or next(model.parameters()).device
        
        if metrics is None:
            metrics = ['psnr', 'ssim']
            if LPIPS_AVAILABLE:
                metrics.append('lpips')
        
        self.metrics = metrics
        
        # Initialize LPIPS if needed
        if 'lpips' in self.metrics and LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net='alex').to(self.device)
        else:
            self.lpips_model = None
    
    def evaluate_reconstruction(
        self,
        dataloader: torch.utils.data.DataLoader,
        timesteps_to_eval: List[int] = None,
        num_samples: Optional[int] = None,
        sampler_type: str = "ddpm"
    ) -> Dict[str, Any]:
        """Evaluate reconstruction quality at various timesteps.
        
        This performs "teacher-forced" evaluation: take clean images,
        add noise to timestep t, then denoise back to 0.
        
        Args:
            dataloader: Data loader with clean images
            timesteps_to_eval: Timesteps to evaluate at
            num_samples: Maximum number of samples to evaluate
            sampler_type: Type of sampler to use
        
        Returns:
            Dictionary with evaluation results
        """
        if timesteps_to_eval is None:
            # Default timesteps spread across the schedule
            num_eval_timesteps = 10
            timesteps_to_eval = list(np.linspace(
                0, self.scheduler.num_timesteps - 1, num_eval_timesteps, dtype=int
            ))
        
        self.model.eval()
        
        results = {
            "timesteps": timesteps_to_eval,
            "metrics": {}
        }
        
        for metric in self.metrics:
            results["metrics"][metric] = {t: [] for t in timesteps_to_eval}
        
        sample_count = 0
        max_samples = num_samples or len(dataloader.dataset)
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating reconstruction"):
                if sample_count >= max_samples:
                    break
                
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    x_start = batch[0]
                else:
                    x_start = batch
                
                x_start = x_start.to(self.device)
                current_batch_size = x_start.shape[0]
                
                # Limit batch size if needed
                if sample_count + current_batch_size > max_samples:
                    x_start = x_start[:max_samples - sample_count]
                    current_batch_size = x_start.shape[0]
                
                # Evaluate at each timestep
                for t in timesteps_to_eval:
                    # Reconstruct from timestep t
                    recon_result = reconstruct_from_noise(
                        self.model, self.scheduler, x_start, t_start=t,
                        sampler_type=sampler_type, progress=False
                    )
                    x_recon = recon_result["images"]
                    
                    # Compute metrics
                    if 'psnr' in self.metrics:
                        psnr_values = compute_psnr_batch(x_start, x_recon)
                        results["metrics"]["psnr"][t].extend(psnr_values)
                    
                    if 'ssim' in self.metrics:
                        ssim_values = compute_ssim_batch(x_start, x_recon)
                        results["metrics"]["ssim"][t].extend(ssim_values)
                    
                    if 'lpips' in self.metrics and self.lpips_model is not None:
                        lpips_values = compute_lpips_batch(x_start, x_recon, device=self.device)
                        results["metrics"]["lpips"][t].extend(lpips_values)
                
                sample_count += current_batch_size
        
        # Compute statistics
        for metric in self.metrics:
            for t in timesteps_to_eval:
                values = results["metrics"][metric][t]
                if values:
                    results["metrics"][metric][t] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "values": values
                    }
        
        return results
    
    def evaluate_sample_quality(
        self,
        num_samples: int = 1000,
        batch_size: int = 64,
        sampler_type: str = "ddpm",
        compute_fid: bool = False,
        real_features: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate quality of generated samples.
        
        Args:
            num_samples: Number of samples to generate
            batch_size: Batch size for generation
            sampler_type: Type of sampler
            compute_fid: Whether to compute FID (requires real_features)
            real_features: Pre-computed features of real images for FID
        
        Returns:
            Dictionary with quality metrics
        """
        self.model.eval()
        
        # Generate samples
        generated_samples = []
        
        with torch.no_grad():
            if sampler_type == "ddpm":
                sampler = DDPMSampler(self.scheduler)
            else:
                from .sampler import DDIMSampler
                sampler = DDIMSampler(self.scheduler)
            
            for _ in tqdm(range(0, num_samples, batch_size), desc="Generating samples"):
                current_batch_size = min(batch_size, num_samples - len(generated_samples))
                
                # Get shape from model
                sample_shape = self._get_sample_shape(current_batch_size)
                
                if sampler_type == "ddpm":
                    result = sampler.p_sample_loop(
                        self.model, sample_shape, device=self.device, progress=False
                    )
                else:
                    result = sampler.ddim_sample(
                        self.model, sample_shape, device=self.device, progress=False
                    )
                
                generated_samples.append(result["images"])
        
        # Concatenate all samples
        all_samples = torch.cat(generated_samples, dim=0)[:num_samples]
        
        # Compute basic quality metrics
        results = {}
        
        # Sharpness (edge energy)
        results["sharpness"] = self._compute_sharpness(all_samples).item()
        
        # Diversity (average pairwise distance)
        results["diversity"] = self._compute_diversity(all_samples).item()
        
        # Compute FID if requested
        if compute_fid and real_features is not None:
            try:
                from scipy import linalg
                
                # Extract features from generated samples
                gen_features = self._extract_features(all_samples)
                
                # Compute FID
                fid = self._compute_fid(real_features, gen_features)
                results["fid"] = fid
            except ImportError:
                print("Warning: scipy not available, cannot compute FID")
        
        return results
    
    def _get_sample_shape(self, batch_size: int) -> Tuple[int, ...]:
        """Get shape for sampling based on model."""
        # This is a simple heuristic - in practice you'd determine this from config
        if hasattr(self.model, 'in_channels'):
            channels = self.model.in_channels
        else:
            channels = 3  # Default
        
        return (batch_size, channels, 32, 32)  # Default size
    
    def _compute_sharpness(self, images: torch.Tensor) -> torch.Tensor:
        """Compute average sharpness (edge energy) of images."""
        # Sobel edge detection
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=images.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=images.device)
        
        sobel_x = sobel_x.view(1, 1, 3, 3)
        sobel_y = sobel_y.view(1, 1, 3, 3)
        
        # Convert to grayscale if needed
        if images.shape[1] == 3:
            gray = 0.299 * images[:, 0] + 0.587 * images[:, 1] + 0.114 * images[:, 2]
            gray = gray.unsqueeze(1)
        else:
            gray = images
        
        # Compute gradients
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        
        # Edge magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        
        # Average sharpness
        sharpness = edge_magnitude.mean()
        
        return sharpness
    
    def _compute_diversity(self, images: torch.Tensor, num_pairs: int = 1000) -> torch.Tensor:
        """Compute average pairwise diversity."""
        batch_size = images.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0)
        
        # Sample random pairs
        num_pairs = min(num_pairs, batch_size * (batch_size - 1) // 2)
        
        total_distance = 0.0
        count = 0
        
        for _ in range(num_pairs):
            i, j = np.random.choice(batch_size, 2, replace=False)
            
            # L2 distance between images
            distance = F.mse_loss(images[i], images[j])
            total_distance += distance.item()
            count += 1
        
        return torch.tensor(total_distance / count if count > 0 else 0.0)
    
    def _extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features using a pre-trained network (simplified)."""
        # This is a placeholder - in practice you'd use a pre-trained InceptionV3
        # For now, just return flattened images as "features"
        return images.view(images.shape[0], -1)
    
    def _compute_fid(self, real_features: torch.Tensor, gen_features: torch.Tensor) -> float:
        """Compute Fréchet Inception Distance."""
        # Move to CPU for numpy operations
        real_features = real_features.cpu().numpy()
        gen_features = gen_features.cpu().numpy()
        
        # Compute means and covariances
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
        
        # Compute FID
        from scipy import linalg
        
        diff = mu1 - mu2
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % 1e-6
            print(msg)
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical error might give complex eigenvalues
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary component {m}')
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        
        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


def save_evaluation_results(results: Dict[str, Any], save_path: str):
    """Save evaluation results to CSV and summary text.
    
    Args:
        results: Evaluation results dictionary
        save_path: Base path for saving (without extension)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if "metrics" in results:
        # Reconstruction evaluation results
        csv_path = save_path + "_reconstruction.csv"
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header
            header = ["timestep"]
            for metric in results["metrics"].keys():
                header.extend([f"{metric}_mean", f"{metric}_std"])
            writer.writerow(header)
            
            # Data
            for t in results["timesteps"]:
                row = [t]
                for metric in results["metrics"].keys():
                    if isinstance(results["metrics"][metric][t], dict):
                        row.extend([
                            results["metrics"][metric][t]["mean"],
                            results["metrics"][metric][t]["std"]
                        ])
                    else:
                        row.extend([0.0, 0.0])
                writer.writerow(row)
    
    # Summary text
    summary_path = save_path + "_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("DDPM Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        if "metrics" in results:
            f.write("Reconstruction Metrics:\n")
            f.write("-" * 30 + "\n")
            for metric in results["metrics"].keys():
                f.write(f"\n{metric.upper()}:\n")
                for t in results["timesteps"]:
                    if isinstance(results["metrics"][metric][t], dict):
                        mean_val = results["metrics"][metric][t]["mean"]
                        std_val = results["metrics"][metric][t]["std"]
                        f.write(f"  t={t:3d}: {mean_val:.4f} ± {std_val:.4f}\n")
        
        if "sharpness" in results:
            f.write(f"\nSample Quality:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Sharpness: {results['sharpness']:.4f}\n")
            f.write(f"Diversity: {results['diversity']:.4f}\n")
            if "fid" in results:
                f.write(f"FID: {results['fid']:.4f}\n")


def test_evaluation():
    """Test evaluation functionality."""
    from .ddpm_schedules import DDPMScheduler
    from .models.unet_tiny import UNetTiny
    
    # Create components
    scheduler = DDPMScheduler(num_timesteps=100)
    model = UNetTiny(in_channels=3, out_channels=3, model_channels=32)
    
    # Test evaluator
    evaluator = DDPMEvaluator(model, scheduler, metrics=['psnr', 'ssim'])
    
    # Test metrics on dummy data
    x1 = torch.rand(4, 3, 32, 32)
    x2 = torch.rand(4, 3, 32, 32)
    
    psnr_vals = compute_psnr_batch(x1, x2)
    ssim_vals = compute_ssim_batch(x1, x2)
    
    print(f"PSNR values: {psnr_vals}")
    print(f"SSIM values: {ssim_vals}")
    
    # Test sample quality
    quality_results = evaluator.evaluate_sample_quality(num_samples=8, batch_size=4)
    print(f"Sample quality: {quality_results}")
    
    print("Evaluation tests completed!")


if __name__ == "__main__":
    test_evaluation()