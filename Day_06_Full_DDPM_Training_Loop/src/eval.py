"""
Evaluation metrics for DDPM: PSNR, SSIM, LPIPS, and FID-proxy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from pathlib import Path
import math

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

from .sampler import DDPMSampler
from .utils import tensor_to_pil


def tensor_to_numpy(tensor: torch.Tensor, normalize: bool = True) -> np.ndarray:
    """Convert tensor to numpy array for skimage metrics"""
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
        
    if normalize:
        # Assume tensor is in [-1, 1], convert to [0, 1]
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
    # Convert to numpy and transpose to HWC
    numpy_array = tensor.detach().cpu().numpy()
    if numpy_array.shape[0] in [1, 3]:  # CHW format
        numpy_array = np.transpose(numpy_array, (1, 2, 0))
        
    # Handle grayscale
    if numpy_array.shape[2] == 1:
        numpy_array = numpy_array.squeeze(2)
        
    return numpy_array


def compute_psnr_tensor(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """Compute PSNR between two tensors"""
    # Ensure values are in [0, 1]
    img1 = torch.clamp((img1 + 1) / 2, 0, 1)
    img2 = torch.clamp((img2 + 1) / 2, 0, 1)
    
    mse = F.mse_loss(img1, img2, reduction='none')
    mse = mse.flatten(1).mean(dim=1)  # Average over spatial dims
    
    # Avoid log(0)
    mse = torch.clamp(mse, min=1e-10)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return psnr


def compute_ssim_tensor(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """
    Compute SSIM between two tensors (simplified version)
    Note: This is a simplified implementation. For accurate SSIM, use skimage.
    """
    # Ensure values are in [0, 1]
    img1 = torch.clamp((img1 + 1) / 2, 0, 1)
    img2 = torch.clamp((img2 + 1) / 2, 0, 1)
    
    # Create Gaussian window
    def gaussian_window(size: int, sigma: float = 1.5):
        coords = torch.arange(size, dtype=torch.float32)
        coords -= size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        return g.outer(g)
    
    window = gaussian_window(window_size).to(img1.device)
    window = window.expand(img1.shape[1], 1, window_size, window_size)
    
    # Constants for stability
    C1 = 0.01**2
    C2 = 0.03**2
    
    # Compute local means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img2.shape[1])
    
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    
    # Compute local variances and covariance
    sigma1_sq = F.conv2d(img1**2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2**2, window, padding=window_size//2, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.shape[1]) - mu1_mu2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
               
    return ssim_map.flatten(1).mean(dim=1)  # Average over spatial dims


class LPIPSMetric:
    """LPIPS perceptual similarity metric"""
    
    def __init__(self, net: str = 'alex', device: str = 'cpu'):
        if not LPIPS_AVAILABLE:
            raise ImportError("lpips package not available. Install with: pip install lpips")
            
        self.device = device
        self.lpips_fn = lpips.LPIPS(net=net).to(device)
        self.lpips_fn.eval()
        
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute LPIPS between two image tensors"""
        with torch.no_grad():
            # LPIPS expects [-1, 1] range
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            
            return self.lpips_fn(img1, img2).flatten()


class InceptionFeatureExtractor(nn.Module):
    """Inception features for FID-proxy metric"""
    
    def __init__(self, device: str = 'cpu'):
        super().__init__()
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision not available")
            
        # Load pretrained Inception v3
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(inception.children())[:-1])
        self.feature_extractor.to(device)
        self.device = device
        
        # Disable gradients
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract Inception features"""
        # Resize to 299x299 for Inception
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
            
        # Ensure RGB (3 channels)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        # Normalize to [0, 1] then to ImageNet stats
        x = torch.clamp((x + 1) / 2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        
        features = self.feature_extractor(x)
        return features.squeeze()


def compute_fid_proxy(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """
    Compute FID-proxy using Inception features
    This is a simplified version that just computes the distance between mean features
    """
    real_mean = real_features.mean(dim=0)
    fake_mean = fake_features.mean(dim=0)
    
    # Compute L2 distance between means (simplified FID)
    fid_proxy = torch.norm(real_mean - fake_mean, p=2).item()
    
    return fid_proxy


class DDPMEvaluator:
    """
    Comprehensive evaluator for DDPM models
    """
    
    def __init__(
        self,
        model: nn.Module,
        schedules: Any,  # DDPMSchedules
        sampler: DDPMSampler,
        device: str = 'cpu',
        use_lpips: bool = True,
        use_fid: bool = True
    ):
        self.model = model
        self.schedules = schedules
        self.sampler = sampler
        self.device = device
        
        # Initialize metrics
        self.lpips_metric = None
        if use_lpips and LPIPS_AVAILABLE:
            try:
                self.lpips_metric = LPIPSMetric(device=device)
            except Exception as e:
                print(f"Warning: Could not initialize LPIPS: {e}")
                
        self.inception_extractor = None
        if use_fid and TORCHVISION_AVAILABLE:
            try:
                self.inception_extractor = InceptionFeatureExtractor(device=device)
            except Exception as e:
                print(f"Warning: Could not initialize Inception extractor: {e}")
                
    def compute_reconstruction_metrics(
        self,
        original_images: torch.Tensor,
        reconstructed_images: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute reconstruction metrics between original and reconstructed images
        """
        metrics = {}
        
        # PSNR
        psnr_values = compute_psnr_tensor(original_images, reconstructed_images)
        metrics['psnr'] = psnr_values.mean().item()
        
        # SSIM (tensor version)
        ssim_values = compute_ssim_tensor(original_images, reconstructed_images)
        metrics['ssim_tensor'] = ssim_values.mean().item()
        
        # SSIM (skimage version - more accurate)
        if SKIMAGE_AVAILABLE:
            ssim_scores = []
            for i in range(original_images.shape[0]):
                orig = tensor_to_numpy(original_images[i])
                recon = tensor_to_numpy(reconstructed_images[i])
                
                if orig.ndim == 3:  # RGB
                    ssim_score = ssim(orig, recon, multichannel=True, data_range=1.0)
                else:  # Grayscale
                    ssim_score = ssim(orig, recon, data_range=1.0)
                ssim_scores.append(ssim_score)
                
            metrics['ssim'] = np.mean(ssim_scores)
            
        # LPIPS
        if self.lpips_metric is not None:
            try:
                lpips_values = self.lpips_metric(original_images, reconstructed_images)
                metrics['lpips'] = lpips_values.mean().item()
            except Exception as e:
                print(f"Warning: LPIPS computation failed: {e}")
                
        return metrics
        
    def evaluate_generation_quality(
        self,
        real_images: torch.Tensor,
        num_generated: int = 1000,
        batch_size: int = 64,
        sampling_method: str = "ddim",
        sampling_steps: int = 50
    ) -> Dict[str, float]:
        """
        Evaluate generation quality using various metrics
        """
        metrics = {}
        
        # Generate samples
        generated_images = []
        num_batches = math.ceil(num_generated / batch_size)
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, num_generated - i * batch_size)
            sample_shape = (current_batch_size, *real_images.shape[1:])
            
            samples = self.sampler.sample(
                model=self.model,
                shape=sample_shape,
                method=sampling_method,
                num_steps=sampling_steps if sampling_method == "ddim" else None,
                device=self.device,
                progress=False
            )
            
            generated_images.append(samples)
            
        generated_images = torch.cat(generated_images, dim=0)[:num_generated]
        
        # Compute FID-proxy
        if self.inception_extractor is not None:
            try:
                with torch.no_grad():
                    real_features = self.inception_extractor(real_images[:num_generated])
                    fake_features = self.inception_extractor(generated_images)
                    
                metrics['fid_proxy'] = compute_fid_proxy(real_features, fake_features)
            except Exception as e:
                print(f"Warning: FID computation failed: {e}")
                
        # Compute diversity metrics
        metrics.update(self._compute_diversity_metrics(generated_images))
        
        return metrics, generated_images
        
    def _compute_diversity_metrics(self, images: torch.Tensor) -> Dict[str, float]:
        """Compute diversity metrics for generated images"""
        metrics = {}
        
        # Pairwise LPIPS diversity
        if self.lpips_metric is not None and len(images) > 1:
            try:
                lpips_distances = []
                # Sample pairs to avoid quadratic complexity
                num_pairs = min(100, len(images) * (len(images) - 1) // 2)
                indices = torch.randperm(len(images))
                
                for i in range(num_pairs):
                    idx1 = indices[i % len(images)]
                    idx2 = indices[(i + 1) % len(images)]
                    
                    dist = self.lpips_metric(
                        images[idx1:idx1+1], 
                        images[idx2:idx2+1]
                    )
                    lpips_distances.append(dist.item())
                    
                metrics['lpips_diversity'] = np.mean(lpips_distances)
            except Exception as e:
                print(f"Warning: LPIPS diversity computation failed: {e}")
                
        # Pixel-space diversity
        pixel_std = images.flatten(1).std(dim=0).mean().item()
        metrics['pixel_diversity'] = pixel_std
        
        return metrics
        
    def compute_calibration_curve(
        self,
        test_images: torch.Tensor,
        num_timesteps: int = 10,
        batch_size: int = 32
    ) -> Dict[str, List[float]]:
        """
        Compute calibration curve showing reconstruction quality vs timestep
        """
        timesteps = torch.linspace(0, self.schedules.num_timesteps - 1, num_timesteps).long()
        
        reconstruction_errors = {
            'timesteps': timesteps.tolist(),
            'mse': [],
            'psnr': []
        }
        
        self.model.eval()
        with torch.no_grad():
            for t in timesteps:
                batch_errors = []
                batch_psnrs = []
                
                for i in range(0, len(test_images), batch_size):
                    batch = test_images[i:i+batch_size].to(self.device)
                    batch_t = torch.full((len(batch),), t, device=self.device)
                    
                    # Add noise
                    noise = torch.randn_like(batch)
                    x_t = self.schedules.q_sample(batch, batch_t, noise)
                    
                    # Predict noise
                    predicted_noise = self.model(x_t, batch_t)
                    
                    # Compute errors
                    mse = F.mse_loss(predicted_noise, noise, reduction='none')
                    mse = mse.flatten(1).mean(dim=1)
                    batch_errors.extend(mse.cpu().tolist())
                    
                    # Compute PSNR for reconstructed images
                    # Predict x_0 from predicted noise
                    sqrt_alphas_cumprod_t = self.schedules.extract(
                        self.schedules.sqrt_alphas_cumprod, batch_t, batch.shape
                    )
                    sqrt_one_minus_alphas_cumprod_t = self.schedules.extract(
                        self.schedules.sqrt_one_minus_alphas_cumprod, batch_t, batch.shape
                    )
                    
                    pred_x0 = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
                    psnr_values = compute_psnr_tensor(batch, pred_x0)
                    batch_psnrs.extend(psnr_values.cpu().tolist())
                    
                reconstruction_errors['mse'].append(np.mean(batch_errors))
                reconstruction_errors['psnr'].append(np.mean(batch_psnrs))
                
        return reconstruction_errors
        
    def comprehensive_evaluation(
        self,
        test_loader: Any,  # DataLoader
        num_samples_eval: int = 1000,
        sampling_methods: List[str] = ['ddim'],
        sampling_steps: List[int] = [50]
    ) -> Dict[str, Any]:
        """
        Run comprehensive evaluation suite
        """
        results = {}
        
        # Get test images
        test_images = []
        for batch in test_loader:
            if isinstance(batch, (list, tuple)):
                test_images.append(batch[0])
            else:
                test_images.append(batch)
                
            if len(torch.cat(test_images)) >= num_samples_eval:
                break
                
        test_images = torch.cat(test_images)[:num_samples_eval]
        
        # Evaluation for each sampling method
        for method in sampling_methods:
            for steps in sampling_steps:
                key = f"{method}_{steps}steps" if method == "ddim" else method
                
                print(f"Evaluating {key}...")
                
                try:
                    method_results, generated = self.evaluate_generation_quality(
                        real_images=test_images,
                        num_generated=min(num_samples_eval, 500),  # Limit for speed
                        sampling_method=method,
                        sampling_steps=steps
                    )
                    results[key] = method_results
                    
                except Exception as e:
                    print(f"Warning: Evaluation failed for {key}: {e}")
                    results[key] = {"error": str(e)}
                    
        # Calibration curve
        try:
            calibration = self.compute_calibration_curve(test_images[:100])  # Subset for speed
            results['calibration'] = calibration
        except Exception as e:
            print(f"Warning: Calibration curve computation failed: {e}")
            
        return results
        
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save evaluation results to file"""
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
                
        results_serializable = convert_numpy(results)
        
        with open(output_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)
            
        print(f"Results saved to {output_path}")