"""
Quality assessment for DDPM samples.

Implements:
- FID-proxy using Inception-v3 or small CNN features
- LPIPS perceptual distance
- Statistical metrics
- Quality vs checkpoint analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import linalg
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")

try:
    from torchvision.models import inception_v3
    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False


class SimpleCNN(nn.Module):
    """Simple CNN for feature extraction when Inception is not available."""
    
    def __init__(self, input_channels: int = 3, feature_dim: int = 512):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv_layers(x)
        return self.fc(features)


class FeatureExtractor:
    """Feature extractor for computing FID-like metrics."""
    
    def __init__(self, model_type: str = "auto", device: Optional[torch.device] = None):
        """
        Initialize feature extractor.
        
        Args:
            model_type: "inception", "simple", or "auto"
            device: Device to run on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        
        if model_type == "auto":
            if INCEPTION_AVAILABLE:
                self.model_type = "inception"
            else:
                self.model_type = "simple"
        
        self._init_model()
        
    def _init_model(self):
        """Initialize the feature extraction model."""
        if self.model_type == "inception":
            if not INCEPTION_AVAILABLE:
                raise ValueError("Inception not available")
            
            self.model = inception_v3(pretrained=True, transform_input=False)
            self.model.fc = nn.Identity()  # Remove final classification layer
            self.model.eval()
            self.model.to(self.device)
            
            # Inception expects 299x299 input
            self.input_size = (299, 299)
            
        elif self.model_type == "simple":
            self.model = SimpleCNN(input_channels=3, feature_dim=512)
            self.model.eval()
            self.model.to(self.device)
            self.input_size = (32, 32)  # Can handle various sizes
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for feature extraction."""
        # Ensure images are in [0, 1] range
        if images.min() < -0.1 or images.max() > 1.1:  # Likely [-1, 1] range
            images = (images + 1) / 2
        
        images = torch.clamp(images, 0, 1)
        
        # Resize if needed
        if self.model_type == "inception":
            images = F.interpolate(images, size=self.input_size, mode='bilinear', align_corners=False)
            
            # Convert grayscale to RGB if needed
            if images.shape[1] == 1:
                images = images.repeat(1, 3, 1, 1)
            
            # Normalize to [-1, 1] for Inception
            images = images * 2 - 1
        
        elif self.model_type == "simple":
            # Handle grayscale/RGB mismatch
            if images.shape[1] == 1:  # Grayscale input
                self.model.conv_layers[0] = nn.Conv2d(1, 64, 3, padding=1).to(self.device)
            elif images.shape[1] == 3:  # RGB input
                pass  # Already configured for RGB
        
        return images
    
    @torch.no_grad()
    def extract_features(self, images: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: Image tensor of shape (N, C, H, W)
            batch_size: Batch size for processing
            
        Returns:
            Feature array of shape (N, feature_dim)
        """
        self.model.eval()
        
        images = self.preprocess_images(images)
        features_list = []
        
        for i in range(0, images.shape[0], batch_size):
            batch = images[i:i+batch_size].to(self.device)
            
            with torch.no_grad():
                if self.model_type == "inception":
                    batch_features = self.model(batch)
                    if isinstance(batch_features, tuple):  # Handle aux outputs
                        batch_features = batch_features[0]
                else:
                    batch_features = self.model(batch)
                
                features_list.append(batch_features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)


def calculate_fid(features_real: np.ndarray, features_generated: np.ndarray) -> float:
    """
    Calculate Fréchet Inception Distance (FID) between real and generated features.
    
    Args:
        features_real: Features from real images, shape (N, D)
        features_generated: Features from generated images, shape (M, D)
        
    Returns:
        FID score (lower is better)
    """
    # Calculate mean and covariance
    mu_real, sigma_real = np.mean(features_real, axis=0), np.cov(features_real, rowvar=False)
    mu_gen, sigma_gen = np.mean(features_generated, axis=0), np.cov(features_generated, rowvar=False)
    
    # Calculate Fréchet distance
    diff = mu_real - mu_gen
    
    # Product of covariances
    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_gen), disp=False)
    
    # Handle numerical issues
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real) + np.trace(sigma_gen) - 2 * np.trace(covmean)
    
    return float(fid)


def calculate_lpips_distance(images_real: torch.Tensor, 
                           images_generated: torch.Tensor,
                           device: Optional[torch.device] = None) -> float:
    """
    Calculate LPIPS distance between real and generated images.
    
    Args:
        images_real: Real images, shape (N, C, H, W)
        images_generated: Generated images, shape (M, C, H, W)
        device: Device to run on
        
    Returns:
        Average LPIPS distance
    """
    if not LPIPS_AVAILABLE:
        raise ImportError("LPIPS not available. Install with: pip install lpips")
    
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize LPIPS model
    lpips_model = lpips.LPIPS(net='alex').to(device)
    
    # Ensure images are in [-1, 1] range
    if images_real.min() >= 0:  # Likely [0, 1] range
        images_real = images_real * 2 - 1
    if images_generated.min() >= 0:
        images_generated = images_generated * 2 - 1
    
    distances = []
    
    # Compare pairs of images
    num_comparisons = min(len(images_real), len(images_generated), 100)  # Limit for efficiency
    
    with torch.no_grad():
        for i in range(num_comparisons):
            real_img = images_real[i:i+1].to(device)
            gen_img = images_generated[i:i+1].to(device)
            
            distance = lpips_model(real_img, gen_img)
            distances.append(distance.item())
    
    return float(np.mean(distances))


class QualityEvaluator:
    """Main class for evaluating sample quality."""
    
    def __init__(self, feature_extractor_type: str = "auto", 
                 device: Optional[torch.device] = None):
        """
        Initialize quality evaluator.
        
        Args:
            feature_extractor_type: Type of feature extractor to use
            device: Device to run on
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = FeatureExtractor(feature_extractor_type, device)
        
        # Cache for real dataset features
        self.real_features_cache = {}
    
    def compute_real_features(self, real_images: torch.Tensor, 
                            cache_key: Optional[str] = None) -> np.ndarray:
        """
        Compute and cache features for real images.
        
        Args:
            real_images: Real images tensor
            cache_key: Optional key for caching features
            
        Returns:
            Feature array
        """
        if cache_key and cache_key in self.real_features_cache:
            return self.real_features_cache[cache_key]
        
        features = self.feature_extractor.extract_features(real_images)
        
        if cache_key:
            self.real_features_cache[cache_key] = features
        
        return features
    
    def evaluate_samples(self, generated_samples: torch.Tensor,
                        real_samples: torch.Tensor,
                        compute_lpips: bool = True) -> Dict[str, float]:
        """
        Evaluate generated samples against real samples.
        
        Args:
            generated_samples: Generated images
            real_samples: Real images for comparison
            compute_lpips: Whether to compute LPIPS
            
        Returns:
            Dictionary of metrics
        """
        results = {}
        
        # Extract features
        print("Extracting features from generated samples...")
        gen_features = self.feature_extractor.extract_features(generated_samples)
        
        print("Extracting features from real samples...")
        real_features = self.feature_extractor.extract_features(real_samples)
        
        # Calculate FID
        print("Computing FID score...")
        fid_score = calculate_fid(real_features, gen_features)
        results['fid_proxy'] = fid_score
        
        # Calculate LPIPS if requested
        if compute_lpips and LPIPS_AVAILABLE:
            try:
                print("Computing LPIPS distance...")
                lpips_score = calculate_lpips_distance(real_samples, generated_samples, self.device)
                results['lpips'] = lpips_score
            except Exception as e:
                print(f"Warning: LPIPS computation failed: {e}")
                results['lpips'] = None
        else:
            results['lpips'] = None
        
        # Basic statistics
        results.update(self._compute_basic_stats(generated_samples))
        
        return results
    
    def _compute_basic_stats(self, images: torch.Tensor) -> Dict[str, float]:
        """Compute basic statistical metrics."""
        images_flat = images.view(images.shape[0], -1)
        
        return {
            'mean_pixel_value': float(images.mean()),
            'std_pixel_value': float(images.std()),
            'min_pixel_value': float(images.min()),
            'max_pixel_value': float(images.max()),
            'sample_diversity': float(images_flat.std(dim=0).mean()),  # Average per-pixel std across samples
        }


def evaluate_checkpoint_quality(sampler, checkpoint_manager, real_samples: torch.Tensor,
                              num_samples: int = 100, image_shape: Tuple[int, int, int] = (3, 32, 32),
                              evaluator: Optional[QualityEvaluator] = None) -> pd.DataFrame:
    """
    Evaluate quality across multiple checkpoints.
    
    Args:
        sampler: DDPM sampler
        checkpoint_manager: Manager for loading checkpoints
        real_samples: Real samples for comparison
        num_samples: Number of samples to generate per checkpoint
        image_shape: Shape of images (C, H, W)
        evaluator: Quality evaluator (will create if None)
        
    Returns:
        DataFrame with quality metrics per checkpoint
    """
    if evaluator is None:
        evaluator = QualityEvaluator()
    
    results = []
    
    for metadata, model in tqdm(checkpoint_manager.iterate_checkpoints(sampler.model), 
                               desc="Evaluating checkpoints"):
        print(f"\nEvaluating checkpoint: {metadata['checkpoint_path']}")
        
        # Generate samples
        shape = (num_samples, *image_shape)
        sample_results = sampler.sample(shape, progress=False)
        generated_samples = sample_results['samples']
        
        # Evaluate quality
        metrics = evaluator.evaluate_samples(generated_samples, real_samples)
        
        # Add metadata
        result = {
            'checkpoint': Path(metadata['checkpoint_path']).name,
            'epoch': metadata.get('epoch'),
            'checkpoint_index': metadata.get('checkpoint_index'),
            **metrics
        }
        
        results.append(result)
        
        print(f"FID-proxy: {metrics['fid_proxy']:.3f}")
        if metrics['lpips'] is not None:
            print(f"LPIPS: {metrics['lpips']:.3f}")
    
    return pd.DataFrame(results)


def save_quality_results(results_df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """Save quality evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = output_dir / "quality_metrics.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved quality metrics to {csv_path}")
    
    # Save summary
    summary_path = output_dir / "quality_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Quality Evaluation Summary\n")
        f.write("=" * 30 + "\n\n")
        
        if len(results_df) > 0:
            f.write(f"Number of checkpoints evaluated: {len(results_df)}\n")
            
            # Best checkpoint by FID
            best_fid_idx = results_df['fid_proxy'].idxmin()
            best_ckpt = results_df.loc[best_fid_idx]
            f.write(f"Best FID score: {best_ckpt['fid_proxy']:.3f} (checkpoint: {best_ckpt['checkpoint']})\n")
            
            # FID range
            fid_min, fid_max = results_df['fid_proxy'].min(), results_df['fid_proxy'].max()
            f.write(f"FID range: {fid_min:.3f} - {fid_max:.3f}\n")
            
            if 'lpips' in results_df.columns and results_df['lpips'].notna().any():
                lpips_mean = results_df['lpips'].mean()
                f.write(f"Average LPIPS: {lpips_mean:.3f}\n")
        
        f.write("\nDetailed Results:\n")
        f.write(results_df.to_string(index=False))
    
    print(f"Saved quality summary to {summary_path}")


# Utility functions for loading real data
def load_dataset_samples(dataset_name: str, data_root: str, 
                        num_samples: int = 1000) -> torch.Tensor:
    """
    Load real samples from dataset for comparison.
    
    Args:
        dataset_name: "mnist" or "cifar10"
        data_root: Root directory for datasets
        num_samples: Number of samples to load
        
    Returns:
        Real sample tensor
    """
    from torchvision import datasets, transforms
    
    if dataset_name.lower() == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # [-1, 1]
        ])
        dataset = datasets.MNIST(data_root, train=True, transform=transform, download=True)
        
    elif dataset_name.lower() == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
        ])
        dataset = datasets.CIFAR10(data_root, train=True, transform=transform, download=True)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Sample subset
    indices = torch.randperm(len(dataset))[:num_samples]
    samples = torch.stack([dataset[i][0] for i in indices])
    
    return samples
