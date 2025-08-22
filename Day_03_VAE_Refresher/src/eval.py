"""
Evaluation utilities for VAE models.
Includes reconstruction metrics (PSNR/SSIM/LPIPS), IWAE bounds, and FID-proxy features.
"""

import os
import csv
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from omegaconf import DictConfig

from .models.vae_conv import VAEConv
from .models.vae_resnet import VAEResNet  
from .models.vae_mlp import VAEMLP
from .dataset import create_unlabeled_dataloaders, get_sample_images
from .losses import compute_iwae_bound
from .utils import load_checkpoint, get_device, setup_logger, create_progress_bar


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[nn.Module, DictConfig]:
    """Load VAE model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', None)
    
    if config is None:
        raise ValueError("Checkpoint does not contain configuration")
    
    # Create model
    model_name = config.model.name
    in_channels = config.model.in_ch
    latent_dim = config.model.latent_dim
    base_channels = config.model.base_ch
    
    if model_name == "vae_conv":
        model = VAEConv(in_channels, latent_dim, base_channels)
    elif model_name == "vae_resnet":
        model = VAEResNet(in_channels, latent_dim, base_channels)
    elif model_name == "vae_mlp":
        img_size = 28 if config.data.dataset == "mnist" else 32
        model = VAEMLP(in_channels, img_size, latent_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, config


def compute_psnr(images1: torch.Tensor, images2: torch.Tensor) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR) between two batches of images.
    
    Args:
        images1: First batch of images [batch, channels, height, width]
        images2: Second batch of images [batch, channels, height, width]
    
    Returns:
        Average PSNR across the batch
    """
    # Convert to numpy and ensure [0, 1] range
    img1_np = torch.clamp(images1, 0, 1).cpu().numpy()
    img2_np = torch.clamp(images2, 0, 1).cpu().numpy()
    
    psnr_values = []
    for i in range(img1_np.shape[0]):
        # Convert from CHW to HWC format
        img1_hwc = np.transpose(img1_np[i], (1, 2, 0))
        img2_hwc = np.transpose(img2_np[i], (1, 2, 0))
        
        # Handle grayscale case
        if img1_hwc.shape[-1] == 1:
            img1_hwc = img1_hwc.squeeze(-1)
            img2_hwc = img2_hwc.squeeze(-1)
        
        psnr = peak_signal_noise_ratio(img1_hwc, img2_hwc, data_range=1.0)
        psnr_values.append(psnr)
    
    return np.mean(psnr_values)


def compute_ssim(images1: torch.Tensor, images2: torch.Tensor) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two batches of images.
    
    Args:
        images1: First batch of images [batch, channels, height, width]
        images2: Second batch of images [batch, channels, height, width]
    
    Returns:
        Average SSIM across the batch
    """
    # Convert to numpy and ensure [0, 1] range
    img1_np = torch.clamp(images1, 0, 1).cpu().numpy()
    img2_np = torch.clamp(images2, 0, 1).cpu().numpy()
    
    ssim_values = []
    for i in range(img1_np.shape[0]):
        # Convert from CHW to HWC format
        img1_hwc = np.transpose(img1_np[i], (1, 2, 0))
        img2_hwc = np.transpose(img2_np[i], (1, 2, 0))
        
        # Handle grayscale case
        if img1_hwc.shape[-1] == 1:
            img1_hwc = img1_hwc.squeeze(-1)
            img2_hwc = img2_hwc.squeeze(-1)
            
        # Compute SSIM
        if len(img1_hwc.shape) == 2:  # Grayscale
            ssim = structural_similarity(img1_hwc, img2_hwc, data_range=1.0)
        else:  # Color
            ssim = structural_similarity(img1_hwc, img2_hwc, multichannel=True, data_range=1.0)
        
        ssim_values.append(ssim)
    
    return np.mean(ssim_values)


def compute_lpips(images1: torch.Tensor, images2: torch.Tensor, device: torch.device) -> float:
    """
    Compute Learned Perceptual Image Patch Similarity (LPIPS).
    
    Args:
        images1: First batch of images [batch, channels, height, width]
        images2: Second batch of images [batch, channels, height, width]
        device: Computation device
    
    Returns:
        Average LPIPS across the batch
    """
    try:
        # LPIPS requires images to be at least 64x64, skip for smaller images
        if images1.shape[-1] < 64 or images1.shape[-2] < 64:
            print(f"Warning: LPIPS skipped - images too small ({images1.shape[-2]}x{images1.shape[-1]}) for reliable computation")
            return 0.0
            
        # Initialize LPIPS network (AlexNet-based)
        lpips_net = lpips.LPIPS(net='alex').to(device)
        
        # Ensure images are in [-1, 1] range for LPIPS
        img1_norm = 2.0 * torch.clamp(images1, 0, 1) - 1.0
        img2_norm = 2.0 * torch.clamp(images2, 0, 1) - 1.0
        
        # Handle grayscale images by repeating channels
        if img1_norm.shape[1] == 1:
            img1_norm = img1_norm.repeat(1, 3, 1, 1)
            img2_norm = img2_norm.repeat(1, 3, 1, 1)
        
        # Upsample small images for LPIPS (optional - for better compatibility)
        if img1_norm.shape[-1] < 224:
            import torch.nn.functional as F
            img1_norm = F.interpolate(img1_norm, size=(224, 224), mode='bilinear', align_corners=False)
            img2_norm = F.interpolate(img2_norm, size=(224, 224), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            lpips_values = lpips_net(img1_norm, img2_norm)
        
        return lpips_values.mean().item()
    
    except Exception as e:
        print(f"Warning: LPIPS computation failed: {e}")
        return 0.0


class SimpleInceptionFeatures(nn.Module):
    """
    Simple feature extractor for FID-like metrics.
    Uses a pre-trained ResNet instead of full Inception for simplicity.
    """
    
    def __init__(self, device: torch.device):
        super().__init__()
        # Use ResNet18 as feature extractor
        import torchvision.models as models
        resnet = models.resnet18(pretrained=True)
        
        # Remove the final classification layer
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.features.eval()
        
        # Preprocessing for ImageNet-pretrained model
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from images."""
        # Handle grayscale by repeating channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Ensure values are in [0, 1] and preprocess
        x = torch.clamp(x, 0, 1)
        x = self.preprocess(x)
        
        with torch.no_grad():
            features = self.features(x)
            features = features.view(features.size(0), -1)  # Flatten
        
        return features


def compute_fid_proxy(
    real_features: torch.Tensor, 
    generated_features: torch.Tensor
) -> float:
    """
    Compute a proxy for FID using precomputed features.
    
    Args:
        real_features: Features from real images [N, feature_dim]
        generated_features: Features from generated images [M, feature_dim]
    
    Returns:
        FID proxy score (lower is better)
    """
    # Convert to numpy
    real_feat = real_features.cpu().numpy()
    gen_feat = generated_features.cpu().numpy()
    
    # Compute statistics
    mu_real = np.mean(real_feat, axis=0)
    sigma_real = np.cov(real_feat, rowvar=False)
    
    mu_gen = np.mean(gen_feat, axis=0)
    sigma_gen = np.cov(gen_feat, rowvar=False)
    
    # Compute FID
    diff = mu_real - mu_gen
    covmean = np.real(np.sqrt(sigma_real @ sigma_gen))
    
    # Handle numerical issues
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * 1e-6
        covmean = np.real(np.sqrt((sigma_real + offset) @ (sigma_gen + offset)))
    
    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    
    return float(fid)


def evaluate_reconstruction_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate reconstruction quality metrics.
    
    Args:
        model: VAE model
        dataloader: Data loader
        device: Computation device
        num_batches: Number of batches to evaluate (None for all)
    
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_psnr = []
    all_ssim = []
    all_lpips = []
    
    with create_progress_bar("Evaluating reconstruction metrics") as progress:
        task = progress.add_task("Reconstruction", total=num_batches or len(dataloader))
        
        for batch_idx, batch_data in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break
            
            # Handle both labeled and unlabeled data
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                x, _ = batch_data
            else:
                x = batch_data
            
            x = x.to(device)
            
            with torch.no_grad():
                # Get reconstructions
                x_recon = model.reconstruct(x)
                
                # Compute metrics
                psnr = compute_psnr(x, x_recon)
                ssim = compute_ssim(x, x_recon)
                lpips_score = compute_lpips(x, x_recon, device)
                
                all_psnr.append(psnr)
                all_ssim.append(ssim)
                all_lpips.append(lpips_score)
            
            progress.update(task, advance=1)
    
    return {
        "psnr": np.mean(all_psnr),
        "psnr_std": np.std(all_psnr),
        "ssim": np.mean(all_ssim),
        "ssim_std": np.std(all_ssim),
        "lpips": np.mean(all_lpips),
        "lpips_std": np.std(all_lpips),
    }


def evaluate_iwae_bound(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 64,
    num_batches: Optional[int] = None,
    recon_loss_type: str = "bce"
) -> float:
    """
    Evaluate IWAE bound (tighter log-likelihood estimate).
    
    Args:
        model: VAE model
        dataloader: Data loader
        device: Computation device
        num_samples: Number of importance samples
        num_batches: Number of batches to evaluate
        recon_loss_type: Type of reconstruction loss
    
    Returns:
        Average IWAE bound
    """
    model.eval()
    iwae_bounds = []
    
    with create_progress_bar("Computing IWAE bound") as progress:
        task = progress.add_task("IWAE", total=num_batches or len(dataloader))
        
        for batch_idx, batch_data in enumerate(dataloader):
            if num_batches and batch_idx >= num_batches:
                break
            
            # Handle both labeled and unlabeled data
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                x, _ = batch_data
            else:
                x = batch_data
            
            x = x.to(device)
            
            # Compute IWAE bound
            iwae_bound = compute_iwae_bound(model, x, num_samples, recon_loss_type)
            iwae_bounds.append(iwae_bound.item())
            
            progress.update(task, advance=1)
    
    return np.mean(iwae_bounds)


def evaluate_fid_proxy(
    model: nn.Module,
    real_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_samples: int = 1000
) -> float:
    """
    Evaluate FID proxy between real and generated samples.
    
    Args:
        model: VAE model
        real_dataloader: Real data loader
        device: Computation device
        num_samples: Number of samples to generate and compare
    
    Returns:
        FID proxy score
    """
    model.eval()
    
    # Initialize feature extractor
    feature_extractor = SimpleInceptionFeatures(device)
    
    # Extract features from real images
    real_features = []
    real_samples_collected = 0
    
    with torch.no_grad():
        for batch_data in real_dataloader:
            if real_samples_collected >= num_samples:
                break
            
            # Handle both labeled and unlabeled data
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                x, _ = batch_data
            else:
                x = batch_data
            
            x = x.to(device)
            
            # Extract features
            feats = feature_extractor(x)
            real_features.append(feats)
            
            real_samples_collected += x.size(0)
        
        real_features = torch.cat(real_features, dim=0)[:num_samples]
    
    # Generate samples and extract features
    generated_features = []
    generated_samples_collected = 0
    
    with torch.no_grad():
        while generated_samples_collected < num_samples:
            batch_size = min(64, num_samples - generated_samples_collected)
            generated_batch = model.sample(batch_size, device)
            
            # Extract features
            feats = feature_extractor(generated_batch)
            generated_features.append(feats)
            
            generated_samples_collected += batch_size
        
        generated_features = torch.cat(generated_features, dim=0)[:num_samples]
    
    # Compute FID proxy
    fid_proxy = compute_fid_proxy(real_features, generated_features)
    
    return fid_proxy


def run_full_evaluation(config: DictConfig, checkpoint_path: str) -> Dict[str, float]:
    """
    Run comprehensive evaluation of VAE model.
    
    Args:
        config: Configuration
        checkpoint_path: Path to model checkpoint
    
    Returns:
        Dictionary of all evaluation metrics
    """
    device = get_device(config.device)
    logger = setup_logger("VAE_Evaluation")
    
    logger.info(f"Loading model from: {checkpoint_path}")
    
    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, device)
    
    # Create data loaders
    train_loader, test_loader = create_unlabeled_dataloaders(
        dataset=config.data.dataset,
        root=config.data.root,
        batch_size=64,  # Smaller batch size for evaluation
        num_workers=config.data.num_workers,
        normalize=config.data.normalize
    )
    
    logger.info("Starting evaluation...")
    
    # Evaluate reconstruction metrics
    logger.info("Computing reconstruction metrics...")
    recon_metrics = evaluate_reconstruction_metrics(
        model, test_loader, device, num_batches=20
    )
    
    # Evaluate IWAE bound
    logger.info("Computing IWAE bound...")
    iwae_bound = evaluate_iwae_bound(
        model, test_loader, device, 
        num_samples=config.eval.iwae_samples,
        num_batches=10,
        recon_loss_type=config.train.recon_loss
    )
    
    # Evaluate FID proxy (only for color images due to feature extractor)
    fid_proxy = 0.0
    if config.model.in_ch == 3:
        logger.info("Computing FID proxy...")
        fid_proxy = evaluate_fid_proxy(model, test_loader, device, num_samples=500)
    
    # Combine all metrics
    all_metrics = {
        **recon_metrics,
        "iwae_bound": iwae_bound,
        "fid_proxy": fid_proxy,
    }
    
    logger.info("Evaluation results:")
    for metric, value in all_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    return all_metrics


def save_metrics_to_csv(metrics: Dict[str, float], output_path: str) -> None:
    """Save evaluation metrics to CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Metric', 'Value'])
        
        for metric, value in metrics.items():
            writer.writerow([metric, value])


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    
    if len(sys.argv) != 3:
        print("Usage: python -m src.eval <config_path> <checkpoint_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    
    config = OmegaConf.load(config_path)
    
    # Run evaluation
    metrics = run_full_evaluation(config, checkpoint_path)
    
    # Save results
    output_path = os.path.join(config.log.out_dir, "logs", "metrics.csv")
    save_metrics_to_csv(metrics, output_path)
    
    print(f"Evaluation completed. Results saved to: {output_path}")