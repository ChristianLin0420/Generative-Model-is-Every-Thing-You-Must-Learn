"""
Quality metrics: FID-proxy, PSNR/SSIM/LPIPS for evaluating different beta schedules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: LPIPS not available. Install with: pip install lpips")


class FIDFeatureExtractor(nn.Module):
    """Simple feature extractor for FID-proxy computation."""
    
    def __init__(self, input_channels: int = 1):
        super().__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.flatten = nn.Flatten()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.flatten(features)


def compute_fid_proxy(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device,
    batch_size: int = 64
) -> float:
    """
    Compute FID-proxy using simple feature extractor.
    
    Args:
        real_images: Real images [N, C, H, W]
        fake_images: Generated images [N, C, H, W]
        device: Device
        batch_size: Batch size for processing
        
    Returns:
        FID-proxy score (lower is better)
    """
    # Create feature extractor
    extractor = FIDFeatureExtractor(real_images.shape[1]).to(device)
    extractor.eval()
    
    def extract_features(images):
        features = []
        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size].to(device)
                batch_features = extractor(batch)
                features.append(batch_features.cpu())
        return torch.cat(features, dim=0)
    
    # Extract features
    real_features = extract_features(real_images)
    fake_features = extract_features(fake_images)
    
    # Compute statistics
    real_mean = real_features.mean(dim=0)
    fake_mean = fake_features.mean(dim=0)
    
    real_cov = torch.cov(real_features.T)
    fake_cov = torch.cov(fake_features.T)
    
    # Compute FID-like distance
    mean_diff = real_mean - fake_mean
    mean_dist = torch.dot(mean_diff, mean_diff)
    
    # Simplified covariance distance (trace difference)
    cov_dist = torch.trace(real_cov) + torch.trace(fake_cov) - 2 * torch.trace(
        torch.sqrt(torch.matmul(real_cov, fake_cov) + 1e-6 * torch.eye(real_cov.shape[0]))
    )
    
    fid_proxy = mean_dist + cov_dist
    return fid_proxy.item()


def compute_psnr_batch(
    images1: torch.Tensor,
    images2: torch.Tensor,
    data_range: float = 2.0  # For [-1, 1] range
) -> float:
    """
    Compute PSNR between two batches of images.
    
    Args:
        images1: First batch [N, C, H, W]
        images2: Second batch [N, C, H, W]
        data_range: Data range (2.0 for [-1, 1], 1.0 for [0, 1])
        
    Returns:
        Average PSNR
    """
    # Convert to numpy and [0, 1] range
    if data_range == 2.0:  # [-1, 1] -> [0, 1]
        img1_np = ((images1 + 1) / 2).clamp(0, 1).cpu().numpy()
        img2_np = ((images2 + 1) / 2).clamp(0, 1).cpu().numpy()
    else:  # Already [0, 1]
        img1_np = images1.clamp(0, 1).cpu().numpy()
        img2_np = images2.clamp(0, 1).cpu().numpy()
    
    psnr_values = []
    for i in range(len(img1_np)):
        img1 = np.transpose(img1_np[i], (1, 2, 0))  # CHW -> HWC
        img2 = np.transpose(img2_np[i], (1, 2, 0))
        
        if img1.shape[2] == 1:  # Grayscale
            img1 = img1.squeeze(-1)
            img2 = img2.squeeze(-1)
        
        psnr = peak_signal_noise_ratio(img1, img2, data_range=1.0)
        psnr_values.append(psnr)
    
    return np.mean(psnr_values)


def compute_ssim_batch(
    images1: torch.Tensor,
    images2: torch.Tensor,
    data_range: float = 2.0
) -> float:
    """
    Compute SSIM between two batches of images.
    
    Args:
        images1: First batch [N, C, H, W]
        images2: Second batch [N, C, H, W]
        data_range: Data range
        
    Returns:
        Average SSIM
    """
    # Convert to numpy and [0, 1] range
    if data_range == 2.0:
        img1_np = ((images1 + 1) / 2).clamp(0, 1).cpu().numpy()
        img2_np = ((images2 + 1) / 2).clamp(0, 1).cpu().numpy()
    else:
        img1_np = images1.clamp(0, 1).cpu().numpy()
        img2_np = images2.clamp(0, 1).cpu().numpy()
    
    ssim_values = []
    for i in range(len(img1_np)):
        img1 = np.transpose(img1_np[i], (1, 2, 0))
        img2 = np.transpose(img2_np[i], (1, 2, 0))
        
        if img1.shape[2] == 1:  # Grayscale
            img1 = img1.squeeze(-1)
            img2 = img2.squeeze(-1)
            multichannel = False
        else:
            multichannel = True
        
        ssim = structural_similarity(
            img1, img2, 
            data_range=1.0, 
            multichannel=multichannel,
            channel_axis=-1 if multichannel else None
        )
        ssim_values.append(ssim)
    
    return np.mean(ssim_values)


def compute_lpips_batch(
    images1: torch.Tensor,
    images2: torch.Tensor,
    device: torch.device,
    net: str = 'alex'
) -> float:
    """
    Compute LPIPS between two batches of images.
    
    Args:
        images1: First batch [N, C, H, W]
        images2: Second batch [N, C, H, W]
        device: Device
        net: Network to use ('alex', 'vgg', 'squeeze')
        
    Returns:
        Average LPIPS distance
    """
    if not LPIPS_AVAILABLE:
        return 0.0
    
    loss_fn = lpips.LPIPS(net=net).to(device)
    
    with torch.no_grad():
        # LPIPS expects [-1, 1] range
        if images1.min() >= 0:  # Convert [0, 1] -> [-1, 1]
            images1 = images1 * 2 - 1
            images2 = images2 * 2 - 1
        
        # Handle grayscale by replicating to 3 channels
        if images1.shape[1] == 1:
            images1 = images1.repeat(1, 3, 1, 1)
            images2 = images2.repeat(1, 3, 1, 1)
        
        distances = loss_fn(images1.to(device), images2.to(device))
        return distances.mean().item()


class QualityEvaluator:
    """Comprehensive quality evaluator for diffusion models."""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def evaluate_reconstruction(
        self,
        original_images: torch.Tensor,
        reconstructed_images: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate reconstruction quality (teacher-forced).
        
        Args:
            original_images: Original clean images [N, C, H, W]
            reconstructed_images: Reconstructed images [N, C, H, W]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # PSNR
        metrics['psnr'] = compute_psnr_batch(original_images, reconstructed_images)
        
        # SSIM
        metrics['ssim'] = compute_ssim_batch(original_images, reconstructed_images)
        
        # LPIPS (if available and RGB)
        if LPIPS_AVAILABLE and original_images.shape[1] >= 3:
            metrics['lpips'] = compute_lpips_batch(
                original_images, reconstructed_images, self.device
            )
        else:
            metrics['lpips'] = 0.0
        
        return metrics
    
    def evaluate_generation(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate generation quality against real dataset.
        
        Args:
            real_images: Real dataset images [N, C, H, W]
            generated_images: Generated images [N, C, H, W]
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # FID-proxy
        metrics['fid_proxy'] = compute_fid_proxy(
            real_images, generated_images, self.device
        )
        
        # Sample quality statistics
        generated_np = generated_images.cpu().numpy()
        metrics['sample_mean'] = float(np.mean(generated_np))
        metrics['sample_std'] = float(np.std(generated_np))
        metrics['sample_min'] = float(np.min(generated_np))
        metrics['sample_max'] = float(np.max(generated_np))
        
        return metrics


def evaluate_run(
    run_dir: str,
    real_dataloader,
    config: Dict[str, any],
    device: torch.device,
    num_samples: int = 1000
) -> Dict[str, float]:
    """
    Evaluate a single run comprehensively.
    
    Args:
        run_dir: Run directory containing checkpoints
        real_dataloader: DataLoader for real data
        config: Configuration dictionary
        device: Device
        num_samples: Number of samples to generate for evaluation
        
    Returns:
        Dictionary of all metrics
    """
    from .models.unet_small import UNetSmall
    from .utils import load_checkpoint
    from .schedules import get_schedule
    from .sampler import DDPMSampler
    
    run_path = Path(run_dir)
    
    # Load model
    model_config = config['model']
    model = UNetSmall(
        in_channels=model_config['in_ch'],
        out_channels=model_config['in_ch'],
        base_channels=model_config['base_ch'],
        channel_multipliers=model_config['ch_mult'],
        time_embed_dim=model_config['time_embed_dim']
    ).to(device)
    
    # Load best checkpoint
    ckpt_path = run_path / 'ckpts' / 'ema.pt'
    if not ckpt_path.exists():
        ckpt_path = run_path / 'ckpts' / 'best.pt'
    
    load_checkpoint(ckpt_path, model, device=device)
    
    # Get schedule
    diffusion_config = config['diffusion']
    schedule_kwargs = {}
    if 'beta_min' in diffusion_config:
        schedule_kwargs['beta_min'] = diffusion_config['beta_min']
    if 'beta_max' in diffusion_config:
        schedule_kwargs['beta_max'] = diffusion_config['beta_max']
    if 'cosine_s' in diffusion_config:
        schedule_kwargs['s'] = diffusion_config['cosine_s']
    
    schedule = get_schedule(
        diffusion_config['schedule'],
        diffusion_config['T'],
        **schedule_kwargs
    )
    
    for key in schedule:
        schedule[key] = schedule[key].to(device)
    
    # Create evaluator
    evaluator = QualityEvaluator(device)
    
    # Generate samples
    sampler = DDPMSampler(
        model,
        schedule['betas'],
        schedule['alphas'],
        schedule['alpha_bars']
    )
    
    # Determine shape
    data_config = config['data']
    if data_config['dataset'].lower() == 'mnist':
        shape = (num_samples, 1, 28, 28)
    else:
        shape = (num_samples, 3, 32, 32)
    
    print(f"Generating {num_samples} samples for evaluation...")
    with torch.no_grad():
        generated_samples = sampler.sample(shape, device)
    
    # Collect real images
    print("Collecting real images...")
    real_images = []
    for batch_idx, (images, _) in enumerate(real_dataloader):
        real_images.append(images)
        if len(real_images) * images.shape[0] >= num_samples:
            break
    
    real_images = torch.cat(real_images, dim=0)[:num_samples]
    
    # Evaluate generation quality
    generation_metrics = evaluator.evaluate_generation(real_images, generated_samples)
    
    # Add timing information if available
    metrics_csv = run_path / 'logs' / 'metrics.csv'
    if metrics_csv.exists():
        df = pd.read_csv(metrics_csv)
        if 'epoch_time' in df.columns:
            generation_metrics['avg_epoch_time'] = df['epoch_time'].mean()
            generation_metrics['total_train_time'] = df['epoch_time'].sum()
    
    return generation_metrics


def create_comparison_report(
    run_dirs: List[str],
    config_paths: List[str],
    real_dataloader,
    device: torch.device,
    save_dir: str
) -> None:
    """
    Create comprehensive comparison report across runs.
    
    Args:
        run_dirs: List of run directories
        config_paths: List of config paths
        real_dataloader: Real data loader
        device: Device
        save_dir: Directory to save comparison results
    """
    from .utils import load_config
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Evaluate all runs
    all_metrics = []
    schedule_names = []
    
    for run_dir, config_path in zip(run_dirs, config_paths):
        config = load_config(config_path)
        schedule_name = config['diffusion']['schedule']
        schedule_names.append(schedule_name)
        
        print(f"Evaluating {schedule_name} schedule...")
        metrics = evaluate_run(run_dir, real_dataloader, config, device)
        metrics['schedule'] = schedule_name
        all_metrics.append(metrics)
    
    # Create comparison DataFrame
    df = pd.DataFrame(all_metrics)
    
    # Save CSV
    csv_path = save_path / 'comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"Comparison CSV saved to {csv_path}")
    
    # Create plots
    create_quality_plots(df, save_path)
    
    # Create markdown report
    create_markdown_report(df, save_path / 'report.md')


def create_quality_plots(df: pd.DataFrame, save_dir: Path) -> None:
    """Create quality comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    metrics_to_plot = ['fid_proxy', 'sample_std', 'avg_epoch_time', 'total_train_time']
    titles = ['FID Proxy (lower=better)', 'Sample Std Dev', 'Avg Epoch Time (s)', 'Total Train Time (s)']
    
    for i, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[i // 2, i % 2]
        
        if metric in df.columns:
            bars = ax.bar(df['schedule'], df[metric])
            ax.set_title(title)
            ax.set_ylabel(metric)
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plot_path = save_dir / 'quality_vs_schedule.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Quality plots saved to {plot_path}")


def create_markdown_report(df: pd.DataFrame, save_path: Path) -> None:
    """Create markdown summary report."""
    report = []
    report.append("# Beta Schedule Comparison Report\n")
    
    # Summary table
    report.append("## Summary Metrics\n")
    report.append(df.to_markdown(index=False))
    report.append("\n")
    
    # Key findings
    report.append("## Key Findings\n")
    
    if 'fid_proxy' in df.columns:
        best_fid_idx = df['fid_proxy'].idxmin()
        best_schedule = df.loc[best_fid_idx, 'schedule']
        report.append(f"- **Best Generation Quality**: {best_schedule} schedule (lowest FID-proxy: {df.loc[best_fid_idx, 'fid_proxy']:.3f})")
    
    if 'avg_epoch_time' in df.columns:
        fastest_idx = df['avg_epoch_time'].idxmin()
        fastest_schedule = df.loc[fastest_idx, 'schedule']
        report.append(f"- **Fastest Training**: {fastest_schedule} schedule ({df.loc[fastest_idx, 'avg_epoch_time']:.2f}s/epoch)")
    
    if 'sample_std' in df.columns:
        most_stable_idx = df['sample_std'].idxmin()
        stable_schedule = df.loc[most_stable_idx, 'schedule']
        report.append(f"- **Most Stable Sampling**: {stable_schedule} schedule (std: {df.loc[most_stable_idx, 'sample_std']:.3f})")
    
    report.append("\n")
    report.append("## Schedule Characteristics\n")
    report.append("- **Linear**: Uniform noise addition, simple implementation")
    report.append("- **Cosine**: Slower early diffusion, preserves fine details longer")
    report.append("- **Quadratic**: Faster early diffusion, aggressive noise schedule")
    
    # Write report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Markdown report saved to {save_path}")


def test_quality_metrics():
    """Test quality metrics with dummy data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy images
    real_images = torch.randn(64, 1, 28, 28)
    fake_images = torch.randn(64, 1, 28, 28)
    
    evaluator = QualityEvaluator(device)
    
    # Test generation metrics
    gen_metrics = evaluator.evaluate_generation(real_images, fake_images)
    print("Generation metrics:", gen_metrics)
    
    # Test reconstruction metrics
    recon_metrics = evaluator.evaluate_reconstruction(real_images, fake_images)
    print("Reconstruction metrics:", recon_metrics)
    
    print("Quality metrics test completed!")


if __name__ == "__main__":
    test_quality_metrics()
