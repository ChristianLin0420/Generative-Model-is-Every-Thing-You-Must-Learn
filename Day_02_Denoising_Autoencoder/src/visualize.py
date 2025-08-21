"""
Visualization utilities for Day 2: Denoising Autoencoder
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from .utils import console, save_image_grid


def create_reconstruction_grid(
    clean_images: torch.Tensor,
    noisy_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
    save_path: Union[str, Path],
    num_samples: int = 8,
    titles: Optional[List[str]] = None
) -> None:
    """
    Create a grid showing clean, noisy, and reconstructed images side by side.
    
    Args:
        clean_images: Clean input images [B, C, H, W]
        noisy_images: Noisy input images [B, C, H, W]
        reconstructed_images: Model reconstructions [B, C, H, W]
        save_path: Path to save the grid
        num_samples: Number of samples to include
        titles: Optional titles for columns
    """
    num_samples = min(num_samples, clean_images.size(0))
    
    # Prepare grid data
    grid_data = []
    for i in range(num_samples):
        grid_data.extend([
            clean_images[i],
            noisy_images[i], 
            reconstructed_images[i]
        ])
    
    # Stack all images
    grid_tensor = torch.stack(grid_data)
    
    # Save grid
    save_image_grid(
        grid_tensor,
        save_path,
        nrow=3,  # 3 images per row (clean, noisy, recon)
        normalize=True,
        range=(0, 1)
    )
    
    console.print(f"[blue]Saved reconstruction grid to {save_path}[/blue]")


def create_sigma_panel(
    model: torch.nn.Module,
    clean_image: torch.Tensor,
    sigmas: List[float],
    save_path: Union[str, Path],
    device: torch.device,
    clip_range: Tuple[float, float] = (0, 1)
) -> None:
    """
    Create a panel showing the same image with different noise levels and reconstructions.
    
    Args:
        model: Trained denoising model
        clean_image: Single clean image [C, H, W]
        sigmas: List of noise levels to test
        save_path: Path to save the panel
        device: Computation device
        clip_range: Value range for clipping
    """
    model.eval()
    
    # Prepare panel data: [clean, noisy1, recon1, noisy2, recon2, ...]
    panel_data = [clean_image]
    
    with torch.no_grad():
        for sigma in sigmas:
            # Add noise
            noise = torch.randn_like(clean_image) * sigma
            noisy_image = torch.clamp(clean_image + noise, *clip_range)
            
            # Reconstruct
            noisy_batch = noisy_image.unsqueeze(0).to(device)
            recon_batch = model(noisy_batch)
            recon_image = recon_batch.squeeze(0).cpu()
            
            panel_data.extend([noisy_image, recon_image])
    
    # Create grid
    panel_tensor = torch.stack(panel_data)
    save_image_grid(
        panel_tensor,
        save_path,
        nrow=len(panel_data),
        normalize=True,
        range=(0, 1)
    )
    
    console.print(f"[blue]Saved sigma panel to {save_path}[/blue]")


def plot_metrics_curves(
    metrics_data: Dict[float, Dict[str, float]],
    save_path: Union[str, Path],
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot metrics (PSNR, SSIM, etc.) vs noise level.
    
    Args:
        metrics_data: Dictionary mapping sigma -> metrics dict
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Extract data
    sigmas = sorted(metrics_data.keys())
    
    metrics_names = list(next(iter(metrics_data.values())).keys())
    metrics_values = {name: [] for name in metrics_names}
    
    for sigma in sigmas:
        for name in metrics_names:
            metrics_values[name].append(metrics_data[sigma].get(name, 0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each metric
    plot_configs = [
        ('psnr', 'PSNR (dB)', 'blue'),
        ('ssim', 'SSIM', 'green'),
        ('mse', 'MSE', 'red'),
        ('mae', 'MAE', 'orange')
    ]
    
    for i, (metric_name, ylabel, color) in enumerate(plot_configs):
        if metric_name in metrics_values and i < len(axes):
            axes[i].plot(sigmas, metrics_values[metric_name], 'o-', color=color, linewidth=2, markersize=6)
            axes[i].set_xlabel('Noise Level (σ)')
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(f'{ylabel} vs Noise Level')
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplots
    for i in range(len(plot_configs), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[blue]Saved metrics curves to {save_path}[/blue]")


def plot_training_curves(
    train_csv: Union[str, Path],
    val_csv: Union[str, Path],
    save_path: Union[str, Path],
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plot training and validation curves from CSV logs.
    
    Args:
        train_csv: Path to training metrics CSV
        val_csv: Path to validation metrics CSV
        save_path: Path to save the plot
        figsize: Figure size
    """
    try:
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
    except FileNotFoundError as e:
        console.print(f"[red]Could not find CSV files: {e}[/red]")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Loss curves
    axes[0, 0].plot(train_df['epoch'], train_df['loss'], 'b-', label='Train', linewidth=2)
    axes[0, 0].plot(val_df['epoch'], val_df['loss'], 'r-', label='Validation', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # PSNR curves
    axes[0, 1].plot(train_df['epoch'], train_df['psnr'], 'b-', label='Train', linewidth=2)
    axes[0, 1].plot(val_df['epoch'], val_df['psnr'], 'r-', label='Validation', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('PSNR Progress')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # SSIM curves
    axes[1, 0].plot(train_df['epoch'], train_df['ssim'], 'b-', label='Train', linewidth=2)
    axes[1, 0].plot(val_df['epoch'], val_df['ssim'], 'r-', label='Validation', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('SSIM')
    axes[1, 0].set_title('SSIM Progress')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # MSE curves
    axes[1, 1].plot(train_df['epoch'], train_df['mse'], 'b-', label='Train', linewidth=2)
    axes[1, 1].plot(val_df['epoch'], val_df['mse'], 'r-', label='Validation', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('MSE')
    axes[1, 1].set_title('Mean Squared Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[blue]Saved training curves to {save_path}[/blue]")


def visualize_feature_maps(
    model: torch.nn.Module,
    input_image: torch.Tensor,
    save_path: Union[str, Path],
    device: torch.device
) -> None:
    """
    Visualize intermediate feature maps from UNet model.
    
    Args:
        model: UNet model with get_feature_maps method
        input_image: Input image [1, C, H, W]
        save_path: Path to save visualization
        device: Computation device
    """
    if not hasattr(model, 'get_feature_maps'):
        console.print("[yellow]Model does not support feature map visualization[/yellow]")
        return
    
    model.eval()
    with torch.no_grad():
        input_batch = input_image.to(device)
        features = model.get_feature_maps(input_batch)
    
    # Create visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    feature_names = list(features.keys())[:8]  # Limit to 8 features
    
    for i, name in enumerate(feature_names):
        if i >= len(axes):
            break
            
        feature_map = features[name][0]  # First batch item
        
        # Average across channels for visualization
        if feature_map.ndim == 3:  # [C, H, W]
            if feature_map.size(0) > 1:
                feature_viz = feature_map.mean(dim=0)  # Average channels
            else:
                feature_viz = feature_map[0]
        else:
            feature_viz = feature_map
        
        feature_viz = feature_viz.cpu().numpy()
        
        im = axes[i].imshow(feature_viz, cmap='viridis')
        axes[i].set_title(f'{name}\nShape: {feature_map.shape}')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    # Remove empty subplots
    for i in range(len(feature_names), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    console.print(f"[blue]Saved feature maps to {save_path}[/blue]")


def create_failure_cases_grid(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_path: Union[str, Path],
    num_cases: int = 8,
    metric: str = 'psnr',
    worst: bool = True
) -> None:
    """
    Create a grid showing failure cases (worst or best reconstructions).
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Computation device
        save_path: Path to save grid
        num_cases: Number of cases to show
        metric: Metric to use for ranking ('psnr', 'ssim', 'mse')
        worst: If True, show worst cases; if False, show best cases
    """
    from .metrics import MetricsCalculator
    
    model.eval()
    metrics_calc = MetricsCalculator(device)
    
    cases = []  # List of (score, clean, noisy, recon) tuples
    
    with torch.no_grad():
        for clean, noisy, _ in test_loader:
            clean = clean.to(device)
            noisy = noisy.to(device)
            
            recon = model(noisy)
            
            # Compute metrics for each sample
            for i in range(clean.size(0)):
                sample_metrics = metrics_calc.compute_all_metrics(
                    recon[i:i+1], clean[i:i+1]
                )
                score = sample_metrics[metric]
                cases.append((score, clean[i], noisy[i], recon[i]))
            
            if len(cases) >= num_cases * 10:  # Collect more than needed
                break
    
    # Sort by metric
    cases.sort(key=lambda x: x[0], reverse=not worst)
    
    # Take top/bottom cases
    selected_cases = cases[:num_cases]
    
    # Create grid
    grid_data = []
    for score, clean, noisy, recon in selected_cases:
        grid_data.extend([clean, noisy, recon])
    
    if grid_data:
        grid_tensor = torch.stack(grid_data)
        save_image_grid(grid_tensor, save_path, nrow=3)
        
        avg_score = np.mean([case[0] for case in selected_cases])
        case_type = "worst" if worst else "best"
        console.print(f"[blue]Saved {case_type} cases to {save_path} (avg {metric}: {avg_score:.4f})[/blue]")


def generate_visualization_report(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    config,
    device: torch.device,
    output_dir: Union[str, Path]
) -> None:
    """
    Generate comprehensive visualization report.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        config: Configuration object
        device: Computation device
        output_dir: Output directory for visualizations
    """
    output_dir = Path(output_dir)
    grids_dir = output_dir / "grids"
    panels_dir = output_dir / "panels"
    reports_dir = output_dir / "reports"
    
    grids_dir.mkdir(parents=True, exist_ok=True)
    panels_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    console.print("[bold blue]Generating visualization report[/bold blue]")
    
    # 1. Basic reconstruction grid
    console.print("1. Creating reconstruction grids...")
    clean, noisy, _ = next(iter(test_loader))
    clean, noisy = clean.to(device), noisy.to(device)
    
    with torch.no_grad():
        recon = model(clean)
    
    create_reconstruction_grid(
        clean.cpu(), noisy.cpu(), recon.cpu(),
        grids_dir / "basic_reconstructions.png"
    )
    
    # 2. Sigma sweep panels
    console.print("2. Creating sigma sweep panels...")
    test_sigmas = config.noise.test_sigmas[:5]  # Limit for visualization
    sample_image = clean[0].cpu()
    
    create_sigma_panel(
        model, sample_image, test_sigmas,
        panels_dir / "sigma_sweep.png",
        device
    )
    
    # 3. Failure cases
    console.print("3. Analyzing failure cases...")
    create_failure_cases_grid(
        model, test_loader, device,
        grids_dir / "worst_cases.png",
        worst=True
    )
    
    create_failure_cases_grid(
        model, test_loader, device,
        grids_dir / "best_cases.png", 
        worst=False
    )
    
    # 4. Training curves (if available)
    console.print("4. Creating training curves...")
    train_csv = output_dir / "logs" / "train_metrics.csv"
    val_csv = output_dir / "logs" / "val_metrics.csv"
    
    if train_csv.exists() and val_csv.exists():
        plot_training_curves(
            train_csv, val_csv,
            reports_dir / "training_curves.png"
        )
    
    console.print("[green]Visualization report complete![/green]")