"""
Visualization utilities for VAE models.
Includes reconstruction grids, latent traversal, 2D scatter plots, and interactive visualizations.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
import torch
import torch.nn as nn
from omegaconf import DictConfig

from .eval import load_model_from_checkpoint
from .dataset import get_sample_images, create_dataloaders
from .utils import get_device, setup_logger, save_image_grid, make_image_grid


def create_reconstruction_grid(
    model: nn.Module,
    sample_images: torch.Tensor,
    device: torch.device,
    num_images: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create reconstruction comparison grid.
    
    Args:
        model: VAE model
        sample_images: Sample images to reconstruct
        device: Computation device
        num_images: Number of images to show
    
    Returns:
        Original images, reconstructed images
    """
    model.eval()
    
    # Select subset of images
    sample_images = sample_images[:num_images].to(device)
    
    with torch.no_grad():
        # Get reconstructions
        reconstructions = model.reconstruct(sample_images)
        
        # Apply sigmoid for BCE models
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'deconv_layers'):
            reconstructions = torch.sigmoid(reconstructions)
    
    return sample_images, reconstructions


def create_latent_traversal(
    model: nn.Module,
    base_image: torch.Tensor,
    device: torch.device,
    dim_to_traverse: int,
    num_steps: int = 11,
    step_size: float = 3.0
) -> torch.Tensor:
    """
    Create latent space traversal for a specific dimension.
    
    Args:
        model: VAE model
        base_image: Base image to start from [1, channels, height, width]
        device: Computation device
        dim_to_traverse: Which latent dimension to traverse
        num_steps: Number of steps in traversal
        step_size: Range of traversal (±step_size)
    
    Returns:
        Traversal images [num_steps, channels, height, width]
    """
    model.eval()
    base_image = base_image.to(device)
    
    with torch.no_grad():
        # Encode base image
        mu, _ = model.encode(base_image)
        base_z = mu.clone()
        
        # Create traversal values
        traverse_values = torch.linspace(-step_size, step_size, num_steps, device=device)
        
        traversal_images = []
        for value in traverse_values:
            # Modify the specific dimension
            z_modified = base_z.clone()
            z_modified[0, dim_to_traverse] = value
            
            # Decode
            img = model.decode(z_modified)
            
            # Apply sigmoid for BCE models
            if hasattr(model, 'decoder') and hasattr(model.decoder, 'deconv_layers'):
                img = torch.sigmoid(img)
            
            traversal_images.append(img)
        
        traversal_grid = torch.cat(traversal_images, dim=0)
    
    return traversal_grid


def create_multi_dim_traversal(
    model: nn.Module,
    base_image: torch.Tensor,
    device: torch.device,
    dims_to_traverse: List[int],
    num_steps: int = 11,
    step_size: float = 3.0
) -> Dict[int, torch.Tensor]:
    """
    Create traversals for multiple latent dimensions.
    
    Args:
        model: VAE model
        base_image: Base image
        device: Computation device
        dims_to_traverse: List of dimensions to traverse
        num_steps: Steps per traversal
        step_size: Range of traversal
    
    Returns:
        Dictionary mapping dimension to traversal images
    """
    traversals = {}
    
    for dim in dims_to_traverse:
        traversals[dim] = create_latent_traversal(
            model, base_image, device, dim, num_steps, step_size
        )
    
    return traversals


def plot_2d_latent_scatter(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 10,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Create 2D scatter plot of latent space (only works for 2D latent space).
    
    Args:
        model: VAE model (must have latent_dim=2)
        dataloader: Labeled dataloader
        device: Computation device
        num_batches: Number of batches to process
        save_path: Path to save plot
    
    Returns:
        Matplotlib figure
    """
    if model.latent_dim != 2:
        raise ValueError("2D scatter plot only works for 2D latent space")
    
    model.eval()
    
    # Collect latent codes and labels
    latent_codes = []
    labels = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            x = x.to(device)
            mu, _ = model.encode(x)
            
            latent_codes.append(mu.cpu().numpy())
            labels.append(y.cpu().numpy())
    
    # Concatenate all batches
    latent_codes = np.vstack(latent_codes)
    labels = np.concatenate(labels)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use different colors for each class
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        mask = labels == label
        ax.scatter(
            latent_codes[mask, 0], 
            latent_codes[mask, 1],
            c=[color], 
            label=f'Class {label}',
            alpha=0.6,
            s=20
        )
    
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    ax.set_title('2D Latent Space Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_latent_distribution_comparison(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    num_batches: int = 20,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare learned latent distribution with prior.
    
    Args:
        model: VAE model
        dataloader: Data loader
        device: Computation device
        num_batches: Number of batches to analyze
        save_path: Path to save plot
    
    Returns:
        Matplotlib figure
    """
    model.eval()
    
    # Collect latent statistics
    all_mu = []
    all_logvar = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_idx >= num_batches:
                break
            
            # Handle both labeled and unlabeled data
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                x, _ = batch_data
            else:
                x = batch_data
            
            x = x.to(device)
            mu, logvar = model.encode(x)
            
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
    
    all_mu = torch.cat(all_mu, dim=0)
    all_logvar = torch.cat(all_logvar, dim=0)
    all_sigma = torch.exp(0.5 * all_logvar)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot mean distributions
    axes[0, 0].hist(all_mu.numpy().flatten(), bins=50, alpha=0.7, density=True, label='Learned μ')
    x_range = np.linspace(-3, 3, 100)
    axes[0, 0].plot(x_range, np.exp(-0.5 * x_range**2) / np.sqrt(2 * np.pi), 'r--', label='N(0,1)')
    axes[0, 0].set_title('Mean Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot std distributions
    axes[0, 1].hist(all_sigma.numpy().flatten(), bins=50, alpha=0.7, density=True, label='Learned σ')
    axes[0, 1].axvline(x=1.0, color='r', linestyle='--', label='Prior σ=1')
    axes[0, 1].set_title('Standard Deviation Distribution')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot mean vs std scatter (first few dimensions)
    max_dims = min(model.latent_dim, 100)  # Limit for visualization
    mu_subset = all_mu[:, :max_dims].numpy().flatten()
    sigma_subset = all_sigma[:, :max_dims].numpy().flatten()
    
    axes[1, 0].scatter(mu_subset, sigma_subset, alpha=0.1, s=1)
    axes[1, 0].set_title('Mean vs Standard Deviation')
    axes[1, 0].set_xlabel('Mean')
    axes[1, 0].set_ylabel('Standard Deviation')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot KL divergence per dimension
    kl_per_dim = -0.5 * (1 + all_logvar - all_mu.pow(2) - all_logvar.exp())
    avg_kl_per_dim = kl_per_dim.mean(dim=0)
    
    axes[1, 1].bar(range(len(avg_kl_per_dim)), avg_kl_per_dim.numpy())
    axes[1, 1].set_title('Average KL Divergence per Dimension')
    axes[1, 1].set_xlabel('Latent Dimension')
    axes[1, 1].set_ylabel('KL Divergence')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_comprehensive_visualization_report(
    config: DictConfig,
    checkpoint_path: str,
    output_dir: str
) -> None:
    """
    Create comprehensive visualization report for VAE.
    
    Args:
        config: Configuration
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save visualizations
    """
    device = get_device(config.device)
    logger = setup_logger("VAE_Visualization")
    
    # Create output directories
    grids_dir = os.path.join(output_dir, "grids")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(grids_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    logger.info(f"Loading model from: {checkpoint_path}")
    
    # Load model
    model, model_config = load_model_from_checkpoint(checkpoint_path, device)
    
    # Get sample images for reconstruction
    sample_images = get_sample_images(
        model_config.data.dataset,
        model_config.data.root,
        num_samples=32,
        normalize=model_config.data.normalize
    )
    
    # 1. Reconstruction Grid
    logger.info("Creating reconstruction grid...")
    original_images, reconstructed_images = create_reconstruction_grid(
        model, sample_images, device, num_images=16
    )
    
    # Combine originals and reconstructions
    comparison_images = torch.cat([original_images, reconstructed_images], dim=0)
    save_image_grid(
        comparison_images,
        os.path.join(grids_dir, "reconstruction_grid.png"),
        nrow=8,
        title="Reconstruction Comparison (Top: Original, Bottom: Reconstructed)"
    )
    
    # 2. Latent Traversals
    logger.info("Creating latent traversals...")
    base_image = sample_images[0:1]
    
    # Select dimensions to traverse (first few dimensions)
    dims_to_traverse = list(range(min(8, model.latent_dim)))
    traversals = create_multi_dim_traversal(
        model, base_image, device, dims_to_traverse, num_steps=11
    )
    
    # Save individual traversal grids
    for dim, traversal in traversals.items():
        save_image_grid(
            traversal,
            os.path.join(grids_dir, f"traverse_dim_{dim}.png"),
            nrow=11,
            title=f"Latent Traversal - Dimension {dim}"
        )
    
    # Create combined traversal grid
    if len(traversals) > 0:
        combined_traversals = torch.cat(list(traversals.values()), dim=0)
        save_image_grid(
            combined_traversals,
            os.path.join(grids_dir, "all_traversals.png"),
            nrow=11,
            title="All Latent Traversals"
        )
    
    # 3. 2D Latent Scatter (if applicable)
    if model.latent_dim == 2:
        logger.info("Creating 2D latent scatter plot...")
        # Need labeled dataloader for scatter plot
        try:
            train_loader, _, _ = create_dataloaders(
                dataset=model_config.data.dataset,
                root=model_config.data.root,
                batch_size=64,
                num_workers=2,
                normalize=model_config.data.normalize
            )
            
            plot_2d_latent_scatter(
                model, train_loader, device, num_batches=20,
                save_path=os.path.join(plots_dir, "latent_scatter_2d.png")
            )
        except Exception as e:
            logger.warning(f"Could not create 2D scatter plot: {e}")
    
    # 4. Latent Distribution Analysis
    logger.info("Creating latent distribution analysis...")
    from .dataset import create_unlabeled_dataloaders
    _, test_loader = create_unlabeled_dataloaders(
        dataset=model_config.data.dataset,
        root=model_config.data.root,
        batch_size=64,
        num_workers=2,
        normalize=model_config.data.normalize
    )
    
    plot_latent_distribution_comparison(
        model, test_loader, device, num_batches=20,
        save_path=os.path.join(plots_dir, "latent_distribution_analysis.png")
    )
    
    # 5. Loss Curves (if tensorboard logs exist)
    logger.info("Attempting to create loss curves...")
    try:
        from torch.utils.tensorboard import SummaryWriter
        import glob
        
        # Find tensorboard logs
        log_dir = os.path.join(config.log.out_dir, "logs")
        tb_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
        
        if tb_files:
            # TODO: Parse tensorboard logs and create loss curves
            logger.info("Tensorboard logs found, but parsing not implemented in this version")
        else:
            logger.info("No tensorboard logs found")
            
    except Exception as e:
        logger.warning(f"Could not create loss curves: {e}")
    
    logger.info(f"Visualization report completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    
    if len(sys.argv) != 4:
        print("Usage: python -m src.visualize <config_path> <checkpoint_path> <output_dir>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    config = OmegaConf.load(config_path)
    
    create_comprehensive_visualization_report(config, checkpoint_path, output_dir)