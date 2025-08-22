"""
Sampling utilities for VAE models.
Includes prior sampling, latent interpolation, and conditional generation.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict
from omegaconf import DictConfig

from .eval import load_model_from_checkpoint
from .dataset import get_sample_images
from .utils import get_device, setup_logger, save_image_grid


def sample_from_prior(
    model: nn.Module,
    num_samples: int,
    device: torch.device,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Sample images from the prior distribution p(z) = N(0, I).
    
    Args:
        model: VAE model
        num_samples: Number of samples to generate
        device: Computation device
        temperature: Temperature for sampling (1.0 = standard, >1 = more diverse)
    
    Returns:
        Generated images [num_samples, channels, height, width]
    """
    model.eval()
    
    with torch.no_grad():
        # Sample from prior with temperature scaling
        z = torch.randn(num_samples, model.latent_dim, device=device) * temperature
        
        # Decode to images
        samples = model.decode(z)
        
        # Apply sigmoid for BCE loss models (to get [0, 1] range)
        if hasattr(model, 'decoder') and isinstance(model.decoder.deconv_layers[-1], nn.ConvTranspose2d):
            # Check if this is likely a BCE model (no final activation in decoder)
            samples = torch.sigmoid(samples)
    
    return samples


def interpolate_between_samples(
    model: nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
    num_steps: int = 10,
    interpolation_mode: str = "linear"
) -> torch.Tensor:
    """
    Interpolate between two samples in latent space.
    
    Args:
        model: VAE model
        x1: First image [1, channels, height, width]
        x2: Second image [1, channels, height, width]
        num_steps: Number of interpolation steps
        interpolation_mode: "linear" or "spherical"
    
    Returns:
        Interpolated images [num_steps, channels, height, width]
    """
    model.eval()
    
    with torch.no_grad():
        # Encode both images to get latent representations
        mu1, _ = model.encode(x1)
        mu2, _ = model.encode(x2)
        
        if interpolation_mode == "linear":
            # Linear interpolation
            alphas = torch.linspace(0, 1, num_steps, device=x1.device)
            interpolated_z = []
            
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                interpolated_z.append(z_interp)
                
        elif interpolation_mode == "spherical":
            # Spherical linear interpolation (SLERP)
            # Normalize vectors
            mu1_norm = mu1 / torch.norm(mu1, dim=1, keepdim=True)
            mu2_norm = mu2 / torch.norm(mu2, dim=1, keepdim=True)
            
            # Compute angle between vectors
            dot = (mu1_norm * mu2_norm).sum(dim=1, keepdim=True)
            dot = torch.clamp(dot, -1.0, 1.0)  # Numerical stability
            theta = torch.acos(dot)
            
            alphas = torch.linspace(0, 1, num_steps, device=x1.device)
            interpolated_z = []
            
            for alpha in alphas:
                if theta.abs() < 1e-6:  # Vectors are almost parallel
                    z_interp = (1 - alpha) * mu1 + alpha * mu2
                else:
                    sin_theta = torch.sin(theta)
                    w1 = torch.sin((1 - alpha) * theta) / sin_theta
                    w2 = torch.sin(alpha * theta) / sin_theta
                    z_interp = w1 * mu1 + w2 * mu2
                
                interpolated_z.append(z_interp)
        else:
            raise ValueError(f"Unknown interpolation mode: {interpolation_mode}")
        
        # Stack all interpolated vectors
        z_stack = torch.cat(interpolated_z, dim=0)
        
        # Decode to images
        interpolated_images = model.decode(z_stack)
        
        # Apply sigmoid for BCE models
        if hasattr(model, 'decoder') and isinstance(model.decoder.deconv_layers[-1], nn.ConvTranspose2d):
            interpolated_images = torch.sigmoid(interpolated_images)
    
    return interpolated_images


def create_interpolation_grid(
    model: nn.Module,
    sample_images: torch.Tensor,
    device: torch.device,
    num_pairs: int = 8,
    num_steps: int = 10
) -> torch.Tensor:
    """
    Create a grid showing multiple interpolations.
    
    Args:
        model: VAE model
        sample_images: Sample images to interpolate between
        device: Computation device
        num_pairs: Number of interpolation pairs
        num_steps: Steps per interpolation
    
    Returns:
        Interpolation grid [num_pairs * num_steps, channels, height, width]
    """
    model.eval()
    
    # Select random pairs
    torch.manual_seed(42)  # For reproducible results
    indices = torch.randperm(len(sample_images))[:num_pairs * 2]
    
    all_interpolations = []
    
    for i in range(0, len(indices), 2):
        idx1, idx2 = indices[i], indices[i + 1]
        x1 = sample_images[idx1:idx1+1].to(device)
        x2 = sample_images[idx2:idx2+1].to(device)
        
        # Create interpolation
        interp = interpolate_between_samples(model, x1, x2, num_steps, "linear")
        all_interpolations.append(interp)
    
    # Stack all interpolations
    interpolation_grid = torch.cat(all_interpolations, dim=0)
    
    return interpolation_grid


def conditional_sample_from_mean(
    model: nn.Module,
    input_images: torch.Tensor,
    num_samples_per_input: int = 5
) -> torch.Tensor:
    """
    Generate samples conditioned on input images (using encoder mean).
    
    Args:
        model: VAE model
        input_images: Input images [batch, channels, height, width]
        num_samples_per_input: Number of samples per input
    
    Returns:
        Conditional samples
    """
    model.eval()
    
    with torch.no_grad():
        # Encode inputs
        mu, logvar = model.encode(input_images)
        
        # Generate multiple samples from the posterior
        all_samples = []
        for _ in range(num_samples_per_input):
            z = model.reparameterize(mu, logvar)
            samples = model.decode(z)
            
            # Apply sigmoid for BCE models
            if hasattr(model, 'decoder') and isinstance(model.decoder.deconv_layers[-1], nn.ConvTranspose2d):
                samples = torch.sigmoid(samples)
            
            all_samples.append(samples)
        
        # Stack samples: [batch * num_samples_per_input, channels, height, width]
        conditional_samples = torch.cat(all_samples, dim=0)
    
    return conditional_samples


def latent_arithmetic(
    model: nn.Module,
    concept_images: List[torch.Tensor],
    device: torch.device,
    operation: str = "add"
) -> torch.Tensor:
    """
    Perform arithmetic operations in latent space.
    
    Args:
        model: VAE model
        concept_images: List of image tensors representing concepts
        device: Computation device
        operation: "add", "subtract", "average"
    
    Returns:
        Result of latent arithmetic
    """
    model.eval()
    
    with torch.no_grad():
        # Encode all concept images
        latent_codes = []
        for img in concept_images:
            img = img.to(device)
            mu, _ = model.encode(img)
            latent_codes.append(mu.mean(dim=0, keepdim=True))  # Average if batch
        
        # Perform operation
        if operation == "add":
            result_z = sum(latent_codes)
        elif operation == "subtract" and len(latent_codes) >= 2:
            result_z = latent_codes[0] - latent_codes[1]
            if len(latent_codes) > 2:
                for z in latent_codes[2:]:
                    result_z = result_z + z
        elif operation == "average":
            result_z = torch.stack(latent_codes).mean(dim=0)
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Decode result
        result_img = model.decode(result_z)
        
        # Apply sigmoid for BCE models
        if hasattr(model, 'decoder') and isinstance(model.decoder.deconv_layers[-1], nn.ConvTranspose2d):
            result_img = torch.sigmoid(result_img)
    
    return result_img


def sample_with_fixed_noise(
    model: nn.Module,
    noise_vector: torch.Tensor,
    device: torch.device,
    variations: List[str] = ["temperature"]
) -> Dict[str, torch.Tensor]:
    """
    Generate samples with fixed noise but different variations.
    
    Args:
        model: VAE model
        noise_vector: Fixed noise vector [1, latent_dim]
        variations: List of variations to apply
        device: Computation device
    
    Returns:
        Dictionary mapping variation names to generated images
    """
    model.eval()
    results = {}
    
    with torch.no_grad():
        # Base sample
        base_sample = model.decode(noise_vector.to(device))
        if hasattr(model, 'decoder') and isinstance(model.decoder.deconv_layers[-1], nn.ConvTranspose2d):
            base_sample = torch.sigmoid(base_sample)
        results["base"] = base_sample
        
        for variation in variations:
            if variation == "temperature":
                # Different temperature scaling
                temperatures = [0.5, 0.8, 1.2, 1.5]
                temp_samples = []
                
                for temp in temperatures:
                    scaled_z = noise_vector * temp
                    sample = model.decode(scaled_z.to(device))
                    if hasattr(model, 'decoder') and isinstance(model.decoder.deconv_layers[-1], nn.ConvTranspose2d):
                        sample = torch.sigmoid(sample)
                    temp_samples.append(sample)
                
                results["temperature"] = torch.cat(temp_samples, dim=0)
            
            elif variation == "interpolation":
                # Interpolate with random vector
                random_z = torch.randn_like(noise_vector)
                alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
                interp_samples = []
                
                for alpha in alphas:
                    interp_z = (1 - alpha) * noise_vector + alpha * random_z
                    sample = model.decode(interp_z.to(device))
                    if hasattr(model, 'decoder') and isinstance(model.decoder.deconv_layers[-1], nn.ConvTranspose2d):
                        sample = torch.sigmoid(sample)
                    interp_samples.append(sample)
                
                results["interpolation"] = torch.cat(interp_samples, dim=0)
    
    return results


def run_sampling(config: DictConfig, checkpoint_path: str, output_dir: str) -> None:
    """
    Run comprehensive sampling from trained VAE.
    
    Args:
        config: Configuration
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save samples
    """
    device = get_device(config.device)
    logger = setup_logger("VAE_Sampling")
    
    # Create output directories
    samples_dir = os.path.join(output_dir, "samples")
    grids_dir = os.path.join(output_dir, "grids")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)
    
    logger.info(f"Loading model from: {checkpoint_path}")
    
    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path, device)
    
    # Generate prior samples
    logger.info("Generating samples from prior...")
    prior_samples = sample_from_prior(model, 64, device, temperature=1.0)
    
    # Save prior samples grid
    save_image_grid(
        prior_samples,
        os.path.join(grids_dir, "prior_samples.png"),
        nrow=8,
        title="Samples from Prior Distribution"
    )
    
    # Generate samples with different temperatures
    logger.info("Generating samples with different temperatures...")
    temp_samples = []
    temperatures = [0.5, 0.8, 1.0, 1.2, 1.5]
    
    for temp in temperatures:
        temp_batch = sample_from_prior(model, 8, device, temperature=temp)
        temp_samples.append(temp_batch)
    
    all_temp_samples = torch.cat(temp_samples, dim=0)
    save_image_grid(
        all_temp_samples,
        os.path.join(grids_dir, "temperature_samples.png"),
        nrow=8,
        title="Samples at Different Temperatures"
    )
    
    # Generate interpolations
    logger.info("Creating interpolation grids...")
    sample_images = get_sample_images(
        config.data.dataset,
        config.data.root,
        num_samples=64,
        normalize=config.data.normalize
    )
    
    interpolation_grid = create_interpolation_grid(
        model, sample_images, device, num_pairs=8, num_steps=10
    )
    
    save_image_grid(
        interpolation_grid,
        os.path.join(grids_dir, "interpolations.png"),
        nrow=10,
        title="Latent Space Interpolations"
    )
    
    # Generate conditional samples (reconstruction variations)
    logger.info("Generating conditional samples...")
    test_images = sample_images[:8].to(device)  # First 8 images
    conditional_samples = conditional_sample_from_mean(model, test_images, num_samples_per_input=5)
    
    # Combine original and conditional samples for comparison
    comparison_grid = []
    for i in range(8):
        # Original image
        comparison_grid.append(test_images[i:i+1])
        # Conditional samples for this image
        start_idx = i * 5
        end_idx = start_idx + 5
        comparison_grid.append(conditional_samples[start_idx:end_idx])
    
    comparison_samples = torch.cat(comparison_grid, dim=0)
    save_image_grid(
        comparison_samples,
        os.path.join(grids_dir, "conditional_samples.png"),
        nrow=6,  # 1 original + 5 samples
        title="Conditional Samples (Original + Variations)"
    )
    
    # Save individual sample batches for further analysis
    logger.info("Saving individual sample batches...")
    
    # Save different sample types
    sample_types = {
        "prior_temp_0.5": sample_from_prior(model, 64, device, 0.5),
        "prior_temp_1.0": sample_from_prior(model, 64, device, 1.0),
        "prior_temp_1.5": sample_from_prior(model, 64, device, 1.5),
    }
    
    for sample_type, samples in sample_types.items():
        save_image_grid(
            samples,
            os.path.join(samples_dir, f"{sample_type}.png"),
            nrow=8,
            title=f"Samples: {sample_type}"
        )
    
    logger.info(f"Sampling completed. Results saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    
    if len(sys.argv) != 4:
        print("Usage: python -m src.sample <config_path> <checkpoint_path> <output_dir>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    checkpoint_path = sys.argv[2]
    output_dir = sys.argv[3]
    
    config = OmegaConf.load(config_path)
    
    run_sampling(config, checkpoint_path, output_dir)