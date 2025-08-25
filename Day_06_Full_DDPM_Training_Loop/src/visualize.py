"""
Visualization utilities for DDPM: sample grids, trajectories, training curves, schedules
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageFont
import io

from .sampler import DDPMSampler
from .utils import tensor_to_pil, save_image_grid
from .ddpm_schedules import DDPMSchedules


def setup_matplotlib():
    """Setup matplotlib with nice defaults"""
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    sns.set_palette("husl")


def make_sample_grid(
    samples: torch.Tensor,
    save_path: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    captions: Optional[List[str]] = None
) -> str:
    """
    Create and save a grid of sample images
    
    Args:
        samples: tensor of images [B, C, H, W]
        save_path: path to save the grid
        nrow: number of images per row
        normalize: whether to normalize images
        value_range: expected value range for normalization
        title: optional title for the grid
        captions: optional captions for each image
        
    Returns:
        path to saved image
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if value_range is None:
        value_range = (-1, 1) if samples.min() < 0 else (0, 1)
        
    # Save the basic grid
    save_image_grid(samples, save_path, nrow=nrow, normalize=normalize, value_range=value_range)
    
    # If title or captions provided, create enhanced version
    if title is not None or captions is not None:
        # Load the saved grid
        grid_image = Image.open(save_path)
        
        # Create new image with space for title/captions
        title_height = 50 if title else 0
        caption_height = 30 if captions else 0
        
        new_height = grid_image.height + title_height + caption_height
        enhanced_image = Image.new('RGB', (grid_image.width, new_height), 'white')
        
        # Add title
        if title:
            draw = ImageDraw.Draw(enhanced_image)
            try:
                font = ImageFont.load_default()
            except:
                font = None
            draw.text((10, 10), title, fill='black', font=font)
            enhanced_image.paste(grid_image, (0, title_height))
        else:
            enhanced_image.paste(grid_image, (0, 0))
            
        enhanced_image.save(save_path)
        
    return save_path


def reverse_trajectory_grid(
    model: nn.Module,
    sampler: DDPMSampler,
    shape: Tuple[int, ...],
    save_path: str,
    timesteps: Optional[List[int]] = None,
    device: str = 'cpu',
    method: str = 'ddim',
    num_steps: int = 50
) -> str:
    """
    Create grid showing reverse sampling trajectory
    
    Args:
        model: trained DDPM model
        sampler: DDPM sampler
        shape: shape for single sample (C, H, W)
        save_path: path to save trajectory grid
        timesteps: specific timesteps to visualize
        device: device to run on
        method: sampling method
        num_steps: number of sampling steps
        
    Returns:
        path to saved trajectory grid
    """
    model.eval()
    
    # Generate single sample with full trajectory
    sample_shape = (1, *shape)
    
    with torch.no_grad():
        final_sample, trajectory = sampler.sample(
            model=model,
            shape=sample_shape,
            method=method,
            num_steps=num_steps if method == 'ddim' else None,
            device=device,
            return_trajectory=True,
            progress=True
        )
        
    # Select timesteps to visualize
    if timesteps is None:
        # Select evenly spaced timesteps
        total_steps = len(trajectory)
        timesteps = np.linspace(0, total_steps - 1, min(10, total_steps)).astype(int)
        
    # Create trajectory tensor
    selected_samples = []
    step_labels = []
    
    for i, step_idx in enumerate(timesteps):
        if step_idx < len(trajectory):
            selected_samples.append(trajectory[step_idx])
            
            # Calculate actual timestep for DDIM
            if method == 'ddim':
                actual_t = int((len(trajectory) - 1 - step_idx) * (sampler.schedules.num_timesteps / len(trajectory)))
            else:
                actual_t = len(trajectory) - 1 - step_idx
                
            step_labels.append(f"t={actual_t}")
            
    trajectory_tensor = torch.cat(selected_samples, dim=0)
    
    # Save grid with labels
    grid_path = make_sample_grid(
        samples=trajectory_tensor,
        save_path=save_path,
        nrow=len(selected_samples),
        title="Reverse Diffusion Trajectory",
        captions=step_labels
    )
    
    return grid_path


def make_animation(
    model: nn.Module,
    sampler: DDPMSampler,
    shape: Tuple[int, ...],
    save_path: str,
    device: str = 'cpu',
    method: str = 'ddim',
    num_steps: int = 50,
    fps: int = 10,
    format: str = 'gif'
) -> str:
    """
    Create animation of reverse sampling process
    
    Args:
        model: trained DDPM model  
        sampler: DDPM sampler
        shape: shape for single sample
        save_path: path to save animation
        device: device to run on
        method: sampling method
        num_steps: number of sampling steps
        fps: frames per second
        format: 'gif' or 'mp4'
        
    Returns:
        path to saved animation
    """
    model.eval()
    
    # Generate trajectory
    sample_shape = (1, *shape)
    
    with torch.no_grad():
        final_sample, trajectory = sampler.sample(
            model=model,
            shape=sample_shape,
            method=method,
            num_steps=num_steps if method == 'ddim' else None,
            device=device,
            return_trajectory=True,
            progress=True
        )
        
    # Convert trajectory to PIL images
    frames = []
    for sample in trajectory:
        pil_image = tensor_to_pil(sample[0])  # Remove batch dimension
        frames.append(pil_image)
        
    # Save animation
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'gif':
        frames[0].save(
            save_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,  # Duration in milliseconds
            loop=0
        )
    elif format.lower() == 'mp4':
        # For MP4, we'd need additional dependencies like imageio-ffmpeg
        # For now, save as GIF with warning
        gif_path = save_path.replace('.mp4', '.gif')
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=1000 // fps,
            loop=0
        )
        print(f"Warning: MP4 not implemented, saved as GIF: {gif_path}")
        return gif_path
        
    return save_path


def plot_training_curves(
    metrics_file: str,
    save_dir: str,
    metrics_to_plot: Optional[List[str]] = None
) -> Dict[str, str]:
    """
    Plot training curves from metrics file
    
    Args:
        metrics_file: path to JSON file with training metrics
        save_dir: directory to save plots
        metrics_to_plot: specific metrics to plot
        
    Returns:
        dict of plot_name -> file_path
    """
    setup_matplotlib()
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        data = json.load(f)
        
    train_metrics = data.get('train_metrics', [])
    val_metrics = data.get('val_metrics', [])
    
    if not train_metrics and not val_metrics:
        print("No metrics found in file")
        return {}
        
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plots_saved = {}
    
    # Default metrics to plot
    if metrics_to_plot is None:
        all_metrics = set()
        for metrics in train_metrics + val_metrics:
            all_metrics.update(metrics.keys())
        
        metrics_to_plot = [m for m in all_metrics 
                          if m not in ['step', 'epoch', 'batch_size', 'step_time', 'epoch_time']]
    
    # Plot each metric
    for metric in metrics_to_plot:
        plt.figure(figsize=(10, 6))
        
        # Extract metric values
        train_values = [m.get(metric) for m in train_metrics if metric in m]
        val_values = [m.get(metric) for m in val_metrics if metric in m]
        
        if train_values:
            epochs = range(len(train_values))
            plt.plot(epochs, train_values, label=f'Train {metric}', marker='o')
            
        if val_values:
            epochs = range(len(val_values))
            plt.plot(epochs, val_values, label=f'Val {metric}', marker='s')
            
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(f'Training Curve: {metric}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_path = save_dir / f'{metric}_curve.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        plots_saved[metric] = str(plot_path)
        
    return plots_saved


def plot_schedule_comparison(
    schedules: DDPMSchedules,
    save_path: str,
    schedule_types: Optional[List[str]] = None
) -> str:
    """
    Plot noise schedules (β, α, ᾱ, SNR)
    
    Args:
        schedules: DDPM schedules object
        save_path: path to save plot
        schedule_types: types of schedules to compare
        
    Returns:
        path to saved plot
    """
    setup_matplotlib()
    
    timesteps = np.arange(schedules.num_timesteps)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('DDPM Noise Schedules', fontsize=16)
    
    # Beta schedule
    ax = axes[0, 0]
    ax.plot(timesteps, schedules.betas.cpu().numpy())
    ax.set_title('Beta Schedule (β_t)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('β_t')
    ax.grid(True, alpha=0.3)
    
    # Alpha schedule  
    ax = axes[0, 1]
    ax.plot(timesteps, schedules.alphas.cpu().numpy())
    ax.set_title('Alpha Schedule (α_t)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('α_t')
    ax.grid(True, alpha=0.3)
    
    # Alpha cumprod schedule
    ax = axes[1, 0]
    ax.plot(timesteps, schedules.alphas_cumprod.cpu().numpy())
    ax.set_title('Alpha Cumulative Product (ᾱ_t)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('ᾱ_t')
    ax.grid(True, alpha=0.3)
    
    # SNR
    ax = axes[1, 1]
    snr = schedules.get_snr().cpu().numpy()
    ax.semilogy(timesteps, snr)
    ax.set_title('Signal-to-Noise Ratio')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('SNR (log scale)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def plot_loss_landscape(
    model: nn.Module,
    schedules: DDPMSchedules, 
    test_batch: torch.Tensor,
    save_path: str,
    num_timesteps: int = 20
) -> str:
    """
    Plot loss landscape across timesteps
    
    Args:
        model: trained DDPM model
        schedules: noise schedules
        test_batch: batch of test images
        save_path: path to save plot
        num_timesteps: number of timesteps to evaluate
        
    Returns:
        path to saved plot
    """
    setup_matplotlib()
    
    model.eval()
    device = next(model.parameters()).device
    test_batch = test_batch.to(device)
    
    timesteps = torch.linspace(0, schedules.num_timesteps - 1, num_timesteps).long()
    losses = []
    
    with torch.no_grad():
        for t in timesteps:
            batch_t = torch.full((len(test_batch),), t, device=device)
            
            # Sample noise and create noisy images
            noise = torch.randn_like(test_batch)
            x_t = schedules.q_sample(test_batch, batch_t, noise)
            
            # Predict noise
            predicted_noise = model(x_t, batch_t)
            
            # Compute loss
            loss = torch.nn.functional.mse_loss(predicted_noise, noise, reduction='mean')
            losses.append(loss.item())
            
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps.cpu().numpy(), losses, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Timestep')
    plt.ylabel('MSE Loss')
    plt.title('Loss Landscape Across Timesteps')
    plt.grid(True, alpha=0.3)
    
    # Save plot
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def create_comparison_grid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    save_path: str,
    num_pairs: int = 8
) -> str:
    """
    Create side-by-side comparison grid of real vs generated images
    
    Args:
        real_images: tensor of real images
        generated_images: tensor of generated images  
        save_path: path to save comparison
        num_pairs: number of pairs to show
        
    Returns:
        path to saved comparison
    """
    # Select random pairs
    indices = torch.randperm(min(len(real_images), len(generated_images)))[:num_pairs]
    
    real_selected = real_images[indices]
    generated_selected = generated_images[indices]
    
    # Interleave real and generated
    comparison_images = []
    for i in range(num_pairs):
        comparison_images.append(real_selected[i])
        comparison_images.append(generated_selected[i])
        
    comparison_tensor = torch.stack(comparison_images)
    
    # Create grid with 2 images per row (real, generated pairs)
    return make_sample_grid(
        samples=comparison_tensor,
        save_path=save_path,
        nrow=2,
        title="Real (left) vs Generated (right) Comparison"
    )


def plot_evaluation_results(
    results: Dict[str, Any],
    save_dir: str
) -> Dict[str, str]:
    """
    Plot evaluation results from comprehensive evaluation
    
    Args:
        results: evaluation results dict
        save_dir: directory to save plots
        
    Returns:
        dict of plot_name -> file_path
    """
    setup_matplotlib()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    plots_saved = {}
    
    # Metrics comparison across methods
    metrics_data = {}
    method_names = []
    
    for key, values in results.items():
        if isinstance(values, dict) and 'error' not in values:
            method_names.append(key)
            for metric, value in values.items():
                if isinstance(value, (int, float)):
                    if metric not in metrics_data:
                        metrics_data[metric] = []
                    metrics_data[metric].append(value)
                    
    # Plot metrics comparison
    if metrics_data and method_names:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Evaluation Metrics Comparison', fontsize=16)
        axes = axes.flatten()
        
        for i, (metric, values) in enumerate(list(metrics_data.items())[:4]):
            if i < len(axes):
                ax = axes[i]
                ax.bar(method_names, values)
                ax.set_title(metric)
                ax.set_ylabel(metric)
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
        plt.tight_layout()
        plot_path = save_dir / 'metrics_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        plots_saved['metrics_comparison'] = str(plot_path)
        
    # Calibration curve
    if 'calibration' in results:
        cal_data = results['calibration']
        if 'timesteps' in cal_data and 'psnr' in cal_data:
            plt.figure(figsize=(10, 6))
            plt.plot(cal_data['timesteps'], cal_data['psnr'], marker='o')
            plt.xlabel('Timestep')
            plt.ylabel('PSNR (dB)')
            plt.title('Reconstruction Quality vs Timestep')
            plt.grid(True, alpha=0.3)
            
            plot_path = save_dir / 'calibration_curve.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            plots_saved['calibration'] = str(plot_path)
            
    return plots_saved


class VisualizationManager:
    """
    Manager class for all visualization tasks
    """
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
        self.grids_dir = self.output_dir / "grids"
        self.curves_dir = self.output_dir / "curves"
        self.animations_dir = self.output_dir / "animations"
        
        # Create directories
        for directory in [self.grids_dir, self.curves_dir, self.animations_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def save_sample_grid(
        self,
        samples: torch.Tensor,
        filename: str,
        **kwargs
    ) -> str:
        """Save sample grid to grids directory"""
        save_path = self.grids_dir / filename
        return make_sample_grid(samples, str(save_path), **kwargs)
        
    def save_trajectory(
        self,
        model: nn.Module,
        sampler: DDPMSampler,
        shape: Tuple[int, ...],
        filename: str,
        **kwargs
    ) -> str:
        """Save trajectory grid to grids directory"""
        save_path = self.grids_dir / filename
        return reverse_trajectory_grid(model, sampler, shape, str(save_path), **kwargs)
        
    def save_animation(
        self,
        model: nn.Module,
        sampler: DDPMSampler,
        shape: Tuple[int, ...],
        filename: str,
        **kwargs
    ) -> str:
        """Save animation to animations directory"""
        save_path = self.animations_dir / filename
        return make_animation(model, sampler, shape, str(save_path), **kwargs)
        
    def save_training_curves(
        self,
        metrics_file: str,
        **kwargs
    ) -> Dict[str, str]:
        """Save training curves to curves directory"""
        return plot_training_curves(metrics_file, str(self.curves_dir), **kwargs)
        
    def save_schedule_plots(
        self,
        schedules: DDPMSchedules,
        filename: str = "schedules.png"
    ) -> str:
        """Save schedule plots to curves directory"""
        save_path = self.curves_dir / filename
        return plot_schedule_comparison(schedules, str(save_path))