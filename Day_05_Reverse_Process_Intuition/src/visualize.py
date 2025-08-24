"""Visualization tools for DDPM process

Implements:
- Reverse trajectory visualization (single sample through timesteps)
- Forward vs reverse comparison panels
- Animation generation (GIF/MP4)
- Grid visualization of multiple samples
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Optional, Tuple, Dict, Any
import os
from PIL import Image
import imageio

from .ddpm_schedules import DDPMScheduler
from .sampler import DDPMSampler, DDIMSampler
from .utils import save_image_grid


def tensor_to_numpy(tensor: torch.Tensor, normalize: bool = True) -> np.ndarray:
    """Convert tensor to numpy array for visualization.
    
    Args:
        tensor: Input tensor [C, H, W] or [B, C, H, W]
        normalize: Whether to normalize to [0, 1]
    
    Returns:
        Numpy array ready for visualization
    """
    if tensor.dim() == 4:
        tensor = tensor[0]  # Take first image from batch
    
    # Move to CPU and convert to numpy
    img = tensor.detach().cpu().numpy()
    
    # Handle channel dimension
    if img.shape[0] == 1:  # Grayscale
        img = img.squeeze(0)
    elif img.shape[0] == 3:  # RGB
        img = np.transpose(img, (1, 2, 0))
    
    # Normalize if needed
    if normalize:
        img = np.clip(img, 0, 1)
    
    return img


def create_reverse_trajectory_grid(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    num_samples: int = 4,
    num_timesteps_to_show: int = 8,
    image_size: Tuple[int, int] = (32, 32),
    channels: int = 3,
    device: torch.device = None,
    sampler_type: str = "ddpm",
    save_path: Optional[str] = None
) -> np.ndarray:
    """Create grid showing reverse trajectory for multiple samples.
    
    Args:
        model: Trained denoising model
        scheduler: DDPM scheduler
        num_samples: Number of different samples to show
        num_timesteps_to_show: Number of timesteps to visualize
        image_size: Size of images
        channels: Number of channels
        device: Device to use
        sampler_type: Type of sampler ("ddpm" or "ddim")
        save_path: Path to save the grid
    
    Returns:
        Visualization grid as numpy array
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    with torch.no_grad():
        # Create sampler
        if sampler_type == "ddpm":
            sampler = DDPMSampler(scheduler)
        elif sampler_type == "ddim":
            sampler = DDIMSampler(scheduler)
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")
        
        # Generate trajectories
        trajectories = []
        
        for _ in range(num_samples):
            # Generate single trajectory
            if sampler_type == "ddpm":
                result = sampler.p_sample_loop(
                    model, (1, channels, *image_size),
                    device=device, return_trajectory=True, progress=False
                )
            else:  # DDIM
                result = sampler.ddim_sample(
                    model, (1, channels, *image_size),
                    device=device, return_trajectory=True, progress=False
                )
            
            trajectory = result["trajectory"]
            
            # Select timesteps to show
            timestep_indices = np.linspace(0, len(trajectory) - 1, num_timesteps_to_show, dtype=int)
            selected_frames = [trajectory[i] for i in timestep_indices]
            
            trajectories.append(selected_frames)
    
    # Create grid
    fig, axes = plt.subplots(num_samples, num_timesteps_to_show, figsize=(num_timesteps_to_show * 2, num_samples * 2))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_timesteps_to_show == 1:
        axes = axes.reshape(-1, 1)
    
    for i, trajectory in enumerate(trajectories):
        for j, frame in enumerate(trajectory):
            img = tensor_to_numpy(frame, normalize=True)
            
            if channels == 1:
                axes[i, j].imshow(img, cmap='gray')
            else:
                axes[i, j].imshow(img)
            
            axes[i, j].axis('off')
            
            # Add timestep labels on top row
            if i == 0:
                timestep = int((len(trajectory) - 1 - j) * scheduler.num_timesteps / len(trajectory))
                axes[i, j].set_title(f't={timestep}', fontsize=10)
    
    plt.suptitle(f'Reverse Trajectory ({sampler_type.upper()})', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Remove alpha channel
    
    plt.close(fig)
    
    return buf


def create_forward_vs_reverse_panel(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    x_start: torch.Tensor,
    timesteps_to_show: List[int] = [10, 25, 50, 75],
    device: torch.device = None,
    save_path: Optional[str] = None
) -> np.ndarray:
    """Create panel comparing forward and reverse processes.
    
    Args:
        model: Trained denoising model
        scheduler: DDPM scheduler  
        x_start: Clean images to start from [B, C, H, W]
        timesteps_to_show: Timesteps to visualize
        device: Device to use
        save_path: Path to save the panel
    
    Returns:
        Visualization panel as numpy array
    """
    if device is None:
        device = next(model.parameters()).device
    
    x_start = x_start.to(device)
    model.eval()
    
    batch_size = x_start.shape[0]
    num_images = min(4, batch_size)  # Show up to 4 images
    num_timesteps = len(timesteps_to_show)
    
    with torch.no_grad():
        # Create forward process samples
        forward_samples = []
        reverse_samples = []
        
        for t in timesteps_to_show:
            # Forward: add noise to clean images
            timesteps_tensor = torch.full((num_images,), t, device=device, dtype=torch.long)
            noise = torch.randn_like(x_start[:num_images])
            x_t_forward = scheduler.add_noise(x_start[:num_images], noise, timesteps_tensor)
            forward_samples.append(x_t_forward)
            
            # Reverse: start from this noisy state and denoise
            sampler = DDPMSampler(scheduler)
            
            # Create a custom sampling that starts from x_t and goes to x_0
            x_t = x_t_forward.clone()
            
            # Sample from t to 0
            for step_t in range(t, -1, -1):
                if step_t > 0:
                    step_t_tensor = torch.full((num_images,), step_t, device=device, dtype=torch.long)
                    step_output = sampler.p_sample_step(model, x_t, step_t_tensor)
                    x_t = step_output["x_prev"]
                else:
                    break
            
            reverse_samples.append(x_t)
    
    # Create visualization
    fig, axes = plt.subplots(3, num_timesteps + 1, figsize=((num_timesteps + 1) * 2, 6))
    
    # Show original images in first column
    for i in range(num_images):
        if i < 3:  # Show up to 3 rows
            img = tensor_to_numpy(x_start[i], normalize=True)
            if x_start.shape[1] == 1:
                axes[i, 0].imshow(img, cmap='gray')
            else:
                axes[i, 0].imshow(img)
            axes[i, 0].axis('off')
            if i == 0:
                axes[i, 0].set_title('Original', fontsize=10)
    
    # Show forward and reverse processes
    for j, t in enumerate(timesteps_to_show):
        col = j + 1
        
        # Forward process (row 0)
        img_forward = tensor_to_numpy(forward_samples[j][0], normalize=True)
        if x_start.shape[1] == 1:
            axes[0, col].imshow(img_forward, cmap='gray')
        else:
            axes[0, col].imshow(img_forward)
        axes[0, col].axis('off')
        if j == 0:
            axes[0, col].set_ylabel('Forward\n(+noise)', rotation=0, ha='right', va='center')
        axes[0, col].set_title(f't={t}', fontsize=10)
        
        # Reverse process (row 1)
        img_reverse = tensor_to_numpy(reverse_samples[j][0], normalize=True)
        if x_start.shape[1] == 1:
            axes[1, col].imshow(img_reverse, cmap='gray')
        else:
            axes[1, col].imshow(img_reverse)
        axes[1, col].axis('off')
        if j == 0:
            axes[1, col].set_ylabel('Reverse\n(-noise)', rotation=0, ha='right', va='center')
        
        # Difference (row 2)
        if x_start.shape[1] == 1:
            diff = np.abs(tensor_to_numpy(x_start[0], normalize=True) - img_reverse)
            axes[2, col].imshow(diff, cmap='hot', vmin=0, vmax=1)
        else:
            diff = np.abs(tensor_to_numpy(x_start[0], normalize=True) - img_reverse)
            axes[2, col].imshow(diff)
        axes[2, col].axis('off')
        if j == 0:
            axes[2, col].set_ylabel('|Diff|', rotation=0, ha='right', va='center')
    
    plt.suptitle('Forward vs Reverse Process Comparison', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    # Convert to numpy array
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    buf = buf[:, :, :3]  # Remove alpha channel
    
    plt.close(fig)
    
    return buf


def create_reverse_animation(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    num_samples: int = 1,
    image_size: Tuple[int, int] = (32, 32),
    channels: int = 3,
    device: torch.device = None,
    save_path: Optional[str] = None,
    fps: int = 10,
    sampler_type: str = "ddpm"
) -> List[np.ndarray]:
    """Create animation of reverse diffusion process.
    
    Args:
        model: Trained denoising model
        scheduler: DDPM scheduler
        num_samples: Number of samples to animate (arranged in grid)
        image_size: Size of each image
        channels: Number of channels
        device: Device to use
        save_path: Path to save animation
        fps: Frames per second
        sampler_type: Type of sampler
    
    Returns:
        List of frames as numpy arrays
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    with torch.no_grad():
        # Create sampler
        if sampler_type == "ddpm":
            sampler = DDPMSampler(scheduler)
        elif sampler_type == "ddim":
            sampler = DDIMSampler(scheduler)
        else:
            raise ValueError(f"Unknown sampler_type: {sampler_type}")
        
        # Generate trajectory
        if sampler_type == "ddpm":
            result = sampler.p_sample_loop(
                model, (num_samples, channels, *image_size),
                device=device, return_trajectory=True, progress=False
            )
        else:  # DDIM
            result = sampler.ddim_sample(
                model, (num_samples, channels, *image_size),
                num_inference_steps=50,  # Fewer steps for animation
                device=device, return_trajectory=True, progress=False
            )
        
        trajectory = result["trajectory"]
    
    # Convert trajectory to frames
    frames = []
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    for step_images in trajectory:
        # Create grid for this timestep
        grid_img = create_image_grid(step_images, nrow=grid_size)
        frames.append(grid_img)
    
    # Reverse frames to show T -> 0
    frames = frames[::-1]
    
    # Save animation
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if save_path.endswith('.gif'):
            imageio.mimsave(save_path, frames, fps=fps, loop=0)
        elif save_path.endswith('.mp4'):
            imageio.mimsave(save_path, frames, fps=fps, quality=8)
        else:
            # Default to GIF
            save_path = save_path + '.gif'
            imageio.mimsave(save_path, frames, fps=fps, loop=0)
    
    return frames


def create_image_grid(images: torch.Tensor, nrow: int = 8) -> np.ndarray:
    """Create a grid of images for visualization.
    
    Args:
        images: Tensor of images [B, C, H, W]
        nrow: Number of images per row
    
    Returns:
        Grid image as numpy array
    """
    from torchvision.utils import make_grid
    
    grid = make_grid(images, nrow=nrow, normalize=True, pad_value=1.0)
    grid_np = tensor_to_numpy(grid, normalize=False)
    
    # Convert to uint8
    grid_np = (grid_np * 255).astype(np.uint8)
    
    return grid_np


def visualize_noise_schedule(scheduler: DDPMScheduler, save_path: Optional[str] = None):
    """Visualize the noise schedule (betas, alphas, etc.).
    
    Args:
        scheduler: DDPM scheduler
        save_path: Path to save the plot
    """
    timesteps = np.arange(scheduler.num_timesteps)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Beta schedule
    axes[0, 0].plot(timesteps, scheduler.betas.cpu().numpy())
    axes[0, 0].set_title('Beta Schedule')
    axes[0, 0].set_xlabel('Timestep')
    axes[0, 0].set_ylabel('Beta')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Alpha schedule
    axes[0, 1].plot(timesteps, scheduler.alphas.cpu().numpy())
    axes[0, 1].set_title('Alpha Schedule')
    axes[0, 1].set_xlabel('Timestep')
    axes[0, 1].set_ylabel('Alpha')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Alpha bar (cumulative product)
    axes[1, 0].plot(timesteps, scheduler.alpha_bar.cpu().numpy())
    axes[1, 0].set_title('Alpha Bar Schedule')
    axes[1, 0].set_xlabel('Timestep')
    axes[1, 0].set_ylabel('Alpha Bar')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Signal-to-Noise Ratio
    snr = scheduler.alpha_bar / (1 - scheduler.alpha_bar)
    axes[1, 1].semilogy(timesteps, snr.cpu().numpy())
    axes[1, 1].set_title('Signal-to-Noise Ratio')
    axes[1, 1].set_xlabel('Timestep')
    axes[1, 1].set_ylabel('SNR (log scale)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def test_visualizations():
    """Test visualization functions."""
    from .ddpm_schedules import DDPMScheduler
    from .models.unet_tiny import UNetTiny
    
    # Create components
    scheduler = DDPMScheduler(num_timesteps=100)
    model = UNetTiny(in_channels=3, out_channels=3, model_channels=32)
    
    # Test noise schedule visualization
    visualize_noise_schedule(scheduler, "test_schedule.png")
    print("Created noise schedule plot")
    
    # Test trajectory grid
    grid = create_reverse_trajectory_grid(
        model, scheduler, num_samples=2, num_timesteps_to_show=4,
        channels=3, save_path="test_trajectory.png"
    )
    print(f"Created trajectory grid: {grid.shape}")
    
    # Test forward vs reverse
    x_start = torch.randn(2, 3, 32, 32)
    panel = create_forward_vs_reverse_panel(
        model, scheduler, x_start, 
        timesteps_to_show=[10, 25, 50],
        save_path="test_comparison.png"
    )
    print(f"Created comparison panel: {panel.shape}")
    
    print("Visualization tests completed!")


if __name__ == "__main__":
    test_visualizations()