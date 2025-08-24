"""Compare forward and reverse processes

This module compares the forward diffusion process (Day 4) with the 
reverse process (Day 5) to demonstrate their relationship and evaluate
how well the learned reverse process matches the theoretical forward process.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
import os
from datetime import datetime

from .ddpm_schedules import DDPMScheduler
from .sampler import DDPMSampler, reconstruct_from_noise
from .visualize import create_forward_vs_reverse_panel, tensor_to_numpy
from .eval import compute_psnr_batch, compute_ssim_batch


def compare_forward_reverse_trajectories(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    x_start: torch.Tensor,
    num_timesteps_to_show: int = 10,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Compare forward and reverse trajectories for the same starting image.
    
    Args:
        model: Trained denoising model
        scheduler: DDPM scheduler
        x_start: Clean starting images [B, C, H, W]
        num_timesteps_to_show: Number of timesteps to analyze
        device: Device to use
        save_path: Path to save comparison plot
    
    Returns:
        Dictionary with comparison results and metrics
    """
    if device is None:
        device = next(model.parameters()).device
    
    x_start = x_start.to(device)
    model.eval()
    
    batch_size = x_start.shape[0]
    timesteps_to_eval = np.linspace(0, scheduler.num_timesteps - 1, num_timesteps_to_show, dtype=int)
    
    forward_trajectory = []
    reverse_trajectory = []
    reconstruction_errors = []
    
    with torch.no_grad():
        # Generate forward trajectory
        for t in timesteps_to_eval:
            # Forward process: add noise
            timesteps_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            noise = torch.randn_like(x_start)
            x_t_forward = scheduler.add_noise(x_start, noise, timesteps_tensor)
            forward_trajectory.append(x_t_forward.cpu())
            
            # Reverse process: start from this noisy state
            if t > 0:
                sampler = DDPMSampler(scheduler)
                recon_result = reconstruct_from_noise(
                    model, scheduler, x_start, t_start=t,
                    sampler_type="ddpm", progress=False
                )
                x_recon = recon_result["images"]
                reverse_trajectory.append(x_recon.cpu())
                
                # Compute reconstruction error
                psnr_values = compute_psnr_batch(x_start, x_recon)
                ssim_values = compute_ssim_batch(x_start, x_recon)
                reconstruction_errors.append({
                    "timestep": t,
                    "psnr": np.mean(psnr_values),
                    "ssim": np.mean(ssim_values)
                })
            else:
                # At t=0, reverse trajectory is just the original
                reverse_trajectory.append(x_start.cpu())
                reconstruction_errors.append({
                    "timestep": t,
                    "psnr": float('inf'),  # Perfect reconstruction
                    "ssim": 1.0
                })
    
    # Create visualization
    if save_path:
        fig, axes = plt.subplots(3, num_timesteps_to_show, figsize=(num_timesteps_to_show * 2, 6))
        
        for i, t in enumerate(timesteps_to_eval):
            # Forward trajectory (top row)
            img_forward = tensor_to_numpy(forward_trajectory[i][0])
            if x_start.shape[1] == 1:
                axes[0, i].imshow(img_forward, cmap='gray')
            else:
                axes[0, i].imshow(img_forward)
            axes[0, i].set_title(f't={t}', fontsize=10)
            axes[0, i].axis('off')
            
            # Reverse trajectory (middle row)
            img_reverse = tensor_to_numpy(reverse_trajectory[i][0])
            if x_start.shape[1] == 1:
                axes[1, i].imshow(img_reverse, cmap='gray')
            else:
                axes[1, i].imshow(img_reverse)
            axes[1, i].axis('off')
            
            # Error visualization (bottom row)
            if t > 0:
                error_img = np.abs(img_forward - img_reverse)
                axes[2, i].imshow(error_img, cmap='hot')
            else:
                axes[2, i].imshow(np.zeros_like(img_forward), cmap='hot')
            axes[2, i].axis('off')
        
        # Add row labels
        axes[0, 0].set_ylabel('Forward\n(+noise)', rotation=0, ha='right', va='center')
        axes[1, 0].set_ylabel('Reverse\n(-noise)', rotation=0, ha='right', va='center')
        axes[2, 0].set_ylabel('|Error|', rotation=0, ha='right', va='center')
        
        plt.suptitle('Forward vs Reverse Trajectory Comparison', fontsize=14)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return {
        "timesteps": timesteps_to_eval.tolist(),
        "forward_trajectory": forward_trajectory,
        "reverse_trajectory": reverse_trajectory,
        "reconstruction_errors": reconstruction_errors,
        "summary": {
            "avg_psnr": np.mean([err["psnr"] for err in reconstruction_errors if err["psnr"] != float('inf')]),
            "avg_ssim": np.mean([err["ssim"] for err in reconstruction_errors]),
            "min_psnr": np.min([err["psnr"] for err in reconstruction_errors if err["psnr"] != float('inf')]),
            "max_ssim": np.max([err["ssim"] for err in reconstruction_errors])
        }
    }


def analyze_schedule_sensitivity(
    model: torch.nn.Module,
    x_start: torch.Tensor,
    schedule_configs: List[Dict[str, Any]],
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze how different noise schedules affect the forward-reverse process.
    
    Args:
        model: Trained denoising model
        x_start: Clean starting images [B, C, H, W]
        schedule_configs: List of schedule configurations to test
        device: Device to use
        save_path: Path to save analysis plot
    
    Returns:
        Dictionary with sensitivity analysis results
    """
    if device is None:
        device = next(model.parameters()).device
    
    x_start = x_start.to(device)
    model.eval()
    
    results = {}
    
    for config in schedule_configs:
        schedule_name = config.get("name", "unnamed")
        
        # Create scheduler with this configuration
        scheduler = DDPMScheduler(
            num_timesteps=config.get("num_timesteps", 100),
            beta_schedule=config.get("beta_schedule", "linear"),
            beta_start=config.get("beta_start", 0.0001),
            beta_end=config.get("beta_end", 0.02)
        ).to(device)
        
        # Run comparison
        comparison_result = compare_forward_reverse_trajectories(
            model, scheduler, x_start, num_timesteps_to_show=5, device=device
        )
        
        results[schedule_name] = {
            "config": config,
            "avg_psnr": comparison_result["summary"]["avg_psnr"],
            "avg_ssim": comparison_result["summary"]["avg_ssim"]
        }
    
    # Create comparison plot
    if save_path:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        schedule_names = list(results.keys())
        psnr_values = [results[name]["avg_psnr"] for name in schedule_names]
        ssim_values = [results[name]["avg_ssim"] for name in schedule_names]
        
        # PSNR comparison
        ax1.bar(schedule_names, psnr_values)
        ax1.set_title('Average PSNR by Schedule')
        ax1.set_ylabel('PSNR (dB)')
        ax1.tick_params(axis='x', rotation=45)
        
        # SSIM comparison
        ax2.bar(schedule_names, ssim_values)
        ax2.set_title('Average SSIM by Schedule')
        ax2.set_ylabel('SSIM')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return results


def generate_comparison_report(
    model: torch.nn.Module,
    scheduler: DDPMScheduler,
    test_images: torch.Tensor,
    output_dir: str,
    dataset_name: str = "unknown"
) -> str:
    """Generate a comprehensive comparison report.
    
    Args:
        model: Trained denoising model
        scheduler: DDPM scheduler
        test_images: Test images [B, C, H, W]
        output_dir: Output directory for report and figures
        dataset_name: Name of dataset
    
    Returns:
        Path to generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    test_images = test_images.to(device)
    
    # Generate comparison data
    comparison_result = compare_forward_reverse_trajectories(
        model, scheduler, test_images[:4],  # Use first 4 images
        num_timesteps_to_show=8,
        device=device,
        save_path=os.path.join(output_dir, "trajectory_comparison.png")
    )
    
    # Test different schedules
    schedule_configs = [
        {"name": "linear", "beta_schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02},
        {"name": "cosine", "beta_schedule": "cosine"},
        {"name": "quadratic", "beta_schedule": "quadratic", "beta_start": 0.0001, "beta_end": 0.02},
    ]
    
    schedule_sensitivity = analyze_schedule_sensitivity(
        model, test_images[:2],  # Use fewer images for schedule analysis
        schedule_configs,
        device=device,
        save_path=os.path.join(output_dir, "schedule_sensitivity.png")
    )
    
    # Create forward vs reverse panel
    create_forward_vs_reverse_panel(
        model, scheduler, test_images[:2],
        timesteps_to_show=[10, 25, 50, 75],
        device=device,
        save_path=os.path.join(output_dir, "forward_vs_reverse_panel.png")
    )
    
    # Generate markdown report
    report_path = os.path.join(output_dir, "reverse_process_comparison.md")
    
    with open(report_path, 'w') as f:
        f.write(f"# Forward vs Reverse Process Comparison\n\n")
        f.write(f"**Dataset:** {dataset_name}\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Summary\n\n")
        f.write("This report compares the forward diffusion process (noise addition) ")
        f.write("with the learned reverse process (denoising) to evaluate how well ")
        f.write("the model has learned to invert the noise addition process.\n\n")
        
        f.write("### Key Metrics\n\n")
        summary = comparison_result["summary"]
        f.write(f"- **Average PSNR:** {summary['avg_psnr']:.2f} dB\n")
        f.write(f"- **Average SSIM:** {summary['avg_ssim']:.4f}\n")
        f.write(f"- **Minimum PSNR:** {summary['min_psnr']:.2f} dB\n")
        f.write(f"- **Maximum SSIM:** {summary['max_ssim']:.4f}\n\n")
        
        f.write("## Reconstruction Quality by Timestep\n\n")
        f.write("| Timestep | PSNR (dB) | SSIM |\n")
        f.write("|----------|-----------|------|\n")
        for error in comparison_result["reconstruction_errors"]:
            t = error["timestep"]
            psnr = error["psnr"]
            ssim = error["ssim"]
            if psnr == float('inf'):
                psnr_str = "∞"
            else:
                psnr_str = f"{psnr:.2f}"
            f.write(f"| {t:3d} | {psnr_str} | {ssim:.4f} |\n")
        f.write("\n")
        
        f.write("## Schedule Sensitivity Analysis\n\n")
        f.write("Comparison of different noise schedules:\n\n")
        f.write("| Schedule | PSNR (dB) | SSIM |\n")
        f.write("|----------|-----------|------|\n")
        for schedule_name, results in schedule_sensitivity.items():
            psnr = results["avg_psnr"]
            ssim = results["avg_ssim"]
            f.write(f"| {schedule_name} | {psnr:.2f} | {ssim:.4f} |\n")
        f.write("\n")
        
        f.write("## Observations\n\n")
        
        # Generate observations based on metrics
        best_psnr_schedule = max(schedule_sensitivity.keys(), 
                                key=lambda k: schedule_sensitivity[k]["avg_psnr"])
        best_ssim_schedule = max(schedule_sensitivity.keys(), 
                                key=lambda k: schedule_sensitivity[k]["avg_ssim"])
        
        f.write(f"- **Best PSNR Schedule:** {best_psnr_schedule} ")
        f.write(f"({schedule_sensitivity[best_psnr_schedule]['avg_psnr']:.2f} dB)\n")
        f.write(f"- **Best SSIM Schedule:** {best_ssim_schedule} ")
        f.write(f"({schedule_sensitivity[best_ssim_schedule]['avg_ssim']:.4f})\n")
        
        if summary["avg_psnr"] > 20:
            f.write("- Reconstruction quality is **good** (PSNR > 20 dB)\n")
        elif summary["avg_psnr"] > 15:
            f.write("- Reconstruction quality is **moderate** (PSNR 15-20 dB)\n")
        else:
            f.write("- Reconstruction quality is **poor** (PSNR < 15 dB)\n")
        
        if summary["avg_ssim"] > 0.8:
            f.write("- Structural similarity is **excellent** (SSIM > 0.8)\n")
        elif summary["avg_ssim"] > 0.6:
            f.write("- Structural similarity is **good** (SSIM 0.6-0.8)\n")
        else:
            f.write("- Structural similarity is **needs improvement** (SSIM < 0.6)\n")
        
        f.write("\n## Figures\n\n")
        f.write("1. **trajectory_comparison.png**: Side-by-side comparison of forward and reverse trajectories\n")
        f.write("2. **schedule_sensitivity.png**: Performance comparison across different noise schedules\n")
        f.write("3. **forward_vs_reverse_panel.png**: Detailed forward vs reverse process visualization\n")
        
        f.write("\n## Technical Details\n\n")
        f.write(f"- **Model Type:** {type(model).__name__}\n")
        f.write(f"- **Scheduler Type:** {type(scheduler).__name__}\n")
        f.write(f"- **Number of Timesteps:** {scheduler.num_timesteps}\n")
        f.write(f"- **Beta Schedule:** Linear (β_start=0.0001, β_end=0.02)\n")
        f.write(f"- **Test Images:** {test_images.shape[0]} samples\n")
    
    return report_path


def test_comparison():
    """Test comparison functionality."""
    from .ddpm_schedules import DDPMScheduler
    from .models.unet_tiny import UNetTiny
    
    # Create components
    scheduler = DDPMScheduler(num_timesteps=50)  # Smaller for testing
    model = UNetTiny(in_channels=3, out_channels=3, model_channels=32)
    
    # Test images
    test_images = torch.randn(2, 3, 32, 32)
    
    # Test trajectory comparison
    result = compare_forward_reverse_trajectories(
        model, scheduler, test_images,
        num_timesteps_to_show=5,
        save_path="test_trajectory.png"
    )
    print(f"Comparison result: {result['summary']}")
    
    # Test report generation
    report_path = generate_comparison_report(
        model, scheduler, test_images,
        output_dir="test_output",
        dataset_name="Test Dataset"
    )
    print(f"Generated report: {report_path}")
    
    print("Comparison tests completed!")


if __name__ == "__main__":
    test_comparison()