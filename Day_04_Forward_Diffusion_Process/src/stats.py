"""Statistics computation and logging for forward diffusion process."""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .forward import (
    q_xt_given_x0, snr, snr_db, kl_divergence_to_unit_gaussian, 
    compute_mse_to_x0, get_timesteps_for_snr_threshold
)


def compute_forward_stats(
    x0: torch.Tensor,
    betas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Compute statistics across all timesteps for forward diffusion.
    
    Args:
        x0: Original images, shape (batch_size, channels, height, width)
        betas: Beta schedule, shape (T,)
        alpha_bars: Alpha bar schedule, shape (T,)
        device: Device to run computations on
    
    Returns:
        Dictionary with statistics tensors
    """
    T = len(betas)
    batch_size = x0.shape[0]
    
    # Move to device
    x0 = x0.to(device)
    betas = betas.to(device)
    alpha_bars = alpha_bars.to(device)
    
    stats = {
        'timesteps': torch.arange(T, device=device),
        'betas': betas,
        'alpha_bars': alpha_bars,
        'snr_linear': snr(alpha_bars),
        'snr_db': snr_db(alpha_bars),
        'mse_to_x0': torch.zeros(T, device=device),
        'kl_to_unit': torch.zeros(T, device=device)
    }
    
    # Compute MSE and KL for each timestep
    print("Computing forward diffusion statistics...")
    for t in tqdm(range(T)):
        t_tensor = torch.full((batch_size,), t, device=device)
        
        # Sample x_t from x_0
        x_t, _ = q_xt_given_x0(x0, t_tensor, alpha_bars)
        
        # Compute MSE
        mse_batch = compute_mse_to_x0(x_t, x0)
        stats['mse_to_x0'][t] = mse_batch.mean()
        
        # Compute KL divergence
        kl_batch = kl_divergence_to_unit_gaussian(x0, t_tensor, alpha_bars)
        stats['kl_to_unit'][t] = kl_batch.mean()
    
    return stats


def save_stats_csv(
    stats: Dict[str, torch.Tensor],
    filepath: Path
) -> None:
    """Save statistics to CSV file.
    
    Args:
        stats: Statistics dictionary from compute_forward_stats
        filepath: Path to save CSV file
    """
    # Convert to numpy for pandas
    data = {}
    for key, tensor in stats.items():
        if tensor.dim() == 0:  # Scalar
            continue
        data[key] = tensor.cpu().numpy()
    
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    print(f"Saved statistics to {filepath}")


def analyze_snr_thresholds(
    alpha_bars: torch.Tensor,
    thresholds_db: List[float] = [-5, -10, -15, -20]
) -> Dict[float, int]:
    """Analyze when SNR drops below various thresholds.
    
    Args:
        alpha_bars: Alpha bar schedule
        thresholds_db: List of SNR thresholds in dB
    
    Returns:
        Dictionary mapping threshold to first timestep below threshold
    """
    snr_values_db = snr_db(alpha_bars)
    threshold_times = {}
    
    for threshold in thresholds_db:
        timesteps_below = get_timesteps_for_snr_threshold(alpha_bars, threshold)
        if len(timesteps_below) > 0:
            threshold_times[threshold] = int(timesteps_below[0])
        else:
            threshold_times[threshold] = len(alpha_bars)  # Never drops below
    
    return threshold_times


def compute_schedule_comparison(
    x0: torch.Tensor,
    T: int,
    device: torch.device,
    schedules: List[str] = ["linear", "cosine", "sigmoid"]
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Compare different noise schedules.
    
    Args:
        x0: Sample images for evaluation
        T: Number of timesteps
        device: Device to run on
        schedules: List of schedule types to compare
    
    Returns:
        Dictionary mapping schedule name to statistics
    """
    from .ddpm_schedules import get_ddpm_schedule
    
    comparison = {}
    
    for schedule_name in schedules:
        print(f"Computing statistics for {schedule_name} schedule...")
        
        # Get schedule
        betas, alphas, alpha_bars = get_ddpm_schedule(T, schedule_name)
        
        # Compute stats
        stats = compute_forward_stats(x0, betas, alpha_bars, device)
        
        # Add schedule name
        stats['schedule'] = schedule_name
        
        # Analyze SNR thresholds
        thresholds = analyze_snr_thresholds(alpha_bars)
        stats['snr_thresholds'] = thresholds
        
        comparison[schedule_name] = stats
    
    return comparison


def print_schedule_summary(stats: Dict[str, torch.Tensor]) -> None:
    """Print summary of schedule statistics.
    
    Args:
        stats: Statistics dictionary
    """
    schedule_name = stats.get('schedule', 'Unknown')
    T = len(stats['betas'])
    
    print(f"\n=== {schedule_name.upper()} Schedule Summary ===")
    print(f"Total timesteps: {T}")
    print(f"Beta range: {stats['betas'].min():.6f} - {stats['betas'].max():.6f}")
    print(f"Final alpha_bar: {stats['alpha_bars'][-1]:.6f}")
    print(f"Final SNR: {stats['snr_db'][-1]:.2f} dB")
    
    if 'snr_thresholds' in stats:
        print("SNR thresholds:")
        for threshold, timestep in stats['snr_thresholds'].items():
            if timestep < T:
                print(f"  {threshold:+.0f} dB at t={timestep}")
            else:
                print(f"  {threshold:+.0f} dB never reached")
    
    print(f"Final MSE to x_0: {stats['mse_to_x0'][-1]:.4f}")
    print(f"Final KL to N(0,I): {stats['kl_to_unit'][-1]:.4f}")


def compute_class_conditional_stats(
    dataloader,
    betas: torch.Tensor,
    alpha_bars: torch.Tensor,
    device: torch.device,
    num_classes: int = 10,
    max_batches: Optional[int] = None
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Compute per-class degradation statistics.
    
    Args:
        dataloader: DataLoader with labeled data
        betas: Beta schedule
        alpha_bars: Alpha bar schedule
        device: Device to run on
        num_classes: Number of classes
        max_batches: Maximum batches to process (for speed)
    
    Returns:
        Dictionary mapping class to statistics
    """
    T = len(betas)
    class_stats = {}
    
    # Initialize statistics for each class
    for c in range(num_classes):
        class_stats[c] = {
            'count': 0,
            'mse_to_x0': torch.zeros(T, device=device),
            'kl_to_unit': torch.zeros(T, device=device)
        }
    
    batch_count = 0
    print("Computing per-class statistics...")
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
        if max_batches and batch_idx >= max_batches:
            break
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Process each class separately
        for c in range(num_classes):
            class_mask = labels == c
            if class_mask.sum() == 0:
                continue
            
            class_images = images[class_mask]
            if len(class_images) == 0:
                continue
            
            # Compute stats for this class
            for t in range(T):
                t_tensor = torch.full((len(class_images),), t, device=device)
                
                # Sample x_t
                x_t, _ = q_xt_given_x0(class_images, t_tensor, alpha_bars)
                
                # Update running averages
                mse_batch = compute_mse_to_x0(x_t, class_images).mean()
                kl_batch = kl_divergence_to_unit_gaussian(
                    class_images, t_tensor, alpha_bars
                ).mean()
                
                old_count = class_stats[c]['count']
                new_count = old_count + len(class_images)
                
                # Running average update
                class_stats[c]['mse_to_x0'][t] = (
                    old_count * class_stats[c]['mse_to_x0'][t] + 
                    len(class_images) * mse_batch
                ) / new_count
                
                class_stats[c]['kl_to_unit'][t] = (
                    old_count * class_stats[c]['kl_to_unit'][t] + 
                    len(class_images) * kl_batch
                ) / new_count
            
            class_stats[c]['count'] = new_count
        
        batch_count += 1
    
    # Remove count from final stats
    for c in range(num_classes):
        del class_stats[c]['count']
        
    return class_stats