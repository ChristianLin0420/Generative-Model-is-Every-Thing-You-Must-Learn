"""
Command line interface for Day 1: Gaussian Noise Basics
"""

import argparse
import csv
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from .dataset import get_mnist_loader, get_sample_batch, denormalize_tensor
from .metrics import MetricsTracker, noise_degradation_metrics
from .noise import sigma_schedule, NoiseScheduler
from .utils import set_seed, get_device, create_output_dirs
from .visualize import make_progressive_grid, make_animation, plot_metrics


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config = OmegaConf.load(config_path)
    return OmegaConf.to_container(config, resolve=True)


def add_noise_command(args):
    """Command to generate noised images and save grids."""
    print("=== Add Noise Command ===")
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    set_seed(config['seed'])
    device = get_device(config['device'])
    create_output_dirs(config['output_dir'])
    
    # Get data loader
    dataloader = get_mnist_loader(
        root=config['dataset']['root'],
        split=config['dataset']['split'],
        batch_size=config['dataset']['batch_size'],
        normalize_range=tuple(config['dataset']['normalize_range']),
        shuffle=True
    )
    
    # Get noise schedule
    if 'sigmas' in config['noise']:
        sigmas = config['noise']['sigmas']
    else:
        scheduler = NoiseScheduler(
            schedule_type=config['noise']['schedule'],
            num_levels=config['noise']['num_levels'],
            min_sigma=config['noise']['min_sigma'],
            max_sigma=config['noise']['max_sigma']
        )
        sigmas = scheduler.get_sigmas()
    
    print(f"Using noise levels: {sigmas}")
    
    # Process batches
    num_batches = args.num_batches or config['experiment']['num_batches']
    metrics_tracker = MetricsTracker()
    
    for batch_idx in range(num_batches):
        print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")
        
        # Get batch
        images, labels = get_sample_batch(dataloader, device)
        
        # Create progressive grid for this batch
        output_path = Path(config['output_dir']) / 'grids' / f'batch_{batch_idx:03d}_noise_grid.png'
        
        make_progressive_grid(
            batch=images,
            sigmas=sigmas,
            path=output_path,
            nrow=config['visualization']['grid_nrow'],
            normalize_range=tuple(config['dataset']['normalize_range']),
            add_labels=True
        )
        
        # Compute metrics for each sigma
        for sigma in sigmas:
            original_denorm = denormalize_tensor(images, tuple(config['dataset']['normalize_range']))
            
            # Add noise
            from .noise import add_gaussian_noise
            generator = torch.Generator(device=device).manual_seed(config['seed'] + batch_idx)
            
            if tuple(config['dataset']['normalize_range']) == (0, 1):
                clip_range = (0, 1)
            elif tuple(config['dataset']['normalize_range']) == (-1, 1):
                clip_range = (-1, 1)
            else:
                clip_range = None
                
            noisy_images = add_gaussian_noise(images, sigma, clip_range, generator)
            noisy_denorm = denormalize_tensor(noisy_images, tuple(config['dataset']['normalize_range']))
            
            # Compute metrics
            metrics = noise_degradation_metrics(
                original_denorm,
                sigma,
                noisy_denorm,
                data_std=1.0,
                max_val=1.0
            )
            metrics['batch_idx'] = batch_idx
            metrics_tracker.add_measurement(
                original_denorm, noisy_denorm, sigma, max_val=1.0, data_std=1.0
            )
    
    # Save metrics
    csv_path = Path(config['output_dir']) / 'logs' / 'noise_metrics.csv'
    metrics_tracker.save_csv(csv_path)
    
    # Create metrics plot
    plot_path = Path(config['output_dir']) / 'logs' / 'metrics_plot.png'
    plot_metrics(csv_path, plot_path, 
                 figsize=tuple(config['visualization']['figsize']),
                 dpi=config['visualization']['plot_dpi'])
    
    print(f"\n✅ Add noise command completed!")
    print(f"   Generated {num_batches} noise grids")
    print(f"   Saved metrics to {csv_path}")
    print(f"   Saved plots to {plot_path}")


def animate_command(args):
    """Command to create noise progression animation."""
    print("=== Animate Command ===")
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    set_seed(config['seed'])
    device = get_device(config['device'])
    create_output_dirs(config['output_dir'])
    
    # Get data loader
    dataloader = get_mnist_loader(
        root=config['dataset']['root'],
        split=config['dataset']['split'],
        batch_size=config['dataset']['batch_size'],
        normalize_range=tuple(config['dataset']['normalize_range']),
        shuffle=True
    )
    
    # Get sample batch
    images, _ = get_sample_batch(dataloader, device)
    
    # Select number of images
    num_images = args.num_images or config['experiment']['num_images']
    selected_images = images[:num_images]
    
    # Get noise schedule
    if 'sigmas' in config['noise']:
        sigmas = config['noise']['sigmas']
    else:
        scheduler = NoiseScheduler(
            schedule_type=config['noise']['schedule'],
            num_levels=config['noise']['num_levels'],
            min_sigma=config['noise']['min_sigma'],
            max_sigma=config['noise']['max_sigma']
        )
        sigmas = scheduler.get_sigmas()
    
    print(f"Creating animation with {len(sigmas)} frames")
    print(f"Using {num_images} images")
    
    # Create animation
    output_path = args.out or (Path(config['output_dir']) / 'animations' / 'mnist_noise.gif')
    output_path = Path(output_path)
    
    make_animation(
        batch=selected_images,
        sigmas=sigmas,
        path=output_path,
        fps=config['visualization']['animation_fps'],
        normalize_range=tuple(config['dataset']['normalize_range'])
    )
    
    print(f"✅ Animation saved to {output_path}")


def stats_command(args):
    """Command to compute and save statistics."""
    print("=== Stats Command ===")
    
    # Load config
    config = load_config(args.config)
    
    # Setup
    set_seed(config['seed'])
    device = get_device(config['device'])
    create_output_dirs(config['output_dir'])
    
    # Get noise schedule
    if 'sigmas' in config['noise']:
        sigmas = config['noise']['sigmas']
    else:
        scheduler = NoiseScheduler(
            schedule_type=config['noise']['schedule'],
            num_levels=config['noise']['num_levels'],
            min_sigma=config['noise']['min_sigma'],
            max_sigma=config['noise']['max_sigma']
        )
        sigmas = scheduler.get_sigmas()
    
    # Compute theoretical statistics
    stats = []
    for sigma in sigmas:
        from .metrics import compute_snr_db
        snr_db = compute_snr_db(sigma, data_std=1.0)
        
        stats.append({
            'sigma': sigma,
            'snr_db': snr_db,
            'noise_power': sigma ** 2,
            'signal_power': 1.0  # Assuming unit variance data
        })
    
    # Save to CSV
    csv_path = Path(config['output_dir']) / 'logs' / 'theoretical_stats.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['sigma', 'snr_db', 'noise_power', 'signal_power']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
    
    print(f"✅ Theoretical statistics saved to {csv_path}")
    
    # Print summary
    print("\nNoise Schedule Summary:")
    print("σ      | SNR (dB) | Noise Power")
    print("-------|----------|------------")
    for stat in stats:
        print(f"{stat['sigma']:.3f}  | {stat['snr_db']:8.2f} | {stat['noise_power']:.6f}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Day 1: Gaussian Noise Basics")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        '--config', 
        type=str, 
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    
    # Add-noise command
    add_noise_parser = subparsers.add_parser(
        'add-noise',
        parents=[common_parser],
        help='Generate noised images and save grids'
    )
    add_noise_parser.add_argument(
        '--num-batches',
        type=int,
        help='Number of batches to process'
    )
    
    # Animate command
    animate_parser = subparsers.add_parser(
        'animate',
        parents=[common_parser],
        help='Create noise progression animation'
    )
    animate_parser.add_argument(
        '--num-images',
        type=int,
        help='Number of images to include in animation'
    )
    animate_parser.add_argument(
        '--out',
        type=str,
        help='Output path for animation'
    )
    
    # Stats command
    stats_parser = subparsers.add_parser(
        'stats',
        parents=[common_parser],
        help='Compute and save noise statistics'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command == 'add-noise':
        add_noise_command(args)
    elif args.command == 'animate':
        animate_command(args)
    elif args.command == 'stats':
        stats_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()