"""
Command-line interface for Day 8 beta schedule comparison.
Supports: train, sample.grid, sample.traj, plot.schedules, compare commands.
"""

import argparse
import sys
import torch
from pathlib import Path
from typing import List, Dict, Any

from .utils import load_config, set_seed, get_device, setup_logging_dir
from .dataset import get_dataloader
from .trainer import DDPMTrainer
from .models.unet_small import UNetSmall
from .schedules import get_schedule
from .sampler import DDPMSampler, DDIMSampler
from .visualize import (
    plot_schedules, trajectory_grid, multi_run_sample_panel, 
    create_reverse_animation, plot_training_curves
)
from .quality import create_comparison_report
from .utils import load_checkpoint, save_image_grid


def train_command(args):
    """Train a DDPM model with specified beta schedule."""
    print(f"Training with config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['seed'])
    
    # Get device
    device = get_device(config['device'])
    print(f"Using device: {device}")
    
    # Create dataloader
    data_config = config['data']
    train_loader = get_dataloader(
        dataset=data_config['dataset'],
        root=data_config['root'],
        batch_size=data_config['batch_size'],
        num_workers=data_config['num_workers'],
        normalize=data_config['normalize'],
        train=True,
        download=True
    )
    
    print(f"Dataset: {data_config['dataset']}, Train samples: {len(train_loader.dataset)}")
    
    # Create trainer
    trainer = DDPMTrainer(config)
    
    # Train
    trainer.train(train_loader)
    
    print("Training completed!")


def sample_grid_command(args):
    """Generate sample grid from trained model."""
    print(f"Generating sample grid with config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    device = get_device(config['device'])
    
    # Load model
    model_config = config['model']
    model = UNetSmall(
        in_channels=model_config['in_ch'],
        out_channels=model_config['in_ch'],
        base_channels=model_config['base_ch'],
        channel_multipliers=model_config['ch_mult'],
        time_embed_dim=model_config['time_embed_dim']
    ).to(device)
    
    # Load checkpoint
    run_dir = Path(config['log']['out_root']) / config['log']['run_name']
    ckpt_path = run_dir / 'ckpts' / 'ema.pt'
    if not ckpt_path.exists():
        ckpt_path = run_dir / 'ckpts' / 'best.pt'
    
    if not ckpt_path.exists():
        print(f"No checkpoint found in {run_dir / 'ckpts'}")
        return
    
    print(f"Loading checkpoint: {ckpt_path}")
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
    
    # Create sampler
    sample_config = config['sample']
    
    if args.ddim:
        sampler = DDIMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars'],
            eta=sample_config.get('eta', 0.0)
        )
    else:
        sampler = DDPMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars']
        )
    
    # Determine image shape
    data_config = config['data']
    if data_config['dataset'].lower() == 'mnist':
        shape = (sample_config['num_images'], 1, 28, 28)
    else:
        shape = (sample_config['num_images'], 3, 32, 32)
    
    # Generate samples
    print(f"Generating {sample_config['num_images']} samples...")
    model.eval()
    with torch.no_grad():
        if args.ddim:
            samples = sampler.sample(
                shape, device, 
                num_steps=sample_config.get('ddim_steps', 50)
            )
        else:
            samples = sampler.sample(shape, device)
    
    # Save grid
    grid_filename = 'samples_ddim.png' if args.ddim else 'samples_ddpm.png'
    grid_path = run_dir / 'grids' / grid_filename
    save_image_grid(samples, grid_path, nrow=8)
    
    print(f"Sample grid saved to {grid_path}")


def sample_trajectory_command(args):
    """Generate trajectory grid and animation."""
    print(f"Generating trajectory with config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    device = get_device(config['device'])
    
    # Load model (same as sample_grid)
    model_config = config['model']
    model = UNetSmall(
        in_channels=model_config['in_ch'],
        out_channels=model_config['in_ch'],
        base_channels=model_config['base_ch'],
        channel_multipliers=model_config['ch_mult'],
        time_embed_dim=model_config['time_embed_dim']
    ).to(device)
    
    # Load checkpoint
    run_dir = Path(config['log']['out_root']) / config['log']['run_name']
    ckpt_path = run_dir / 'ckpts' / 'ema.pt'
    if not ckpt_path.exists():
        ckpt_path = run_dir / 'ckpts' / 'best.pt'
    
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
    
    # Generate trajectory grid
    print("Generating trajectory grid...")
    traj_grid_path = run_dir / 'grids' / 'trajectory_grid.png'
    trajectory_grid(
        model, schedule, device,
        num_samples=8, num_timesteps=10,
        save_path=traj_grid_path
    )
    
    # Generate animation
    print("Generating reverse animation...")
    anim_path = run_dir / 'animations' / 'reverse_traj.gif'
    create_reverse_animation(
        model, schedule, device,
        save_path=anim_path,
        num_frames=50, duration=100
    )
    
    print("Trajectory generation completed!")


def plot_schedules_command(args):
    """Plot schedule comparison."""
    print("Plotting schedule comparison...")
    
    # Load all configs
    configs = []
    for config_path in args.configs:
        config = load_config(config_path)
        configs.append(config)
    
    # Create plots directory
    output_dir = Path("outputs/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot schedules
    save_path = output_dir / "schedules_overlay.png"
    plot_schedules(configs, save_path=save_path)
    
    print(f"Schedule plots saved to {save_path}")


def compare_command(args):
    """Compare multiple runs and generate comprehensive report."""
    print("Comparing runs and generating report...")
    
    # Load configs
    configs = []
    run_dirs = []
    
    for config_path in args.configs:
        config = load_config(config_path)
        configs.append(config)
        
        run_dir = Path(config['log']['out_root']) / config['log']['run_name']
        run_dirs.append(str(run_dir))
    
    # Get device
    device = get_device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create real dataloader (use first config for dataset info)
    data_config = configs[0]['data']
    real_dataloader = get_dataloader(
        dataset=data_config['dataset'],
        root=data_config['root'],
        batch_size=64,  # Smaller batch for evaluation
        num_workers=2,
        normalize=data_config['normalize'],
        train=False,  # Use test set for evaluation
        download=True,
        shuffle=False
    )
    
    # Create comparison report
    comparison_dir = "outputs/comparison"
    create_comparison_report(
        run_dirs, args.configs, real_dataloader, device, comparison_dir
    )
    
    # Plot training curves
    curves_path = Path(comparison_dir) / "training_curves.png"
    plot_training_curves(run_dirs, save_path=curves_path)
    
    # Create multi-run sample panel
    print("Generating multi-run sample panel...")
    model_paths = []
    for run_dir in run_dirs:
        ema_path = Path(run_dir) / 'ckpts' / 'ema.pt'
        if ema_path.exists():
            model_paths.append(str(ema_path))
        else:
            best_path = Path(run_dir) / 'ckpts' / 'best.pt'
            model_paths.append(str(best_path))
    
    panel_path = Path(comparison_dir) / "multi_run_samples.png"
    multi_run_sample_panel(
        model_paths, args.configs, device, 
        num_samples=16, save_path=panel_path
    )
    
    print(f"Comparison completed! Results saved to {comparison_dir}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Day 8: Beta Schedule Comparison CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with linear schedule
  python -m src.cli train --config configs/linear.yaml
  
  # Generate sample grid
  python -m src.cli sample.grid --config configs/cosine.yaml
  
  # Generate DDIM samples  
  python -m src.cli sample.grid --config configs/cosine.yaml --ddim
  
  # Generate trajectory visualization
  python -m src.cli sample.traj --config configs/quadratic.yaml
  
  # Plot schedule comparison
  python -m src.cli plot.schedules --configs configs/linear.yaml configs/cosine.yaml configs/quadratic.yaml
  
  # Compare all runs
  python -m src.cli compare --configs configs/linear.yaml configs/cosine.yaml configs/quadratic.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train DDPM model')
    train_parser.add_argument('--config', type=str, required=True,
                             help='Path to config file')
    
    # Sample grid command
    sample_grid_parser = subparsers.add_parser('sample.grid', help='Generate sample grid')
    sample_grid_parser.add_argument('--config', type=str, required=True,
                                   help='Path to config file')
    sample_grid_parser.add_argument('--ddim', action='store_true',
                                   help='Use DDIM sampling instead of DDPM')
    
    # Sample trajectory command
    sample_traj_parser = subparsers.add_parser('sample.traj', help='Generate trajectory visualization')
    sample_traj_parser.add_argument('--config', type=str, required=True,
                                   help='Path to config file')
    
    # Plot schedules command
    plot_parser = subparsers.add_parser('plot.schedules', help='Plot schedule comparison')
    plot_parser.add_argument('--configs', type=str, nargs='+', required=True,
                           help='Paths to config files')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare runs and generate report')
    compare_parser.add_argument('--configs', type=str, nargs='+', required=True,
                              help='Paths to config files')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Execute command
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'sample.grid':
            sample_grid_command(args)
        elif args.command == 'sample.traj':
            sample_trajectory_command(args)
        elif args.command == 'plot.schedules':
            plot_schedules_command(args)
        elif args.command == 'compare':
            compare_command(args)
        else:
            print(f"Unknown command: {args.command}")
            sys.exit(1)
    
    except Exception as e:
        print(f"Error executing {args.command}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()