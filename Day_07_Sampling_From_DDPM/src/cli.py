"""
Command-line interface for DDPM sampling operations.

Commands:
- sample.grid: Generate sample grids from checkpoints
- sample.traj: Generate trajectory animations
- compare.ckpts: Compare multiple checkpoints
- eval.quality: Evaluate sample quality metrics
"""

import argparse
import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from .utils import set_seed, get_device, create_output_dirs, Timer, log_system_info
from .ddpm_schedules import get_schedule_coefficients
from .models.unet_small import create_unet_small
from .checkpoints import CheckpointManager
from .sampler import DDPMSampler, create_sampler
from .visualize import (
    reverse_trajectory_grid, make_animation, multi_checkpoint_panel,
    plot_sampling_curves, create_sampling_report
)
from .quality import QualityEvaluator, evaluate_checkpoint_quality, load_dataset_samples, save_quality_results


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_from_config(config: Dict[str, Any]) -> tuple:
    """
    Set up device, seed, and output directories from config.
    
    Returns:
        Tuple of (device, output_dir)
    """
    # Set seed
    seed = config.get('seed', 42)
    set_seed(seed)
    
    # Get device
    device_str = config.get('device', 'cuda')
    device = get_device(device_str)
    
    # Create output directories
    output_dir = Path(config.get('log', {}).get('out_dir', './outputs'))
    create_output_dirs(output_dir)
    
    print(log_system_info(device))
    
    return device, output_dir


def create_model_and_sampler(config: Dict[str, Any], device: torch.device) -> tuple:
    """
    Create model and sampler from config.
    
    Returns:
        Tuple of (model, sampler, schedules)
    """
    # Create model
    model = create_unet_small(config)
    model.to(device)
    model.eval()
    
    # Create sampler
    sampler = create_sampler(model, config, device)
    
    return model, sampler, sampler.schedules


def cmd_sample_grid(args, config: Dict[str, Any]) -> None:
    """Generate sample grids from checkpoints."""
    print("=== DDPM Sample Grid Generation ===")
    
    device, output_dir = setup_from_config(config)
    model, sampler, schedules = create_model_and_sampler(config, device)
    
    # Setup checkpoint manager
    ckpt_config = config['checkpoints']
    ckpt_manager = CheckpointManager(
        ckpt_dir=ckpt_config['dir'],
        ckpt_list=ckpt_config['list'],
        use_ema=ckpt_config.get('use_ema', True),
        device=device
    )
    
    # Sample configuration
    sample_config = config['sample']
    num_images = sample_config.get('num_images', 64)
    
    # Determine image shape from model config
    model_config = config['model']
    in_ch = model_config['in_ch']
    if config['data']['dataset'] == 'mnist':
        # Use 32x32 for compatibility with Day 6 models
        img_shape = (in_ch, 32, 32)
    elif config['data']['dataset'] == 'cifar10':
        img_shape = (in_ch, 32, 32)
    else:
        img_shape = (in_ch, 32, 32)  # Default
    
    timer = Timer()
    
    # Generate samples for each checkpoint
    if args.ckpt:
        # Single checkpoint specified
        ckpt_list = [args.ckpt]
    else:
        # Use all checkpoints from config
        ckpt_list = ckpt_config['list']
    
    for ckpt_name in ckpt_list:
        print(f"\nGenerating samples for checkpoint: {ckpt_name}")
        
        # Find and load checkpoint
        ckpt_idx = -1
        for i, ckpt_path in enumerate(ckpt_manager.ckpt_paths):
            if ckpt_path.name == ckpt_name:
                ckpt_idx = i
                break
        
        if ckpt_idx == -1:
            print(f"Checkpoint {ckpt_name} not found, skipping...")
            continue
        
        metadata = ckpt_manager.load_checkpoint_into_model(model, ckpt_idx)
        epoch = metadata.get('epoch', 'unknown')
        
        # Generate samples
        timer.start('sampling')
        shape = (num_images, *img_shape)
        
        sampling_kwargs = {}
        if hasattr(args, 'ddim') and args.ddim:
            sampling_kwargs.update({
                'ddim': True,
                'num_steps': sample_config.get('ddim_steps', 50)
            })
        
        variance_type = config['diffusion'].get('posterior_var', 'posterior')
        if variance_type == 'beta':
            sampling_kwargs['variance_type'] = 'beta'
        else:
            sampling_kwargs['variance_type'] = 'posterior'
        
        results = sampler.sample(shape, **sampling_kwargs)
        samples = results['samples']
        
        sampling_time = timer.stop('sampling')
        print(f"Generated {num_images} samples in {sampling_time:.2f}s")
        
        # Save sample grid
        save_path = output_dir / "grids" / f"samples_ep{epoch}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create grid and save
        from .utils import save_image_grid
        nrow = int(num_images ** 0.5)
        save_image_grid(samples, save_path, nrow=nrow, normalize=True, value_range=(-1, 1))
        
        print(f"Saved sample grid to {save_path}")


def cmd_sample_trajectory(args, config: Dict[str, Any]) -> None:
    """Generate trajectory animations."""
    print("=== DDPM Trajectory Animation ===")
    
    device, output_dir = setup_from_config(config)
    model, sampler, schedules = create_model_and_sampler(config, device)
    
    # Setup checkpoint manager
    ckpt_config = config['checkpoints']
    ckpt_manager = CheckpointManager(
        ckpt_dir=ckpt_config['dir'],
        ckpt_list=ckpt_config['list'],
        use_ema=ckpt_config.get('use_ema', True),
        device=device
    )
    
    # Determine image shape
    model_config = config['model']
    in_ch = model_config['in_ch']
    if config['data']['dataset'] == 'mnist':
        # Use 32x32 for compatibility with Day 6 models
        img_shape = (in_ch, 32, 32)
    elif config['data']['dataset'] == 'cifar10':
        img_shape = (in_ch, 32, 32)
    else:
        img_shape = (in_ch, 32, 32)
    
    # Load checkpoint
    ckpt_name = args.ckpt if args.ckpt else ckpt_config['list'][0]
    ckpt_idx = -1
    for i, ckpt_path in enumerate(ckpt_manager.ckpt_paths):
        if ckpt_path.name == ckpt_name:
            ckpt_idx = i
            break
    
    if ckpt_idx == -1:
        print(f"Checkpoint {ckpt_name} not found!")
        return
    
    metadata = ckpt_manager.load_checkpoint_into_model(model, ckpt_idx)
    epoch = metadata.get('epoch', 'unknown')
    
    print(f"Generating trajectory for checkpoint: {ckpt_name} (epoch {epoch})")
    
    # Generate single trajectory
    shape = (1, *img_shape)
    record_every = args.record_every if hasattr(args, 'record_every') else 20
    
    results = sampler.sample_single_trajectory(
        shape, 
        record_every=record_every,
        progress=True,
        variance_type=config['diffusion'].get('posterior_var', 'posterior')
    )
    
    trajectory = results['trajectory']
    trajectory_steps = results['trajectory_steps']
    
    print(f"Generated trajectory with {len(trajectory)} frames")
    
    # Create animation
    save_path = output_dir / "animations" / f"traj_ep{epoch}.gif"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    make_animation(
        trajectory=trajectory,
        trajectory_steps=trajectory_steps,
        save_path=save_path,
        sample_idx=args.idx if hasattr(args, 'idx') else 0,
        fps=args.fps if hasattr(args, 'fps') else 8
    )


def cmd_compare_checkpoints(args, config: Dict[str, Any]) -> None:
    """Compare multiple checkpoints."""
    print("=== DDPM Checkpoint Comparison ===")
    
    device, output_dir = setup_from_config(config)
    model, sampler, schedules = create_model_and_sampler(config, device)
    
    # Setup checkpoint manager
    ckpt_config = config['checkpoints']
    ckpt_manager = CheckpointManager(
        ckpt_dir=ckpt_config['dir'],
        ckpt_list=ckpt_config['list'],
        use_ema=ckpt_config.get('use_ema', True),
        device=device
    )
    
    # Determine image shape
    model_config = config['model']
    in_ch = model_config['in_ch']
    if config['data']['dataset'] == 'mnist':
        img_shape = (in_ch, 28, 28)
    elif config['data']['dataset'] == 'cifar10':
        img_shape = (in_ch, 32, 32)
    else:
        img_shape = (in_ch, 32, 32)
    
    # Generate samples for each checkpoint
    sample_grids = {}
    num_samples = min(64, config['sample'].get('num_images', 64))
    shape = (num_samples, *img_shape)
    
    timer = Timer()
    
    for metadata, model in ckpt_manager.iterate_checkpoints(model):
        ckpt_name = Path(metadata['checkpoint_path']).name
        epoch = metadata.get('epoch', 'unknown')
        
        print(f"Generating samples for {ckpt_name} (epoch {epoch})")
        
        timer.start('sampling')
        results = sampler.sample(shape, progress=False)
        sampling_time = timer.stop('sampling')
        
        sample_grids[f"Epoch {epoch}"] = results['samples']
        print(f"Generated {num_samples} samples in {sampling_time:.2f}s")
    
    # Create comparison panel
    save_path = output_dir / "grids" / "checkpoint_comparison.png"
    multi_checkpoint_panel(
        sample_grids=sample_grids,
        save_path=save_path,
        nrow=8
    )
    
    print(f"Saved checkpoint comparison to {save_path}")


def cmd_eval_quality(args, config: Dict[str, Any]) -> None:
    """Evaluate sample quality metrics."""
    print("=== DDPM Quality Evaluation ===")
    
    device, output_dir = setup_from_config(config)
    model, sampler, schedules = create_model_and_sampler(config, device)
    
    # Setup checkpoint manager
    ckpt_config = config['checkpoints']
    ckpt_manager = CheckpointManager(
        ckpt_dir=ckpt_config['dir'],
        ckpt_list=ckpt_config['list'],
        use_ema=ckpt_config.get('use_ema', True),
        device=device
    )
    
    # Load real samples for comparison
    data_config = config['data']
    dataset_name = data_config['dataset']
    data_root = data_config['root']
    
    print(f"Loading real {dataset_name} samples...")
    real_samples = load_dataset_samples(
        dataset_name=dataset_name,
        data_root=data_root,
        num_samples=1000
    )
    
    # Determine image shape
    model_config = config['model']
    in_ch = model_config['in_ch']
    if dataset_name == 'mnist':
        # Use 32x32 for compatibility with Day 6 models
        img_shape = (in_ch, 32, 32)
    elif dataset_name == 'cifar10':
        img_shape = (in_ch, 32, 32)
    else:
        img_shape = (in_ch, 32, 32)
    
    # Evaluate quality
    evaluator = QualityEvaluator(device=device)
    
    num_samples = args.num_samples if hasattr(args, 'num_samples') else 200
    print(f"Evaluating with {num_samples} generated samples per checkpoint")
    
    results_df = evaluate_checkpoint_quality(
        sampler=sampler,
        checkpoint_manager=ckpt_manager,
        real_samples=real_samples,
        num_samples=num_samples,
        image_shape=img_shape,
        evaluator=evaluator
    )
    
    # Save results
    save_quality_results(results_df, output_dir / "logs")
    
    # Create quality curves if we have multiple checkpoints
    if len(results_df) > 1 and 'epoch' in results_df.columns:
        # Filter out rows with missing epoch info
        df_with_epochs = results_df.dropna(subset=['epoch'])
        if len(df_with_epochs) > 1:
            curve_path = output_dir / "curves" / "quality_vs_checkpoint.png"
            curve_path.parent.mkdir(parents=True, exist_ok=True)
            plot_sampling_curves(df_with_epochs, save_path=curve_path)
    
    # Create report
    checkpoint_results = {}
    for _, row in results_df.iterrows():
        checkpoint_results[row['checkpoint']] = row.to_dict()
    
    create_sampling_report(
        checkpoint_results=checkpoint_results,
        output_dir=output_dir / "reports",
        dataset_name=dataset_name
    )
    
    print("Quality evaluation completed!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="DDPM Sampling CLI")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    def add_common_args(subparser):
        subparser.add_argument('--config', required=True, help='Path to config YAML file')
    
    # sample.grid command
    grid_parser = subparsers.add_parser('sample.grid', help='Generate sample grids')
    add_common_args(grid_parser)
    grid_parser.add_argument('--ckpt', help='Specific checkpoint to use')
    grid_parser.add_argument('--ddim', action='store_true', help='Use DDIM sampling')
    
    # sample.traj command
    traj_parser = subparsers.add_parser('sample.traj', help='Generate trajectory animations')
    add_common_args(traj_parser)
    traj_parser.add_argument('--ckpt', help='Specific checkpoint to use')
    traj_parser.add_argument('--idx', type=int, default=0, help='Sample index to animate')
    traj_parser.add_argument('--record-every', type=int, default=20, help='Record every N steps')
    traj_parser.add_argument('--fps', type=int, default=8, help='Animation FPS')
    
    # compare.ckpts command
    compare_parser = subparsers.add_parser('compare.ckpts', help='Compare multiple checkpoints')
    add_common_args(compare_parser)
    
    # eval.quality command
    eval_parser = subparsers.add_parser('eval.quality', help='Evaluate sample quality')
    add_common_args(eval_parser)
    eval_parser.add_argument('--num-samples', type=int, default=200, 
                           help='Number of samples per checkpoint for evaluation')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Load config
    config = load_config(args.config)
    
    # Route to appropriate command
    if args.command == 'sample.grid':
        cmd_sample_grid(args, config)
    elif args.command == 'sample.traj':
        cmd_sample_trajectory(args, config)
    elif args.command == 'compare.ckpts':
        cmd_compare_checkpoints(args, config)
    elif args.command == 'eval.quality':
        cmd_eval_quality(args, config)
    else:
        print(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
