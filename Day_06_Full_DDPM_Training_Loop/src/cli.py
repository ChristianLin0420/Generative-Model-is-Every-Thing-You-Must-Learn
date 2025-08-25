"""
Command-line interface for DDPM training, sampling, evaluation, and visualization
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import load_config, set_seed, get_device, create_output_dirs, Logger
from .dataset import create_dataloaders, get_dataset_stats
from .ddpm_schedules import DDPMSchedules
from .models import UNetSmall
from .models.simple_unet import SimpleUNet
from .losses import get_loss_fn
from .trainer import DDPMTrainer
from .sampler import DDPMSampler
from .eval import DDPMEvaluator
from .visualize import VisualizationManager, plot_training_curves, plot_schedule_comparison


def create_model_from_config(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create model from configuration"""
    model_config = config["model"]
    
    if model_config["type"] == "unet_small":
        # Use simple UNet for now to avoid architecture issues
        model = SimpleUNet(
            in_channels=model_config["in_channels"],
            out_channels=model_config["out_channels"],
            model_channels=model_config["model_channels"],
            time_embed_dim=model_config["time_embed_dim"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_config['type']}")
        
    return model.to(device)


def create_schedules_from_config(config: Dict[str, Any], device: torch.device) -> DDPMSchedules:
    """Create schedules from configuration"""
    schedule_config = config["schedules"]
    
    return DDPMSchedules(
        num_timesteps=schedule_config["num_timesteps"],
        schedule_type=schedule_config["schedule_type"],
        beta_start=schedule_config.get("beta_start", 0.0001),
        beta_end=schedule_config.get("beta_end", 0.02),
        cosine_s=schedule_config.get("cosine_s", 0.008),
        device=str(device)
    )


def train_command(args):
    """Training command"""
    print(f"🚀 Starting DDMP training with config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.epochs is not None:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
        
    # Setup
    set_seed(config.get("seed", 42))
    device = get_device(args.device or config.get("device", "auto"))
    
    output_dir = args.output_dir or config["output"]["base_dir"]
    create_output_dirs(output_dir)
    
    logger = Logger(Path(output_dir) / "logs", "training")
    logger.info(f"Starting training with config: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"Output directory: {output_dir}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    dataloaders = create_dataloaders(config)
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]
    
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = create_model_from_config(config, device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create schedules
    schedules = create_schedules_from_config(config, device)
    
    # Create loss function
    loss_fn = get_loss_fn(schedules, config["loss"])
    
    # Create trainer
    trainer = DDPMTrainer(
        model=model,
        schedules=schedules,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=output_dir
    )
    
    # Resume from checkpoint if specified
    resume_from = args.resume
    if resume_from:
        logger.info(f"Resuming from checkpoint: {resume_from}")
        
    # Start training
    start_time = time.time()
    trainer.train(resume_from=resume_from)
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/3600:.2f} hours)")
    print(f"✅ Training completed! Check outputs in: {output_dir}")


def sample_grid_command(args):
    """Generate sample grid"""
    print(f"🎨 Generating sample grid with config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup
    set_seed(config.get("seed", 42))
    device = get_device(args.device or config.get("device", "auto"))
    output_dir = args.output_dir or config["output"]["base_dir"]
    
    # Create model and schedules
    model = create_model_from_config(config, device)
    schedules = create_schedules_from_config(config, device)
    sampler = DDPMSampler(schedules)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint or Path(output_dir) / "ckpts" / "ema.pt"
    if not checkpoint_path.exists():
        checkpoint_path = Path(output_dir) / "ckpts" / "latest.pt"
        
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if "ema_state_dict" in checkpoint:
            # Load EMA weights
            ema_state_dict = checkpoint["ema_state_dict"]
            model.load_state_dict(ema_state_dict)
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
    else:
        print(f"Warning: No checkpoint found at {checkpoint_path}, using random weights")
        
    # Get sample shape from dataset
    dataset_stats = get_dataset_stats(config["dataset"]["name"])
    channels = dataset_stats["channels"]
    image_size = config["dataset"]["image_size"]
    
    num_samples = args.num_samples or config["sampling"].get("num_samples", 64)
    sample_shape = (num_samples, channels, image_size, image_size)
    
    # Generate samples
    print(f"Generating {num_samples} samples...")
    start_time = time.time()
    
    samples = sampler.sample(
        model=model,
        shape=sample_shape,
        method=args.method or config["sampling"]["method"],
        num_steps=args.num_steps or config["sampling"]["num_steps"],
        eta=config["sampling"].get("eta", 0.0),
        device=device,
        progress=True
    )
    
    generation_time = time.time() - start_time
    print(f"Generated samples in {generation_time:.2f} seconds")
    
    # Save grid
    viz_manager = VisualizationManager(output_dir)
    
    filename = args.output_name or f"samples_{args.method or config['sampling']['method']}_{num_samples}.png"
    grid_path = viz_manager.save_sample_grid(
        samples=samples,
        filename=filename,
        nrow=int(num_samples**0.5),
        title=f"Generated Samples ({args.method or config['sampling']['method']})"
    )
    
    print(f"✅ Sample grid saved: {grid_path}")


def sample_trajectory_command(args):
    """Generate trajectory animation"""
    print(f"🎬 Generating trajectory animation with config: {args.config}")
    
    # Load configuration and setup (similar to sample_grid_command)
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device(args.device or config.get("device", "auto"))
    output_dir = args.output_dir or config["output"]["base_dir"]
    
    # Create model, schedules, sampler
    model = create_model_from_config(config, device)
    schedules = create_schedules_from_config(config, device)
    sampler = DDPMSampler(schedules)
    
    # Load checkpoint (similar to sample_grid_command)
    checkpoint_path = args.checkpoint or Path(output_dir) / "ckpts" / "ema.pt"
    if not checkpoint_path.exists():
        checkpoint_path = Path(output_dir) / "ckpts" / "latest.pt"
        
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "ema_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["ema_state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    
    # Get sample shape
    dataset_stats = get_dataset_stats(config["dataset"]["name"])
    channels = dataset_stats["channels"]
    image_size = config["dataset"]["image_size"]
    sample_shape = (channels, image_size, image_size)
    
    # Generate animation
    viz_manager = VisualizationManager(output_dir)
    
    filename = args.output_name or "reverse_trajectory.gif"
    animation_path = viz_manager.save_animation(
        model=model,
        sampler=sampler,
        shape=sample_shape,
        filename=filename,
        method=args.method or config["sampling"]["method"],
        num_steps=args.num_steps or config["sampling"]["num_steps"],
        fps=args.fps or 10,
        device=device
    )
    
    print(f"✅ Trajectory animation saved: {animation_path}")


def eval_command(args):
    """Evaluation command"""
    print(f"📊 Running evaluation with config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device(args.device or config.get("device", "auto"))
    output_dir = args.output_dir or config["output"]["base_dir"]
    
    # Create model and schedules
    model = create_model_from_config(config, device)
    schedules = create_schedules_from_config(config, device)
    sampler = DDPMSampler(schedules)
    
    # Load checkpoint
    checkpoint_path = args.checkpoint or Path(output_dir) / "ckpts" / "ema.pt"
    if not checkpoint_path.exists():
        checkpoint_path = Path(output_dir) / "ckpts" / "latest.pt"
        
    if checkpoint_path.exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "ema_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["ema_state_dict"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    # Create data loaders
    dataloaders = create_dataloaders(config)
    val_loader = dataloaders["val"]
    
    # Create evaluator
    evaluator = DDPMEvaluator(
        model=model,
        schedules=schedules,
        sampler=sampler,
        device=device,
        use_lpips=config["evaluation"].get("use_lpips", True),
        use_fid=config["evaluation"].get("use_fid", True)
    )
    
    # Run comprehensive evaluation
    print("Running comprehensive evaluation...")
    start_time = time.time()
    
    results = evaluator.comprehensive_evaluation(
        test_loader=val_loader,
        num_samples_eval=args.num_samples or config["evaluation"]["num_samples"],
        sampling_methods=["ddim", "ddpm"] if args.all_methods else [args.method or "ddim"],
        sampling_steps=[50] if not args.all_methods else [10, 20, 50, 100]
    )
    
    eval_time = time.time() - start_time
    print(f"Evaluation completed in {eval_time:.2f} seconds")
    
    # Save results
    results_path = Path(output_dir) / "logs" / "evaluation_results.json"
    evaluator.save_results(results, str(results_path))
    
    # Print summary
    print("\n📊 Evaluation Results Summary:")
    print("=" * 50)
    
    for method, metrics in results.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            print(f"\n{method.upper()}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                    
    print(f"\n✅ Detailed results saved: {results_path}")


def viz_curves_command(args):
    """Visualize training curves"""
    print(f"📈 Plotting training curves from: {args.log_dir}")
    
    output_dir = args.output_dir or "outputs"
    log_dir = Path(args.log_dir or Path(output_dir) / "logs")
    
    # Look for metrics file
    metrics_file = log_dir / "training_metrics.json"
    if not metrics_file.exists():
        print(f"Error: Metrics file not found: {metrics_file}")
        return
        
    # Plot curves
    curves_dir = Path(output_dir) / "curves"
    plots_saved = plot_training_curves(
        metrics_file=str(metrics_file),
        save_dir=str(curves_dir),
        metrics_to_plot=args.metrics
    )
    
    print(f"✅ Training curves saved to: {curves_dir}")
    for plot_name, plot_path in plots_saved.items():
        print(f"  - {plot_name}: {plot_path}")


def viz_schedules_command(args):
    """Visualize noise schedules"""
    print(f"📉 Plotting noise schedules from config: {args.config}")
    
    config = load_config(args.config)
    device = get_device("cpu")  # Use CPU for plotting
    
    schedules = create_schedules_from_config(config, device)
    
    output_dir = args.output_dir or config["output"]["base_dir"]
    save_path = Path(output_dir) / "curves" / "schedules.png"
    
    plot_path = plot_schedule_comparison(schedules, str(save_path))
    
    print(f"✅ Schedule plots saved: {plot_path}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="DDPM Training and Evaluation CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python -m src.cli train --config configs/mnist.yaml

  # Generate samples
  python -m src.cli sample.grid --config configs/mnist.yaml --num_samples 64

  # Create trajectory animation
  python -m src.cli sample.traj --config configs/mnist.yaml

  # Evaluate model
  python -m src.cli eval --config configs/mnist.yaml

  # Plot training curves
  python -m src.cli viz.curves --log_dir outputs/logs

  # Plot schedules
  python -m src.cli viz.schedules --config configs/mnist.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    common_parser.add_argument('--device', type=str, help='Device to use (cpu, cuda, mps, auto)')
    common_parser.add_argument('--output_dir', type=str, help='Output directory')
    common_parser.add_argument('--seed', type=int, help='Random seed')
    
    # Train command
    train_parser = subparsers.add_parser('train', parents=[common_parser], help='Train DDPM model')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, help='Batch size')
    train_parser.add_argument('--learning_rate', type=float, help='Learning rate')
    train_parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    # Sample grid command
    sample_grid_parser = subparsers.add_parser('sample.grid', parents=[common_parser], 
                                              help='Generate sample grid')
    sample_grid_parser.add_argument('--num_samples', type=int, help='Number of samples to generate')
    sample_grid_parser.add_argument('--method', type=str, choices=['ddpm', 'ddim'], help='Sampling method')
    sample_grid_parser.add_argument('--num_steps', type=int, help='Number of sampling steps')
    sample_grid_parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
    sample_grid_parser.add_argument('--output_name', type=str, help='Output filename')
    
    # Sample trajectory command
    sample_traj_parser = subparsers.add_parser('sample.traj', parents=[common_parser],
                                              help='Generate trajectory animation')
    sample_traj_parser.add_argument('--method', type=str, choices=['ddpm', 'ddim'], help='Sampling method')
    sample_traj_parser.add_argument('--num_steps', type=int, help='Number of sampling steps')
    sample_traj_parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
    sample_traj_parser.add_argument('--output_name', type=str, help='Output filename')
    sample_traj_parser.add_argument('--fps', type=int, default=10, help='Animation FPS')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', parents=[common_parser], help='Evaluate model')
    eval_parser.add_argument('--num_samples', type=int, help='Number of samples for evaluation')
    eval_parser.add_argument('--checkpoint', type=str, help='Checkpoint path')
    eval_parser.add_argument('--method', type=str, choices=['ddpm', 'ddim'], help='Sampling method')
    eval_parser.add_argument('--all_methods', action='store_true', help='Evaluate all methods and step counts')
    
    # Visualization commands
    viz_curves_parser = subparsers.add_parser('viz.curves', help='Plot training curves')
    viz_curves_parser.add_argument('--log_dir', type=str, help='Log directory')
    viz_curves_parser.add_argument('--output_dir', type=str, help='Output directory')
    viz_curves_parser.add_argument('--metrics', type=str, nargs='*', help='Specific metrics to plot')
    
    viz_schedules_parser = subparsers.add_parser('viz.schedules', parents=[common_parser],
                                               help='Plot noise schedules')
    
    # Parse arguments and run command
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
        
    # Set seed if provided
    if hasattr(args, 'seed') and args.seed is not None:
        set_seed(args.seed)
        
    # Route to appropriate command
    if args.command == 'train':
        train_command(args)
    elif args.command == 'sample.grid':
        sample_grid_command(args)
    elif args.command == 'sample.traj':
        sample_trajectory_command(args)
    elif args.command == 'eval':
        eval_command(args)
    elif args.command == 'viz.curves':
        viz_curves_command(args)
    elif args.command == 'viz.schedules':
        viz_schedules_command(args)
    else:
        parser.print_help()


# Entry points for setup.py
def train_command_entry():
    """Entry point for ddpm-train command"""
    sys.argv[0] = 'ddmp-train'
    sys.argv.insert(1, 'train')
    main()


def sample_command_entry():
    """Entry point for ddpm-sample command"""
    sys.argv[0] = 'ddpm-sample'
    sys.argv.insert(1, 'sample.grid')
    main()


def eval_command_entry():
    """Entry point for ddpm-eval command"""
    sys.argv[0] = 'ddpm-eval'
    sys.argv.insert(1, 'eval')
    main()


if __name__ == '__main__':
    main()