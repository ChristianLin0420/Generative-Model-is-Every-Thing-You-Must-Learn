"""Command Line Interface for DDPM Day 5

Provides entry points for:
- Training DDPM models
- Sampling (trajectories and grids)
- Evaluation and comparison
- Visualization generation
"""

import argparse
import torch
import os
from typing import Dict, Any
import yaml
from omegaconf import OmegaConf

from .utils import set_seed, get_device, load_checkpoint, count_parameters
from .ddpm_schedules import DDPMScheduler
from .dataset import create_dataset_from_config
from .models.unet_tiny import UNetTiny
from .trainer import DDPMTrainer
from .sampler import DDPMSampler, DDIMSampler
from .eval import DDPMEvaluator, save_evaluation_results
from .visualize import (
    create_reverse_trajectory_grid, 
    create_reverse_animation,
    create_forward_vs_reverse_panel,
    visualize_noise_schedule
)
from .compare_forward_reverse import generate_comparison_report


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_model_from_config(config: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Create model from configuration."""
    model_config = config["model"]
    dataset_config = config["dataset"]
    
    # Get model parameters
    model_type = model_config.get("type", "unet_tiny")
    
    if model_type == "unet_tiny":
        model = UNetTiny(
            in_channels=dataset_config["channels"],
            out_channels=dataset_config["channels"],
            model_channels=model_config.get("model_channels", 64),
            num_res_blocks=model_config.get("num_res_blocks", 2),
            channel_mult=model_config.get("channel_mult", [1, 2, 2]),
            attention_resolutions=model_config.get("attention_resolutions", [16]),
            num_heads=model_config.get("num_heads", 4),
            time_embed_dim=model_config.get("time_embed_dim", None),
            dropout=model_config.get("dropout", 0.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    return model


def create_scheduler_from_config(config: Dict[str, Any], device: torch.device) -> DDPMScheduler:
    """Create scheduler from configuration."""
    ddpm_config = config["ddpm"]
    
    scheduler = DDPMScheduler(
        num_timesteps=ddpm_config.get("num_timesteps", 1000),
        beta_schedule=ddpm_config.get("beta_schedule", "linear"),
        beta_start=ddpm_config.get("beta_start", 0.0001),
        beta_end=ddpm_config.get("beta_end", 0.02),
        prediction_type=ddpm_config.get("prediction_type", "epsilon")
    )
    
    scheduler = scheduler.to(device)
    return scheduler


def train_command(args):
    """Train DDPM model."""
    print(f"Training DDPM model with config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    
    # Set up device and seed
    device = get_device() if config.get("device") == "auto" else torch.device(config.get("device", "cpu"))
    set_seed(config.get("seed", 42))
    
    print(f"Using device: {device}")
    
    # Create datasets
    train_dataset, test_dataset, train_loader, test_loader = create_dataset_from_config(config)
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model and scheduler
    model = create_model_from_config(config, device)
    scheduler = create_scheduler_from_config(config, device)
    
    # Print model info
    param_info = count_parameters(model)
    print(f"Model parameters: {param_info['total']:,} total, {param_info['trainable']:,} trainable")
    
    # Create trainer
    trainer = DDPMTrainer(
        model=model,
        scheduler=scheduler,
        train_dataloader=train_loader,
        val_dataloader=test_loader,
        device=device,
        config=config
    )
    
    # Start training
    trainer.train()
    
    print("Training completed!")


def sample_trajectories_command(args):
    """Generate sampling trajectories."""
    print(f"Generating sampling trajectories from checkpoint: {args.ckpt}")
    
    # Load configuration from checkpoint
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    config = checkpoint.get("config", {})
    
    device = get_device()
    
    # Create model and scheduler
    model = create_model_from_config(config, device)
    scheduler = create_scheduler_from_config(config, device)
    
    # Load checkpoint
    load_checkpoint(args.ckpt, model, device=device)
    
    print(f"Loaded model from {args.ckpt}")
    
    # Generate trajectories
    output_path = args.output or "outputs/grids/reverse_traj_grid.png"
    
    grid = create_reverse_trajectory_grid(
        model=model,
        scheduler=scheduler,
        num_samples=args.num_samples,
        num_timesteps_to_show=8,
        image_size=(config["dataset"]["image_size"], config["dataset"]["image_size"]),
        channels=config["dataset"]["channels"],
        device=device,
        sampler_type=args.sampler,
        save_path=output_path
    )
    
    print(f"Saved trajectory grid to {output_path}")
    
    # Also create animation
    if args.animation:
        anim_path = output_path.replace(".png", ".gif")
        
        frames = create_reverse_animation(
            model=model,
            scheduler=scheduler,
            num_samples=min(args.num_samples, 4),  # Fewer for animation
            image_size=(config["dataset"]["image_size"], config["dataset"]["image_size"]),
            channels=config["dataset"]["channels"],
            device=device,
            save_path=anim_path,
            sampler_type=args.sampler
        )
        
        print(f"Saved animation to {anim_path}")


def sample_grid_command(args):
    """Generate grid of samples."""
    print(f"Generating sample grid from checkpoint: {args.ckpt}")
    
    # Load configuration from checkpoint
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    config = checkpoint.get("config", {})
    
    device = get_device()
    
    # Create model and scheduler
    model = create_model_from_config(config, device)
    scheduler = create_scheduler_from_config(config, device)
    
    # Load checkpoint
    load_checkpoint(args.ckpt, model, device=device)
    
    print(f"Loaded model from {args.ckpt}")
    
    # Generate samples
    model.eval()
    
    if args.sampler == "ddpm":
        sampler = DDPMSampler(scheduler)
        result = sampler.p_sample_loop(
            model=model,
            shape=(args.num_samples, config["dataset"]["channels"], 
                   config["dataset"]["image_size"], config["dataset"]["image_size"]),
            device=device,
            progress=True
        )
    else:  # DDIM
        sampler = DDIMSampler(scheduler)
        result = sampler.ddim_sample(
            model=model,
            shape=(args.num_samples, config["dataset"]["channels"], 
                   config["dataset"]["image_size"], config["dataset"]["image_size"]),
            num_inference_steps=args.steps,
            device=device,
            progress=True
        )
    
    samples = result["images"]
    
    # Save samples
    from .utils import save_image_grid
    
    output_path = args.output or "outputs/samples/samples_grid.png"
    save_image_grid(samples, output_path, nrow=8, normalize=True)
    
    print(f"Saved {args.num_samples} samples to {output_path}")


def visualize_compare_command(args):
    """Create forward vs reverse comparison visualization."""
    print(f"Creating forward vs reverse comparison from checkpoint: {args.ckpt}")
    
    # Load configuration from checkpoint
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    config = checkpoint.get("config", {})
    
    device = get_device()
    
    # Create model and scheduler
    model = create_model_from_config(config, device)
    scheduler = create_scheduler_from_config(config, device)
    
    # Load checkpoint
    load_checkpoint(args.ckpt, model, device=device)
    
    # Create test dataset
    _, test_dataset, _, test_loader = create_dataset_from_config(config)
    
    # Get some test images
    test_batch = next(iter(test_loader))
    if isinstance(test_batch, (list, tuple)):
        test_images = test_batch[0][:4]  # Use first 4 images
    else:
        test_images = test_batch[:4]
    
    print(f"Using {test_images.shape[0]} test images for comparison")
    
    # Create forward vs reverse panel
    output_path = args.output or "outputs/grids/forward_vs_reverse.png"
    
    panel = create_forward_vs_reverse_panel(
        model=model,
        scheduler=scheduler,
        x_start=test_images,
        timesteps_to_show=[10, 25, 50, 75, 90],
        device=device,
        save_path=output_path
    )
    
    print(f"Saved comparison panel to {output_path}")
    
    # Generate comprehensive report
    if args.report:
        report_dir = args.output.replace(".png", "_report") if args.output else "outputs/reports"
        report_path = generate_comparison_report(
            model=model,
            scheduler=scheduler,
            test_images=test_images,
            output_dir=report_dir,
            dataset_name=config["dataset"]["name"]
        )
        
        print(f"Generated comparison report: {report_path}")


def evaluate_command(args):
    """Run evaluation metrics."""
    print(f"Evaluating model from checkpoint: {args.ckpt}")
    
    # Load configuration from checkpoint
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    config = checkpoint.get("config", {})
    
    device = get_device()
    
    # Create model and scheduler
    model = create_model_from_config(config, device)
    scheduler = create_scheduler_from_config(config, device)
    
    # Load checkpoint
    load_checkpoint(args.ckpt, model, device=device)
    
    # Create test dataset
    _, test_dataset, _, test_loader = create_dataset_from_config(config)
    
    print(f"Evaluating on {len(test_dataset)} test samples")
    
    # Create evaluator
    evaluator = DDPMEvaluator(
        model=model,
        scheduler=scheduler,
        device=device,
        metrics=['psnr', 'ssim', 'lpips']
    )
    
    # Run reconstruction evaluation
    eval_config = config.get("eval", {})
    timesteps_to_eval = eval_config.get("timesteps_to_eval", [10, 25, 50, 75, 99])
    
    print("Running reconstruction evaluation...")
    recon_results = evaluator.evaluate_reconstruction(
        dataloader=test_loader,
        timesteps_to_eval=timesteps_to_eval,
        num_samples=args.num_samples,
        sampler_type=args.sampler
    )
    
    # Run sample quality evaluation
    print("Running sample quality evaluation...")
    quality_results = evaluator.evaluate_sample_quality(
        num_samples=min(args.num_samples, 1000),  # Limit for speed
        sampler_type=args.sampler
    )
    
    # Combine results
    all_results = {
        "reconstruction": recon_results,
        "quality": quality_results
    }
    
    # Save results
    output_path = args.output or "outputs/reports/evaluation_results"
    save_evaluation_results(all_results, output_path)
    
    # Print summary
    print("\nEvaluation Summary:")
    print("=" * 50)
    
    if "reconstruction" in all_results:
        print("Reconstruction Metrics:")
        for metric in recon_results["metrics"]:
            print(f"  {metric.upper()}:")
            for t in timesteps_to_eval[:3]:  # Show first 3 timesteps
                if isinstance(recon_results["metrics"][metric][t], dict):
                    mean_val = recon_results["metrics"][metric][t]["mean"]
                    std_val = recon_results["metrics"][metric][t]["std"]
                    print(f"    t={t:3d}: {mean_val:.4f} ± {std_val:.4f}")
    
    if "quality" in all_results:
        print("\nSample Quality:")
        for metric, value in quality_results.items():
            print(f"  {metric}: {value:.4f}")
    
    print(f"\nFull results saved to {output_path}_*.csv and {output_path}_summary.txt")


def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        description="DDPM Day 5: Reverse Process Intuition",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train DDPM model')
    train_parser.add_argument('--config', type=str, required=True, 
                             help='Path to config file')
    train_parser.add_argument('--epochs', type=int, help='Override epochs')
    train_parser.add_argument('--batch-size', type=int, help='Override batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Override learning rate')
    
    # Sample trajectories command
    traj_parser = subparsers.add_parser('sample.traj', help='Generate sampling trajectories')
    traj_parser.add_argument('--ckpt', type=str, required=True,
                            help='Path to model checkpoint')
    traj_parser.add_argument('--num-samples', type=int, default=4,
                            help='Number of samples to generate')
    traj_parser.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddpm',
                            help='Sampling method')
    traj_parser.add_argument('--output', type=str, 
                            help='Output path (default: outputs/grids/reverse_traj_grid.png)')
    traj_parser.add_argument('--animation', action='store_true',
                            help='Also create animation')
    
    # Sample grid command
    grid_parser = subparsers.add_parser('sample.grid', help='Generate grid of samples')
    grid_parser.add_argument('--ckpt', type=str, required=True,
                            help='Path to model checkpoint')
    grid_parser.add_argument('--num-samples', type=int, default=64,
                            help='Number of samples to generate')
    grid_parser.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddpm',
                            help='Sampling method')
    grid_parser.add_argument('--steps', type=int, default=50,
                            help='Number of sampling steps (for DDIM)')
    grid_parser.add_argument('--output', type=str,
                            help='Output path (default: outputs/samples/samples_grid.png)')
    
    # Visualization compare command
    viz_parser = subparsers.add_parser('viz.compare', help='Create forward vs reverse comparison')
    viz_parser.add_argument('--ckpt', type=str, required=True,
                           help='Path to model checkpoint')
    viz_parser.add_argument('--output', type=str,
                           help='Output path (default: outputs/grids/forward_vs_reverse.png)')
    viz_parser.add_argument('--report', action='store_true',
                           help='Generate comprehensive comparison report')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Run evaluation metrics')
    eval_parser.add_argument('--ckpt', type=str, required=True,
                            help='Path to model checkpoint')
    eval_parser.add_argument('--num-samples', type=int, default=100,
                            help='Number of samples for evaluation')
    eval_parser.add_argument('--sampler', choices=['ddpm', 'ddim'], default='ddpm',
                            help='Sampling method')
    eval_parser.add_argument('--output', type=str,
                            help='Output path (default: outputs/reports/evaluation_results)')
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'sample.traj':
            sample_trajectories_command(args)
        elif args.command == 'sample.grid':
            sample_grid_command(args)
        elif args.command == 'viz.compare':
            visualize_compare_command(args)
        elif args.command == 'eval':
            evaluate_command(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())