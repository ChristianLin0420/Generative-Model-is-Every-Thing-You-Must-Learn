"""
Command-line interface for VAE training, evaluation, and visualization.
Provides unified entry points for all VAE operations.
"""

import os
import sys
import argparse
from typing import Optional, List
from omegaconf import OmegaConf, DictConfig
from rich.console import Console
from rich.table import Table

from .train import train_vae
from .eval import run_full_evaluation, save_metrics_to_csv
from .sample import run_sampling
from .visualize import create_comprehensive_visualization_report

console = Console()


def load_config(config_path: str) -> DictConfig:
    """Load configuration file with validation."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    config = OmegaConf.load(config_path)
    return config


def get_latest_checkpoint(ckpt_dir: str, checkpoint_type: str = "best") -> str:
    """Get the latest checkpoint path."""
    if checkpoint_type == "best":
        ckpt_path = os.path.join(ckpt_dir, "best.pt")
    elif checkpoint_type == "latest":
        # Find the latest epoch checkpoint
        import glob
        ckpt_files = glob.glob(os.path.join(ckpt_dir, "epoch_*.pt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        
        # Extract epoch numbers and find the latest
        epoch_numbers = []
        for f in ckpt_files:
            try:
                epoch_num = int(os.path.basename(f).split('_')[1].split('.')[0])
                epoch_numbers.append((epoch_num, f))
            except (ValueError, IndexError):
                continue
        
        if not epoch_numbers:
            raise ValueError(f"No valid epoch checkpoints found in {ckpt_dir}")
        
        epoch_numbers.sort(key=lambda x: x[0], reverse=True)
        ckpt_path = epoch_numbers[0][1]
    else:
        ckpt_path = os.path.join(ckpt_dir, f"{checkpoint_type}.pt")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    return ckpt_path


def cmd_train(args) -> None:
    """Train VAE model."""
    console.print("[bold blue]Starting VAE Training[/bold blue]")
    
    config = load_config(args.config)
    console.print(f"Loaded configuration: {args.config}")
    console.print(f"Dataset: {config.data.dataset}")
    console.print(f"Model: {config.model.name}")
    console.print(f"Epochs: {config.train.epochs}")
    
    try:
        train_vae(config)
        console.print("[bold green]Training completed successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        sys.exit(1)


def cmd_eval(args) -> None:
    """Evaluate trained VAE model."""
    console.print("[bold blue]Starting VAE Evaluation[/bold blue]")
    
    config = load_config(args.config)
    
    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        ckpt_dir = os.path.join(config.log.out_dir, "ckpts")
        checkpoint_path = get_latest_checkpoint(ckpt_dir, "best")
    
    console.print(f"Using checkpoint: {checkpoint_path}")
    
    try:
        # Run evaluation
        metrics = run_full_evaluation(config, checkpoint_path)
        
        # Create metrics table
        table = Table(title="VAE Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for metric, value in metrics.items():
            table.add_row(metric, f"{value:.4f}")
        
        console.print(table)
        
        # Save metrics
        output_path = os.path.join(config.log.out_dir, "logs", "metrics.csv")
        save_metrics_to_csv(metrics, output_path)
        console.print(f"Metrics saved to: {output_path}")
        
        console.print("[bold green]Evaluation completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        sys.exit(1)


def cmd_sample_prior(args) -> None:
    """Generate samples from prior distribution."""
    console.print("[bold blue]Generating samples from prior[/bold blue]")
    
    config = load_config(args.config)
    
    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        ckpt_dir = os.path.join(config.log.out_dir, "ckpts")
        checkpoint_path = get_latest_checkpoint(ckpt_dir, "best")
    
    output_dir = args.output_dir or config.log.out_dir
    
    try:
        run_sampling(config, checkpoint_path, output_dir)
        console.print(f"[bold green]Samples generated and saved to: {output_dir}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Sampling failed: {e}[/bold red]")
        sys.exit(1)


def cmd_sample_interpolate(args) -> None:
    """Generate interpolation samples."""
    console.print("[bold blue]Generating interpolation samples[/bold blue]")
    
    config = load_config(args.config)
    
    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        ckpt_dir = os.path.join(config.log.out_dir, "ckpts")
        checkpoint_path = get_latest_checkpoint(ckpt_dir, "best")
    
    output_dir = args.output_dir or config.log.out_dir
    
    try:
        # This functionality is part of run_sampling
        run_sampling(config, checkpoint_path, output_dir)
        console.print(f"[bold green]Interpolations generated and saved to: {output_dir}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Interpolation generation failed: {e}[/bold red]")
        sys.exit(1)


def cmd_viz_recon_grid(args) -> None:
    """Generate reconstruction grid."""
    console.print("[bold blue]Creating reconstruction grid[/bold blue]")
    
    config = load_config(args.config)
    
    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        ckpt_dir = os.path.join(config.log.out_dir, "ckpts")
        checkpoint_path = get_latest_checkpoint(ckpt_dir, "best")
    
    output_dir = args.output_dir or config.log.out_dir
    
    try:
        create_comprehensive_visualization_report(config, checkpoint_path, output_dir)
        console.print(f"[bold green]Visualization created and saved to: {output_dir}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Visualization failed: {e}[/bold red]")
        sys.exit(1)


def cmd_viz_traverse(args) -> None:
    """Generate latent traversal visualizations."""
    console.print("[bold blue]Creating latent traversals[/bold blue]")
    
    config = load_config(args.config)
    
    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        ckpt_dir = os.path.join(config.log.out_dir, "ckpts")
        checkpoint_path = get_latest_checkpoint(ckpt_dir, "best")
    
    output_dir = args.output_dir or config.log.out_dir
    
    try:
        create_comprehensive_visualization_report(config, checkpoint_path, output_dir)
        console.print(f"[bold green]Latent traversals created and saved to: {output_dir}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]Latent traversal creation failed: {e}[/bold red]")
        sys.exit(1)


def cmd_viz_latent_scatter(args) -> None:
    """Generate 2D latent scatter plot."""
    console.print("[bold blue]Creating 2D latent scatter plot[/bold blue]")
    
    config = load_config(args.config)
    
    # Get checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        ckpt_dir = os.path.join(config.log.out_dir, "ckpts")
        checkpoint_path = get_latest_checkpoint(ckpt_dir, "best")
    
    output_dir = args.output_dir or config.log.out_dir
    
    try:
        create_comprehensive_visualization_report(config, checkpoint_path, output_dir)
        console.print(f"[bold green]2D scatter plot created and saved to: {output_dir}[/bold green]")
    except Exception as e:
        console.print(f"[bold red]2D scatter plot creation failed: {e}[/bold red]")
        sys.exit(1)


def cmd_compare_dae(args) -> None:
    """Compare VAE with DAE from Day 2."""
    console.print("[bold blue]Comparing VAE with DAE[/bold blue]")
    
    if not args.dae_ckpt:
        console.print("[bold red]DAE checkpoint path is required for comparison[/bold red]")
        sys.exit(1)
    
    config = load_config(args.config)
    
    # Get VAE checkpoint path
    if args.checkpoint:
        vae_checkpoint_path = args.checkpoint
    else:
        ckpt_dir = os.path.join(config.log.out_dir, "ckpts")
        vae_checkpoint_path = get_latest_checkpoint(ckpt_dir, "best")
    
    try:
        # Import and run DAE comparison
        from .compare_dae import run_vae_dae_comparison
        
        output_dir = args.output_dir or config.log.out_dir
        run_vae_dae_comparison(config, vae_checkpoint_path, args.dae_ckpt, output_dir)
        
        console.print(f"[bold green]VAE vs DAE comparison completed and saved to: {output_dir}[/bold green]")
    except ImportError:
        console.print("[bold yellow]DAE comparison module not yet implemented[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Comparison failed: {e}[/bold red]")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="VAE Training, Evaluation, and Visualization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train VAE
  python -m src.cli train --config configs/mnist.yaml

  # Evaluate trained model
  python -m src.cli eval --config configs/mnist.yaml

  # Generate samples
  python -m src.cli sample.prior --config configs/mnist.yaml

  # Create visualizations
  python -m src.cli viz.recon_grid --config configs/mnist.yaml
  python -m src.cli viz.traverse --config configs/mnist.yaml

  # Compare with DAE
  python -m src.cli compare.dae --config configs/mnist.yaml --dae_ckpt path/to/dae.pt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train VAE model')
    train_parser.add_argument('--config', required=True, help='Path to configuration file')
    train_parser.set_defaults(func=cmd_train)
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained VAE model')
    eval_parser.add_argument('--config', required=True, help='Path to configuration file')
    eval_parser.add_argument('--checkpoint', help='Path to checkpoint file (default: best.pt)')
    eval_parser.set_defaults(func=cmd_eval)
    
    # Sampling commands
    sample_parser = subparsers.add_parser('sample', help='Sampling commands')
    sample_subparsers = sample_parser.add_subparsers(dest='sample_command')
    
    # Prior sampling
    prior_parser = sample_subparsers.add_parser('prior', help='Sample from prior distribution')
    prior_parser.add_argument('--config', required=True, help='Path to configuration file')
    prior_parser.add_argument('--checkpoint', help='Path to checkpoint file')
    prior_parser.add_argument('--output_dir', help='Output directory')
    prior_parser.set_defaults(func=cmd_sample_prior)
    
    # Interpolation sampling
    interp_parser = sample_subparsers.add_parser('interpolate', help='Generate interpolations')
    interp_parser.add_argument('--config', required=True, help='Path to configuration file')
    interp_parser.add_argument('--checkpoint', help='Path to checkpoint file')
    interp_parser.add_argument('--output_dir', help='Output directory')
    interp_parser.set_defaults(func=cmd_sample_interpolate)
    
    # Visualization commands
    viz_parser = subparsers.add_parser('viz', help='Visualization commands')
    viz_subparsers = viz_parser.add_subparsers(dest='viz_command')
    
    # Reconstruction grid
    recon_parser = viz_subparsers.add_parser('recon_grid', help='Create reconstruction grid')
    recon_parser.add_argument('--config', required=True, help='Path to configuration file')
    recon_parser.add_argument('--checkpoint', help='Path to checkpoint file')
    recon_parser.add_argument('--output_dir', help='Output directory')
    recon_parser.set_defaults(func=cmd_viz_recon_grid)
    
    # Latent traversal
    traverse_parser = viz_subparsers.add_parser('traverse', help='Create latent traversals')
    traverse_parser.add_argument('--config', required=True, help='Path to configuration file')
    traverse_parser.add_argument('--checkpoint', help='Path to checkpoint file')
    traverse_parser.add_argument('--dims', type=int, nargs='*', help='Dimensions to traverse')
    traverse_parser.add_argument('--output_dir', help='Output directory')
    traverse_parser.set_defaults(func=cmd_viz_traverse)
    
    # 2D latent scatter
    scatter_parser = viz_subparsers.add_parser('latent_scatter', help='Create 2D latent scatter plot')
    scatter_parser.add_argument('--config', required=True, help='Path to configuration file')
    scatter_parser.add_argument('--checkpoint', help='Path to checkpoint file')
    scatter_parser.add_argument('--output_dir', help='Output directory')
    scatter_parser.set_defaults(func=cmd_viz_latent_scatter)
    
    # Comparison commands
    compare_parser = subparsers.add_parser('compare', help='Comparison commands')
    compare_subparsers = compare_parser.add_subparsers(dest='compare_command')
    
    # DAE comparison
    dae_parser = compare_subparsers.add_parser('dae', help='Compare with DAE')
    dae_parser.add_argument('--config', required=True, help='Path to VAE configuration file')
    dae_parser.add_argument('--checkpoint', help='Path to VAE checkpoint file')
    dae_parser.add_argument('--dae_ckpt', required=True, help='Path to DAE checkpoint file')
    dae_parser.add_argument('--output_dir', help='Output directory')
    dae_parser.set_defaults(func=cmd_compare_dae)
    
    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not hasattr(args, 'func'):
        parser.print_help()
        return
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        console.print("\n[bold red]Operation cancelled by user[/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {e}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()