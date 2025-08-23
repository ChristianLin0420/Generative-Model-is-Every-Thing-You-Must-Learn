"""Command-line interface for forward diffusion experiments."""

import argparse
import torch
from pathlib import Path
from omegaconf import OmegaConf
from rich.console import Console
from rich.progress import track
import sys

from .utils import set_seed, get_device
from .dataset import get_dataloader_from_config, get_sample_batch
from .ddpm_schedules import get_ddpm_schedule
from .forward import sample_trajectory
from .stats import (
    compute_forward_stats, save_stats_csv, compute_schedule_comparison,
    print_schedule_summary
)
from .visualize import (
    plot_schedules, plot_snr_analysis, create_trajectory_grid,
    create_trajectory_animation, plot_pixel_histograms,
    plot_mse_and_kl_curves
)

console = Console()


def load_config(config_path: str):
    """Load configuration from YAML file."""
    try:
        config = OmegaConf.load(config_path)
        return config
    except Exception as e:
        console.print(f"[red]Error loading config {config_path}: {e}[/red]")
        sys.exit(1)


def setup_experiment(config):
    """Set up experiment with config parameters."""
    set_seed(config.seed)
    device = get_device(config.device)
    
    console.print(f"[green]Using device: {device}[/green]")
    console.print(f"[green]Dataset: {config.data.dataset}[/green]")
    console.print(f"[green]Diffusion steps: {config.diffusion.T}[/green]")
    console.print(f"[green]Schedule: {config.diffusion.schedule}[/green]")
    
    return device


def cmd_trajectory_grid(args):
    """Generate trajectory grid visualization."""
    config = load_config(args.config)
    device = setup_experiment(config)
    
    console.print("[blue]Generating trajectory grid...[/blue]")
    
    # Get sample batch
    x0_batch, _ = get_sample_batch(
        config.data.dataset,
        config.data.root,
        batch_size=16,
        normalize_mode=config.data.normalize
    )
    x0_batch = x0_batch.to(device)
    
    # Get diffusion schedule
    betas, alphas, alpha_bars = get_ddpm_schedule(
        config.diffusion.T,
        config.diffusion.schedule
    )
    alpha_bars = alpha_bars.to(device)
    
    # Create output directory
    output_dir = Path(config.log.out_dir) / "grids"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate grid
    save_path = output_dir / f"{config.data.dataset}_traj_grid.png"
    trajectory = create_trajectory_grid(
        x0_batch,
        config.diffusion.timesteps_to_show,
        alpha_bars,
        save_path=save_path,
        normalize=(config.data.normalize == "minus_one_one"),
        title=f"Forward Diffusion Trajectory - {config.data.dataset.upper()}"
    )
    
    console.print(f"[green]Saved trajectory grid to {save_path}[/green]")


def cmd_trajectory_animate(args):
    """Generate trajectory animation."""
    config = load_config(args.config)
    device = setup_experiment(config)
    
    console.print("[blue]Generating trajectory animation...[/blue]")
    
    # Get single image
    x0_batch, _ = get_sample_batch(
        config.data.dataset,
        config.data.root,
        batch_size=1,
        normalize_mode=config.data.normalize
    )
    x0_image = x0_batch.to(device)
    
    # Get diffusion schedule
    betas, alphas, alpha_bars = get_ddpm_schedule(
        config.diffusion.T,
        config.diffusion.schedule
    )
    alpha_bars = alpha_bars.to(device)
    
    # Create output directory
    output_dir = Path(config.log.out_dir) / "animations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate animation
    idx = getattr(args, 'idx', 0)
    save_path = output_dir / f"sample_{idx:03d}.gif"
    
    trajectory = create_trajectory_animation(
        x0_image,
        config.diffusion.T,
        alpha_bars,
        save_path=save_path,
        normalize=(config.data.normalize == "minus_one_one"),
        duration=0.05  # Faster animation
    )
    
    console.print(f"[green]Saved animation to {save_path}[/green]")


def cmd_plot_schedules(args):
    """Plot noise schedules comparison."""
    config = load_config(args.config)
    device = setup_experiment(config)
    
    console.print("[blue]Plotting schedules...[/blue]")
    
    # Compare different schedules
    schedules = getattr(args, 'schedules', ['linear', 'cosine', 'sigmoid'])
    if isinstance(schedules, str):
        schedules = [schedules]
    
    schedules_data = {}
    for schedule_name in schedules:
        console.print(f"Computing {schedule_name} schedule...")
        betas, alphas, alpha_bars = get_ddpm_schedule(config.diffusion.T, schedule_name)
        
        schedules_data[schedule_name] = {
            'timesteps': torch.arange(config.diffusion.T),
            'betas': betas,
            'alpha_bars': alpha_bars,
            'snr_db': torch.log10(alpha_bars / (1.0 - alpha_bars)) * 10.0
        }
    
    # Create output directory
    output_dir = Path(config.log.out_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot schedules
    save_path = output_dir / "beta_alpha_snr.png"
    plot_schedules(schedules_data, save_path=save_path)
    
    console.print(f"[green]Saved schedule plots to {save_path}[/green]")


def cmd_plot_snr(args):
    """Plot SNR analysis."""
    config = load_config(args.config)
    device = setup_experiment(config)
    
    console.print("[blue]Plotting SNR analysis...[/blue]")
    
    # Get sample for statistics computation
    x0_batch, _ = get_sample_batch(
        config.data.dataset,
        config.data.root,
        batch_size=64,
        normalize_mode=config.data.normalize
    )
    
    # Compare schedules with statistics
    schedules_comparison = compute_schedule_comparison(
        x0_batch, config.diffusion.T, device
    )
    
    # Create output directory  
    output_dir = Path(config.log.out_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot SNR analysis
    save_path = output_dir / "snr_analysis.png"
    plot_snr_analysis(schedules_comparison, save_path=save_path)
    
    # Print summaries
    for schedule_name, stats in schedules_comparison.items():
        print_schedule_summary(stats)
    
    console.print(f"[green]Saved SNR analysis to {save_path}[/green]")


def cmd_plot_histograms(args):
    """Plot pixel histograms at different timesteps."""
    config = load_config(args.config)
    device = setup_experiment(config)
    
    console.print("[blue]Plotting pixel histograms...[/blue]")
    
    # Get sample batch
    x0_batch, _ = get_sample_batch(
        config.data.dataset,
        config.data.root,
        batch_size=1000,  # More samples for better histograms
        normalize_mode=config.data.normalize
    )
    x0_batch = x0_batch.to(device)
    
    # Get diffusion schedule
    betas, alphas, alpha_bars = get_ddpm_schedule(
        config.diffusion.T,
        config.diffusion.schedule
    )
    alpha_bars = alpha_bars.to(device)
    
    # Create output directory
    output_dir = Path(config.log.out_dir) / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot histograms
    save_path = output_dir / f"hist_t_{config.diffusion.schedule}.png"
    plot_pixel_histograms(
        x0_batch,
        config.diffusion.timesteps_to_show,
        alpha_bars,
        save_path=save_path
    )
    
    console.print(f"[green]Saved histograms to {save_path}[/green]")


def cmd_compute_stats(args):
    """Compute and save forward diffusion statistics."""
    config = load_config(args.config)
    device = setup_experiment(config)
    
    console.print("[blue]Computing forward diffusion statistics...[/blue]")
    
    # Get sample batch
    x0_batch, _ = get_sample_batch(
        config.data.dataset,
        config.data.root,
        batch_size=64,
        normalize_mode=config.data.normalize
    )
    x0_batch = x0_batch.to(device)
    
    # Get diffusion schedule
    betas, alphas, alpha_bars = get_ddpm_schedule(
        config.diffusion.T,
        config.diffusion.schedule
    )
    betas = betas.to(device)
    alpha_bars = alpha_bars.to(device)
    
    # Compute statistics
    stats = compute_forward_stats(x0_batch, betas, alpha_bars, device)
    
    # Create output directories
    logs_dir = Path(config.log.out_dir) / "logs"
    plots_dir = Path(config.log.out_dir) / "plots"
    logs_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save CSV
    csv_path = logs_dir / "forward_stats.csv"
    save_stats_csv(stats, csv_path)
    
    # Plot MSE and KL curves
    plot_path = plots_dir / "mse_kl_curves.png"
    plot_mse_and_kl_curves(stats, save_path=plot_path)
    
    console.print(f"[green]Saved statistics CSV to {csv_path}[/green]")
    console.print(f"[green]Saved MSE/KL plots to {plot_path}[/green]")


def cmd_run_all(args):
    """Run all experiments for a dataset."""
    config = load_config(args.config)
    
    console.print(f"[bold blue]Running all experiments for {config.data.dataset.upper()}...[/bold blue]")
    
    # Override args for each command
    class MockArgs:
        def __init__(self, config_path):
            self.config = config_path
            self.idx = 0
            self.schedules = ['linear', 'cosine', 'sigmoid']
    
    mock_args = MockArgs(args.config)
    
    # Run all commands
    commands = [
        ("Trajectory Grid", cmd_trajectory_grid),
        ("Trajectory Animation", cmd_trajectory_animate),
        ("Schedule Plots", cmd_plot_schedules),
        ("SNR Analysis", cmd_plot_snr),
        ("Pixel Histograms", cmd_plot_histograms),
        ("Statistics", cmd_compute_stats)
    ]
    
    for name, func in track(commands, description="Running experiments..."):
        console.print(f"[yellow]Running: {name}[/yellow]")
        try:
            func(mock_args)
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")
    
    console.print("[bold green]All experiments completed![/bold green]")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Forward Diffusion Process CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Trajectory grid command
    traj_grid_parser = subparsers.add_parser(
        'traj.grid',
        help='Generate trajectory grid visualization'
    )
    traj_grid_parser.add_argument('--config', required=True, help='Config file path')
    
    # Trajectory animation command
    traj_anim_parser = subparsers.add_parser(
        'traj.animate',
        help='Generate trajectory animation'
    )
    traj_anim_parser.add_argument('--config', required=True, help='Config file path')
    traj_anim_parser.add_argument('--idx', type=int, default=0, help='Sample index for animation')
    
    # Plot schedules command
    plot_sched_parser = subparsers.add_parser(
        'plots.schedules',
        help='Plot noise schedules'
    )
    plot_sched_parser.add_argument('--config', required=True, help='Config file path')
    plot_sched_parser.add_argument('--schedules', nargs='+', 
                                  default=['linear', 'cosine', 'sigmoid'],
                                  help='Schedules to compare')
    
    # Plot SNR command  
    plot_snr_parser = subparsers.add_parser(
        'plots.snr',
        help='Plot SNR analysis'
    )
    plot_snr_parser.add_argument('--config', required=True, help='Config file path')
    
    # Plot histograms command
    plot_hist_parser = subparsers.add_parser(
        'plots.hist',
        help='Plot pixel histograms'
    )
    plot_hist_parser.add_argument('--config', required=True, help='Config file path')
    
    # Compute stats command
    stats_parser = subparsers.add_parser(
        'compute.stats',
        help='Compute and save statistics'
    )
    stats_parser.add_argument('--config', required=True, help='Config file path')
    
    # Run all command
    all_parser = subparsers.add_parser(
        'run.all',
        help='Run all experiments'
    )
    all_parser.add_argument('--config', required=True, help='Config file path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Map commands to functions
    command_map = {
        'traj.grid': cmd_trajectory_grid,
        'traj.animate': cmd_trajectory_animate,
        'plots.schedules': cmd_plot_schedules,
        'plots.snr': cmd_plot_snr,
        'plots.hist': cmd_plot_histograms,
        'compute.stats': cmd_compute_stats,
        'run.all': cmd_run_all
    }
    
    if args.command in command_map:
        try:
            command_map[args.command](args)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            raise
    else:
        console.print(f"[red]Unknown command: {args.command}[/red]")
        parser.print_help()


if __name__ == "__main__":
    main()