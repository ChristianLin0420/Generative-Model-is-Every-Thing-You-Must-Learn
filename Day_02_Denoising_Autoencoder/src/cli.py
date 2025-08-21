"""
Command Line Interface for Day 2: Denoising Autoencoder
"""

import argparse
import sys
from pathlib import Path

from omegaconf import DictConfig

from .utils import console, get_device, load_config, set_seed


def train_command(args):
    """Train denoising autoencoder."""
    from .train import train_model
    
    console.print(f"[bold green]Starting DAE training with config: {args.config}[/bold green]")
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.epochs:
        config.train.epochs = args.epochs
    if args.lr:
        config.train.lr = args.lr
    if args.batch_size:
        config.data.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    # Train model
    final_metrics = train_model(config)
    
    if final_metrics:
        console.print(f"[green]Training completed! Best PSNR: {final_metrics.get('psnr', 0):.2f} dB[/green]")
    else:
        console.print("[yellow]Training completed (no metrics returned)[/yellow]")


def eval_command(args):
    """Evaluate trained model."""
    from .eval import evaluate_model
    
    console.print(f"[bold blue]Evaluating model with config: {args.config}[/bold blue]")
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line args
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.data.batch_size = args.batch_size
    
    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        # Look for best model
        ckpt_dir = Path(config.log.out_dir) / "ckpts"
        checkpoint_path = ckpt_dir / "best_model.pth"
        
        if not checkpoint_path.exists():
            # Look for latest checkpoint
            checkpoints = list(ckpt_dir.glob("checkpoint_epoch_*.pth"))
            if checkpoints:
                checkpoint_path = max(checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
            else:
                console.print("[red]No checkpoint found! Please train a model first.[/red]")
                return
    
    console.print(f"Using checkpoint: {checkpoint_path}")
    
    # Evaluate
    results = evaluate_model(config, str(checkpoint_path))
    
    if results:
        overall = results.get('overall', {})
        console.print(f"[green]Evaluation complete![/green]")
        console.print(f"[green]PSNR: {overall.get('psnr', 0):.2f} dB[/green]")
        console.print(f"[green]SSIM: {overall.get('ssim', 0):.4f}[/green]")


def viz_recon_grid_command(args):
    """Generate reconstruction grids."""
    from .dataset import get_dataset_loaders
    from .models import create_model
    from .visualize import create_reconstruction_grid
    
    console.print("[blue]Generating reconstruction grids[/blue]")
    
    # Load config and setup
    config = load_config(args.config)
    device = get_device(config.device)
    set_seed(config.seed)
    
    # Load data
    _, test_loader = get_dataset_loaders(
        dataset_name=config.data.dataset,
        root=config.data.root,
        batch_size=config.data.batch_size,
        num_workers=config.data.get('num_workers', 4),
        normalize=config.data.get('normalize', 'zero_one'),
        train_sigmas=config.noise.train_sigmas,
        test_sigmas=config.noise.test_sigmas,
        generator_seed=config.seed
    )
    
    # Create and load model
    model = create_model(
        model_name=config.model.name,
        in_ch=config.model.in_ch,
        out_ch=config.model.get('out_ch', config.model.in_ch),
        **{k: v for k, v in config.model.items() if k not in ['name', 'in_ch', 'out_ch']}
    ).to(device)
    
    # Find and load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        ckpt_dir = Path(config.log.out_dir) / "ckpts"
        checkpoint_path = ckpt_dir / "best_model.pth"
    
    if Path(checkpoint_path).exists():
        from .utils import load_checkpoint
        load_checkpoint(checkpoint_path, model, device=device)
        console.print(f"Loaded model from {checkpoint_path}")
    else:
        console.print("[yellow]No checkpoint found, using untrained model[/yellow]")
    
    # Generate grids
    model.eval()
    with torch.no_grad():
        for batch_idx, (clean, noisy, sigmas) in enumerate(test_loader):
            if batch_idx >= args.num_batches:
                break
            
            clean = clean.to(device)
            noisy = noisy.to(device)
            recon = model(noisy)
            
            # Save grid
            output_path = Path(config.log.out_dir) / "grids" / f"recon_grid_batch_{batch_idx:03d}.png"
            create_reconstruction_grid(
                clean.cpu(), noisy.cpu(), recon.cpu(),
                output_path, num_samples=args.num_images
            )
    
    console.print(f"[green]Generated {args.num_batches} reconstruction grids[/green]")


def viz_sigma_panel_command(args):
    """Generate sigma sweep panels."""
    from .dataset import get_dataset_loaders
    from .models import create_model
    from .visualize import create_sigma_panel
    
    console.print("[blue]Generating sigma sweep panels[/blue]")
    
    # Load config and setup
    config = load_config(args.config)
    device = get_device(config.device)
    set_seed(config.seed)
    
    # Load data
    _, test_loader = get_dataset_loaders(
        dataset_name=config.data.dataset,
        root=config.data.root,
        batch_size=config.data.batch_size,
        num_workers=config.data.get('num_workers', 4),
        normalize=config.data.get('normalize', 'zero_one'),
        train_sigmas=config.noise.train_sigmas,
        test_sigmas=config.noise.test_sigmas,
        generator_seed=config.seed
    )
    
    # Create and load model
    model = create_model(
        model_name=config.model.name,
        in_ch=config.model.in_ch,
        out_ch=config.model.get('out_ch', config.model.in_ch),
        **{k: v for k, v in config.model.items() if k not in ['name', 'in_ch', 'out_ch']}
    ).to(device)
    
    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        ckpt_dir = Path(config.log.out_dir) / "ckpts"
        checkpoint_path = ckpt_dir / "best_model.pth"
    
    if Path(checkpoint_path).exists():
        from .utils import load_checkpoint
        load_checkpoint(checkpoint_path, model, device=device)
    
    # Generate panels
    clean, noisy, _ = next(iter(test_loader))
    
    for i in range(min(args.num_images, clean.size(0))):
        sample_image = clean[i]
        output_path = Path(config.log.out_dir) / "panels" / f"sigma_panel_{i:03d}.png"
        
        create_sigma_panel(
            model, sample_image, config.noise.test_sigmas,
            output_path, device
        )
    
    console.print(f"[green]Generated {min(args.num_images, clean.size(0))} sigma panels[/green]")


def compare_limitations_command(args):
    """Analyze model limitations vs generative models."""
    from .dataset import get_dataset_loaders
    from .models import create_model
    from .compare import analyze_limitations
    
    console.print("[bold magenta]Analyzing DAE limitations vs generative models[/bold magenta]")
    
    # Load config and setup
    config = load_config(args.config)
    device = get_device(config.device)
    set_seed(config.seed)
    
    # Load data
    _, test_loader = get_dataset_loaders(
        dataset_name=config.data.dataset,
        root=config.data.root,
        batch_size=config.data.batch_size,
        num_workers=config.data.get('num_workers', 4),
        normalize=config.data.get('normalize', 'zero_one'),
        train_sigmas=config.noise.train_sigmas,
        test_sigmas=config.noise.test_sigmas,
        generator_seed=config.seed
    )
    
    # Create and load model
    model = create_model(
        model_name=config.model.name,
        in_ch=config.model.in_ch,
        out_ch=config.model.get('out_ch', config.model.in_ch),
        **{k: v for k, v in config.model.items() if k not in ['name', 'in_ch', 'out_ch']}
    ).to(device)
    
    # Load checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        ckpt_dir = Path(config.log.out_dir) / "ckpts"
        checkpoint_path = ckpt_dir / "best_model.pth"
    
    if Path(checkpoint_path).exists():
        from .utils import load_checkpoint
        load_checkpoint(checkpoint_path, model, device=device)
        console.print(f"Loaded model from {checkpoint_path}")
    else:
        console.print("[yellow]No checkpoint found, analyzing untrained model[/yellow]")
    
    # Run analysis
    results = analyze_limitations(model, test_loader, config, device)
    
    if results:
        console.print("[green]Limitations analysis complete! Check outputs/reports/ for detailed results.[/green]")
    else:
        console.print("[red]Analysis failed[/red]")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Day 2: Denoising Autoencoder")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--config', type=str, required=True, help='Path to config file')
    common_parser.add_argument('--device', type=str, help='Device to use (cuda/cpu)')
    
    # Train command
    train_parser = subparsers.add_parser('train', parents=[common_parser], help='Train DAE model')
    train_parser.add_argument('--epochs', type=int, help='Number of training epochs')
    train_parser.add_argument('--lr', type=float, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    
    # Eval command
    eval_parser = subparsers.add_parser('eval', parents=[common_parser], help='Evaluate trained model')
    eval_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    eval_parser.add_argument('--batch-size', type=int, help='Batch size for evaluation')
    
    # Visualization commands
    viz_parser = subparsers.add_parser('viz.recon_grid', parents=[common_parser], help='Generate reconstruction grids')
    viz_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    viz_parser.add_argument('--num-batches', type=int, default=3, help='Number of batches to process')
    viz_parser.add_argument('--num-images', type=int, default=8, help='Number of images per grid')
    
    sigma_parser = subparsers.add_parser('viz.sigma_panel', parents=[common_parser], help='Generate sigma sweep panels')
    sigma_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    sigma_parser.add_argument('--num-images', type=int, default=4, help='Number of images to process')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare.limitations', parents=[common_parser], help='Analyze limitations vs generative models')
    compare_parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file')
    
    # Parse arguments
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Import torch here to avoid import issues
    import torch
    global torch
    
    # Execute command
    try:
        if args.command == 'train':
            train_command(args)
        elif args.command == 'eval':
            eval_command(args)
        elif args.command == 'viz.recon_grid':
            viz_recon_grid_command(args)
        elif args.command == 'viz.sigma_panel':
            viz_sigma_panel_command(args)
        elif args.command == 'compare.limitations':
            compare_limitations_command(args)
        else:
            console.print(f"[red]Unknown command: {args.command}[/red]")
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()