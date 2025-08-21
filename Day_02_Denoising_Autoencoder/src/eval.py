"""
Evaluation utilities for Day 2: Denoising Autoencoder
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from omegaconf import DictConfig
from rich.progress import track
from torch.utils.data import DataLoader

from .dataset import FixedNoiseDataset, get_dataset_loaders
from .metrics import MetricsCalculator, compute_sigma_wise_metrics
from .utils import console, load_checkpoint, save_image_grid, save_metrics


class DAEEvaluator:
    """Comprehensive evaluation for Denoising Autoencoder."""
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        device: torch.device,
        checkpoint_path: Optional[str] = None
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Load checkpoint if provided
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        
        self.model.eval()
        
        # Metrics calculator
        self.metrics_calc = MetricsCalculator(device)
        
        # Output directory
        self.output_dir = Path(config.log.out_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        console.print("[green]Evaluator initialized[/green]")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = load_checkpoint(
            checkpoint_path, 
            self.model, 
            device=self.device
        )
        
        # Load EMA weights if available
        if 'ema_state_dict' in checkpoint:
            console.print("[yellow]Loading EMA weights for evaluation[/yellow]")
            self.model.load_state_dict(checkpoint['ema_state_dict'])
        
        console.print(f"[green]Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}[/green]")
    
    def evaluate_dataset(
        self,
        test_loader: DataLoader,
        save_reconstructions: bool = True,
        max_batches: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on entire dataset.
        
        Args:
            test_loader: Test data loader
            save_reconstructions: Whether to save reconstruction grids
            max_batches: Limit evaluation to N batches for speed
        
        Returns:
            Dictionary with averaged metrics
        """
        self.model.eval()
        
        all_metrics = []
        reconstruction_grids = []
        
        with torch.no_grad():
            for batch_idx, (clean, noisy, sigmas) in enumerate(track(test_loader, description="Evaluating")):
                if max_batches and batch_idx >= max_batches:
                    break
                
                clean = clean.to(self.device)
                noisy = noisy.to(self.device)
                
                # Forward pass
                recon = self.model(noisy)
                
                # Compute metrics
                batch_metrics = self.metrics_calc.compute_all_metrics(
                    recon, clean, 
                    max_val=1.0,
                    compute_lpips=self.config.eval.get('compute_lpips', False)
                )
                all_metrics.append(batch_metrics)
                
                # Save some reconstructions
                if save_reconstructions and batch_idx < 5:  # Save first few batches
                    num_samples = min(8, clean.size(0))
                    grid_data = []
                    
                    for i in range(num_samples):
                        grid_data.extend([clean[i], noisy[i], recon[i]])
                    
                    reconstruction_grids.append(torch.stack(grid_data))
        
        # Average metrics
        if all_metrics:
            avg_metrics = {}
            for key in all_metrics[0].keys():
                avg_metrics[key] = sum(batch[key] for batch in all_metrics) / len(all_metrics)
        else:
            avg_metrics = {}
        
        # Save reconstruction grids
        if save_reconstructions and reconstruction_grids:
            full_grid = torch.cat(reconstruction_grids, dim=0)
            save_path = self.output_dir / "grids" / "evaluation_reconstructions.png"
            save_image_grid(full_grid, save_path, nrow=3)
        
        return avg_metrics
    
    def evaluate_noise_robustness(
        self,
        base_dataset,
        test_sigmas: List[float],
        num_samples: int = 500
    ) -> Dict[float, Dict[str, float]]:
        """
        Evaluate robustness to different noise levels.
        
        Args:
            base_dataset: Clean dataset
            test_sigmas: List of noise levels to test
            num_samples: Number of samples per noise level
        
        Returns:
            Dictionary mapping sigma -> metrics
        """
        console.print(f"[blue]Evaluating noise robustness on {len(test_sigmas)} noise levels[/blue]")
        
        sigma_metrics = {}
        
        for sigma in track(test_sigmas, description="Testing noise levels"):
            # Create fixed noise dataset for this sigma
            noisy_dataset = FixedNoiseDataset(
                base_dataset,
                noise_sigma=sigma,
                clip_range=(0, 1),
                seed=self.config.seed
            )
            
            # Limit samples
            if len(noisy_dataset) > num_samples:
                indices = torch.randperm(len(noisy_dataset))[:num_samples]
                noisy_dataset = torch.utils.data.Subset(noisy_dataset, indices)
            
            # Create dataloader
            test_loader = DataLoader(
                noisy_dataset,
                batch_size=32,
                shuffle=False,
                num_workers=2
            )
            
            # Evaluate on this noise level
            metrics = self.evaluate_dataset(test_loader, save_reconstructions=False)
            sigma_metrics[sigma] = metrics
            
            console.print(f"  σ = {sigma:.3f}: PSNR = {metrics.get('psnr', 0):.2f}, SSIM = {metrics.get('ssim', 0):.4f}")
        
        return sigma_metrics
    
    def create_sigma_sweep_panels(
        self,
        base_dataset,
        test_sigmas: List[float],
        num_images: int = 4
    ):
        """
        Create panels showing same images across different noise levels.
        
        Args:
            base_dataset: Clean dataset
            test_sigmas: List of noise levels
            num_images: Number of different images to show
        """
        # Select random images
        indices = torch.randperm(len(base_dataset))[:num_images]
        
        panels_data = []
        
        for img_idx, idx in enumerate(indices):
            clean_img, _ = base_dataset[idx]
            panel_row = [clean_img]  # Start with clean image
            
            # Add noisy and reconstructed versions for each sigma
            with torch.no_grad():
                for sigma in test_sigmas:
                    # Add noise
                    noisy_img = clean_img + torch.randn_like(clean_img) * sigma
                    noisy_img = torch.clamp(noisy_img, 0, 1)
                    
                    # Reconstruct
                    noisy_batch = noisy_img.unsqueeze(0).to(self.device)
                    recon_batch = self.model(noisy_batch)
                    recon_img = recon_batch.squeeze(0).cpu()
                    
                    panel_row.extend([noisy_img, recon_img])
            
            panels_data.append(torch.stack(panel_row))
        
        # Save panel
        if panels_data:
            full_panel = torch.cat(panels_data, dim=0)
            save_path = self.output_dir / "panels" / "sigma_sweep_panel.png"
            save_image_grid(full_panel, save_path, nrow=len(test_sigmas) * 2 + 1)
            
            console.print(f"[blue]Saved sigma sweep panel to {save_path}[/blue]")
    
    def evaluate_generalization(
        self,
        test_loader: DataLoader,
        out_of_distribution_sigmas: List[float]
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate generalization to noise levels not seen during training.
        
        Args:
            test_loader: Test data loader
            out_of_distribution_sigmas: Noise levels outside training range
        
        Returns:
            Dictionary with in-distribution vs out-of-distribution metrics
        """
        train_sigmas = set(self.config.noise.train_sigmas)
        test_sigmas = set(self.config.noise.test_sigmas)
        
        in_dist_metrics = []
        out_dist_metrics = []
        
        self.model.eval()
        with torch.no_grad():
            for clean, noisy, sigmas in track(test_loader, description="Evaluating generalization"):
                clean = clean.to(self.device)
                noisy = noisy.to(self.device)
                
                recon = self.model(noisy)
                
                # Separate metrics by sigma type
                for i, sigma in enumerate(sigmas):
                    sigma_val = sigma.item()
                    batch_metrics = self.metrics_calc.compute_all_metrics(
                        recon[i:i+1], clean[i:i+1]
                    )
                    
                    if sigma_val in train_sigmas:
                        in_dist_metrics.append(batch_metrics)
                    elif sigma_val in out_of_distribution_sigmas:
                        out_dist_metrics.append(batch_metrics)
        
        # Average metrics
        results = {}
        
        if in_dist_metrics:
            results['in_distribution'] = {}
            for key in in_dist_metrics[0].keys():
                results['in_distribution'][key] = sum(m[key] for m in in_dist_metrics) / len(in_dist_metrics)
        
        if out_dist_metrics:
            results['out_of_distribution'] = {}
            for key in out_dist_metrics[0].keys():
                results['out_of_distribution'][key] = sum(m[key] for m in out_dist_metrics) / len(out_dist_metrics)
        
        return results
    
    def generate_evaluation_report(
        self,
        test_loader: DataLoader,
        base_dataset,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        Generate comprehensive evaluation report.
        
        Returns:
            Complete evaluation results dictionary
        """
        console.print("[bold blue]Generating comprehensive evaluation report[/bold blue]")
        
        results = {}
        
        # 1. Overall dataset metrics
        console.print("1. Computing overall metrics...")
        results['overall'] = self.evaluate_dataset(test_loader)
        
        # 2. Noise robustness
        console.print("2. Testing noise robustness...")
        results['noise_robustness'] = self.evaluate_noise_robustness(
            base_dataset, 
            self.config.noise.test_sigmas,
            num_samples=300
        )
        
        # 3. Generalization to unseen noise levels
        if hasattr(self.config.noise, 'out_of_dist_sigmas'):
            console.print("3. Testing generalization...")
            results['generalization'] = self.evaluate_generalization(
                test_loader,
                self.config.noise.out_of_dist_sigmas
            )
        
        # 4. Create visualizations
        console.print("4. Creating visualizations...")
        self.create_sigma_sweep_panels(base_dataset, self.config.noise.test_sigmas[:5])
        
        # Save results
        if save_path is None:
            save_path = self.output_dir / "logs" / "evaluation_results.json"
        
        save_metrics(results, save_path)
        
        # Print summary
        overall_psnr = results['overall'].get('psnr', 0)
        overall_ssim = results['overall'].get('ssim', 0)
        
        console.print(f"[green]Evaluation complete![/green]")
        console.print(f"[green]Overall PSNR: {overall_psnr:.2f} dB[/green]")
        console.print(f"[green]Overall SSIM: {overall_ssim:.4f}[/green]")
        
        return results


def evaluate_model(config: DictConfig, checkpoint_path: str) -> Dict:
    """
    Main evaluation function.
    
    Args:
        config: Evaluation configuration
        checkpoint_path: Path to model checkpoint
    
    Returns:
        Evaluation results dictionary
    """
    from .models import create_model
    from .utils import get_device, set_seed
    
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    
    # Create model
    model = create_model(
        model_name=config.model.name,
        in_ch=config.model.in_ch,
        out_ch=config.model.get('out_ch', config.model.in_ch),
        **{k: v for k, v in config.model.items() if k not in ['name', 'in_ch', 'out_ch']}
    ).to(device)
    
    # Create evaluator
    evaluator = DAEEvaluator(model, config, device, checkpoint_path)
    
    # Get data
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
    
    # Get base dataset for robustness testing (MNIST only)
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms
    
    if config.data.dataset.lower() != "mnist":
        raise ValueError(f"Only MNIST dataset is supported, got: {config.data.dataset}")
    
    base_dataset = MNIST(root=config.data.root, train=False, transform=transforms.ToTensor())
    
    # Run evaluation
    results = evaluator.generate_evaluation_report(test_loader, base_dataset)
    
    return results