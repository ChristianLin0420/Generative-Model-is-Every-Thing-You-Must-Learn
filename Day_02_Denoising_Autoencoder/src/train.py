"""
Training loop for Day 2: Denoising Autoencoder
"""

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from rich.progress import track
try:
    from torch.amp import GradScaler, autocast
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .loss import get_loss_function
from .metrics import EpochMetrics, MetricsCalculator, MetricsLogger
from .utils import AverageMeter, EMAModel, console, save_checkpoint, save_image_grid


class DAETrainer:
    """Denoising Autoencoder Trainer with full features."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DictConfig,
        device: torch.device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Setup loss function
        self.criterion = get_loss_function(
            loss_type=config.train.loss,
            perceptual_weight=config.train.get('perceptual_weight', 0.0)
        ).to(device)
        
        # Mixed precision training
        self.use_amp = config.train.get('amp', False) and torch.cuda.is_available()
        if self.use_amp:
            try:
                self.scaler = GradScaler('cuda')
            except:
                self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # EMA model
        self.use_ema = config.train.get('ema', False)
        if self.use_ema:
            self.ema_model = EMAModel(
                model, 
                decay=config.train.get('ema_decay', 0.999),
                device=device
            )
        else:
            self.ema_model = None
        
        # Metrics and logging
        self.metrics_calc = MetricsCalculator(device)
        self.metrics_logger = MetricsLogger(Path(config.log.out_dir) / "logs")
        
        # Training state
        self.current_epoch = 0
        self.best_val_psnr = 0.0
        self.global_step = 0
        
        # Create output directories
        self.output_dir = Path(config.log.out_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "ckpts").mkdir(exist_ok=True)
        (self.output_dir / "grids").mkdir(exist_ok=True)
        
        console.print(f"[green]Trainer initialized with {sum(p.numel() for p in model.parameters()):,} parameters[/green]")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        opt_name = self.config.train.opt.lower()
        
        if opt_name == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.train.lr,
                weight_decay=self.config.train.get('weight_decay', 1e-4),
                betas=(0.9, 0.999)
            )
        elif opt_name == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.train.lr,
                weight_decay=self.config.train.get('weight_decay', 1e-4)
            )
        elif opt_name == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.train.lr,
                weight_decay=self.config.train.get('weight_decay', 1e-4),
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_name}")
    
    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        scheduler_type = self.config.train.get('scheduler', 'none')
        
        if scheduler_type == 'none':
            return None
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.train.epochs,
                eta_min=self.config.train.lr * 0.01
            )
        elif scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.train.epochs // 3,
                gamma=0.1
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        # Metrics tracking
        metrics = EpochMetrics(['loss', 'mse', 'psnr', 'ssim'])
        loss_meter = AverageMeter('Loss', ':.6f')
        
        # Progress bar
        progress_bar = track(
            self.train_loader, 
            description=f"Epoch {epoch:3d}/{self.config.train.epochs}"
        )
        
        for batch_idx, (clean, noisy, sigmas) in enumerate(progress_bar):
            clean = clean.to(self.device)
            noisy = noisy.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast(self.device.type):
                    recon = self.model(noisy)
                    loss_dict = self.criterion(recon, clean)
                    loss = loss_dict['total'] if isinstance(loss_dict, dict) else loss_dict
            else:
                recon = self.model(noisy)
                loss_dict = self.criterion(recon, clean)
                loss = loss_dict['total'] if isinstance(loss_dict, dict) else loss_dict
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.train.get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.train.grad_clip
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.config.train.get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.train.grad_clip
                    )
                
                self.optimizer.step()
            
            # Update EMA
            if self.ema_model:
                self.ema_model.update(self.model)
            
            # Update metrics
            with torch.no_grad():
                batch_metrics = self.metrics_calc.compute_all_metrics(recon, clean)
                metrics.update(
                    loss=loss.item(),
                    **batch_metrics
                )
            
            loss_meter.update(loss.item(), clean.size(0))
            self.global_step += 1
            
            # Log periodically
            if batch_idx % self.config.log.get('log_every', 100) == 0:
                console.print(
                    f"Train Epoch {epoch} [{batch_idx:6d}/{len(self.train_loader):6d}] "
                    f"Loss: {loss_meter.avg:.6f} LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
        
        return metrics.get_averages()
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        # Use EMA model for validation if available
        eval_model = self.ema_model.ema_model if self.ema_model else self.model
        eval_model.eval()
        
        metrics = EpochMetrics(['loss', 'mse', 'psnr', 'ssim'])
        
        with torch.no_grad():
            for clean, noisy, sigmas in track(self.val_loader, description="Validating"):
                clean = clean.to(self.device)
                noisy = noisy.to(self.device)
                
                recon = eval_model(noisy)
                loss_dict = self.criterion(recon, clean)
                loss = loss_dict['total'] if isinstance(loss_dict, dict) else loss_dict
                
                # Compute metrics
                batch_metrics = self.metrics_calc.compute_all_metrics(recon, clean)
                metrics.update(
                    loss=loss.item(),
                    **batch_metrics
                )
        
        return metrics.get_averages()
    
    def save_visualization(self, epoch: int, num_samples: int = 8):
        """Save reconstruction visualizations."""
        eval_model = self.ema_model.ema_model if self.ema_model else self.model
        eval_model.eval()
        
        with torch.no_grad():
            # Get a batch for visualization
            clean, noisy, sigmas = next(iter(self.val_loader))
            clean = clean[:num_samples].to(self.device)
            noisy = noisy[:num_samples].to(self.device)
            
            recon = eval_model(noisy)
            
            # Create grid: [clean, noisy, recon] for each image
            grid_images = []
            for i in range(num_samples):
                grid_images.extend([clean[i], noisy[i], recon[i]])
            
            grid_tensor = torch.stack(grid_images)
            save_path = self.output_dir / "grids" / f"recon_epoch_{epoch:04d}.png"
            save_image_grid(grid_tensor, save_path, nrow=3)
    
    def save_model(self, epoch: int, val_metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'config': self.config
        }
        
        if self.ema_model:
            checkpoint_data['ema_state_dict'] = self.ema_model.state_dict()
        
        if self.scheduler:
            checkpoint_data['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        save_path = self.output_dir / "ckpts" / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint_data, save_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "ckpts" / "best_model.pth"
            torch.save(checkpoint_data, best_path)
            console.print(f"[green]Saved best model with PSNR: {val_metrics['psnr']:.4f}[/green]")
    
    def train(self):
        """Full training loop."""
        console.print(f"[bold blue]Starting training for {self.config.train.epochs} epochs[/bold blue]")
        
        start_time = time.time()
        
        for epoch in range(1, self.config.train.epochs + 1):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            if epoch % self.config.log.get('eval_every', 1) == 0:
                val_metrics = self.validate(epoch)
                
                # Check if best model
                is_best = val_metrics['psnr'] > self.best_val_psnr
                if is_best:
                    self.best_val_psnr = val_metrics['psnr']
                
                # Log metrics
                self.metrics_logger.log_epoch(epoch, train_metrics, val_metrics)
                
                # Print progress
                console.print(
                    f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_metrics['loss']:.4f} PSNR: {train_metrics['psnr']:.2f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} PSNR: {val_metrics['psnr']:.2f} SSIM: {val_metrics['ssim']:.4f}"
                )
                
                # Save model
                if epoch % self.config.log.get('save_every', 5) == 0 or is_best:
                    self.save_model(epoch, val_metrics, is_best)
                
                # Save visualizations
                if epoch % self.config.log.get('viz_every', 5) == 0:
                    self.save_visualization(epoch)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
        
        # Final save
        self.metrics_logger.save_csv()
        
        total_time = time.time() - start_time
        console.print(f"[green]Training completed in {total_time:.2f}s[/green]")
        console.print(f"[green]Best validation PSNR: {self.best_val_psnr:.4f}[/green]")


def train_model(config: DictConfig) -> Dict[str, float]:
    """
    Main training function.
    
    Args:
        config: Training configuration
    
    Returns:
        Dictionary with final validation metrics
    """
    from .dataset import get_dataset_loaders
    from .models import create_model
    from .utils import get_device, set_seed
    
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    
    # Data
    train_loader, val_loader = get_dataset_loaders(
        dataset_name=config.data.dataset,
        root=config.data.root,
        batch_size=config.data.batch_size,
        num_workers=config.data.get('num_workers', 4),
        normalize=config.data.get('normalize', 'zero_one'),
        train_sigmas=config.noise.train_sigmas,
        test_sigmas=config.noise.test_sigmas,
        generator_seed=config.seed
    )
    
    # Model
    model = create_model(
        model_name=config.model.name,
        in_ch=config.model.in_ch,
        out_ch=config.model.get('out_ch', config.model.in_ch),
        base_ch=config.model.get('base_ch', 64),
        num_downs=config.model.get('num_downs', 3),
        norm_type=config.model.get('norm_type', 'group'),
        dropout=config.model.get('dropout', 0.1)
    ).to(device)
    
    # Trainer
    trainer = DAETrainer(model, train_loader, val_loader, config, device)
    
    # Train
    trainer.train()
    
    # Return final metrics
    best_metrics = trainer.metrics_logger.get_best_epoch('psnr', 'max')
    return best_metrics