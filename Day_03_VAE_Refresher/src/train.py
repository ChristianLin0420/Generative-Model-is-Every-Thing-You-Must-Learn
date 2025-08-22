"""
Training script for VAE models.
Supports full training loop with AMP, KL warmup/cyclical annealing, and EMA.
"""

import os
import time
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig

from .models.vae_conv import VAEConv
from .models.vae_resnet import VAEResNet
from .models.vae_mlp import VAEMLP
from .dataset import create_unlabeled_dataloaders
from .losses import elbo_loss, BetaScheduler
from .utils import (
    set_seed, get_device, save_checkpoint, setup_logger,
    count_parameters, get_model_size_mb, EMA, MetricsTracker,
    create_progress_bar
)


def create_model(config: DictConfig) -> nn.Module:
    """Create VAE model based on configuration."""
    model_name = config.model.name
    in_channels = config.model.in_ch
    latent_dim = config.model.latent_dim
    base_channels = config.model.base_ch
    
    if model_name == "vae_conv":
        model = VAEConv(in_channels, latent_dim, base_channels)
    elif model_name == "vae_resnet":
        model = VAEResNet(in_channels, latent_dim, base_channels)
    elif model_name == "vae_mlp":
        # For MLP, we need to know the image size
        img_size = 28 if config.data.dataset == "mnist" else 32
        model = VAEMLP(in_channels, img_size, latent_dim)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def create_optimizer(model: nn.Module, config: DictConfig) -> optim.Optimizer:
    """Create optimizer based on configuration."""
    optimizer_name = config.train.opt.lower()
    learning_rate = config.train.lr
    
    if optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, config: DictConfig) -> Optional[optim.lr_scheduler.LRScheduler]:
    """Create learning rate scheduler."""
    # Cosine annealing with warm restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=config.train.lr * 0.01
    )
    return scheduler


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    beta_scheduler: BetaScheduler,
    epoch: int,
    device: torch.device,
    config: DictConfig,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    ema: Optional[EMA] = None,
    logger: Optional[Any] = None
) -> Dict[str, float]:
    """
    Train model for one epoch.
    
    Args:
        model: VAE model
        dataloader: Training data loader
        optimizer: Optimizer
        beta_scheduler: Beta annealing scheduler
        epoch: Current epoch number
        device: Training device
        config: Configuration
        scaler: AMP gradient scaler
        ema: Exponential moving average
        logger: Logger instance
    
    Returns:
        Dictionary of training metrics
    """
    model.train()
    metrics_tracker = MetricsTracker()
    
    # Get current beta value
    beta = beta_scheduler.get_beta(epoch)
    
    with create_progress_bar(f"Training Epoch {epoch+1}") as progress:
        task = progress.add_task("Training", total=len(dataloader))
        
        for batch_idx, batch_data in enumerate(dataloader):
            # Handle both labeled and unlabeled data
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                x, _ = batch_data
            else:
                x = batch_data
            
            x = x.to(device)
            optimizer.zero_grad()
            
            # Forward pass with AMP if enabled
            if config.train.amp and scaler is not None:
                with torch.cuda.amp.autocast():
                    x_hat, mu, logvar = model(x)
                    loss, loss_components = elbo_loss(
                        x_hat, x, mu, logvar, beta, config.train.recon_loss
                    )
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                x_hat, mu, logvar = model(x)
                loss, loss_components = elbo_loss(
                    x_hat, x, mu, logvar, beta, config.train.recon_loss
                )
                
                # Backward pass
                loss.backward()
                optimizer.step()
            
            # Update EMA if enabled
            if ema is not None:
                ema.update(model)
            
            # Update metrics
            batch_metrics = {
                "loss": loss.item(),
                "recon_loss": loss_components["recon_loss"].item(),
                "kl_loss": loss_components["kl_loss"].item(),
                "beta": beta,
            }
            metrics_tracker.update(batch_metrics, x.size(0))
            
            # Update progress bar
            progress.update(task, advance=1, 
                          description=f"Training Epoch {epoch+1} - Loss: {loss.item():.4f}")
            
            # Log batch metrics periodically
            if logger and batch_idx % 100 == 0:
                step = epoch * len(dataloader) + batch_idx
                logger.info(
                    f"Epoch {epoch+1}, Batch {batch_idx}: "
                    f"Loss={loss.item():.4f}, "
                    f"Recon={loss_components['recon_loss'].item():.4f}, "
                    f"KL={loss_components['kl_loss'].item():.4f}, "
                    f"Beta={beta:.4f}"
                )
    
    return metrics_tracker.get_averages()


def validate_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    beta: float,
    device: torch.device,
    config: DictConfig
) -> Dict[str, float]:
    """
    Validate model for one epoch.
    
    Args:
        model: VAE model
        dataloader: Validation data loader
        beta: Current beta value
        device: Device
        config: Configuration
    
    Returns:
        Dictionary of validation metrics
    """
    model.eval()
    metrics_tracker = MetricsTracker()
    
    with torch.no_grad():
        for batch_data in dataloader:
            # Handle both labeled and unlabeled data
            if isinstance(batch_data, (list, tuple)) and len(batch_data) == 2:
                x, _ = batch_data
            else:
                x = batch_data
            
            x = x.to(device)
            
            # Forward pass
            x_hat, mu, logvar = model(x)
            loss, loss_components = elbo_loss(
                x_hat, x, mu, logvar, beta, config.train.recon_loss
            )
            
            # Update metrics
            batch_metrics = {
                "loss": loss.item(),
                "recon_loss": loss_components["recon_loss"].item(),
                "kl_loss": loss_components["kl_loss"].item(),
                "beta": beta,
            }
            metrics_tracker.update(batch_metrics, x.size(0))
    
    return metrics_tracker.get_averages()


def train_vae(config: DictConfig) -> None:
    """
    Main training function for VAE.
    
    Args:
        config: Training configuration
    """
    # Setup
    set_seed(config.seed)
    device = get_device(config.device)
    
    # Create output directories
    log_dir = os.path.join(config.log.out_dir, "logs")
    ckpt_dir = os.path.join(config.log.out_dir, "ckpts")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logger("VAE_Training", os.path.join(log_dir, "train.log"))
    writer = SummaryWriter(log_dir)
    
    logger.info(f"Starting VAE training with config: {config}")
    logger.info(f"Using device: {device}")
    
    # Create model
    model = create_model(config).to(device)
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Parameters: {count_parameters(model):,}")
    logger.info(f"Model size: {get_model_size_mb(model):.2f} MB")
    
    # Create data loaders
    train_loader, test_loader = create_unlabeled_dataloaders(
        dataset=config.data.dataset,
        root=config.data.root,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        normalize=config.data.normalize
    )
    
    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Test batches: {len(test_loader)}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Create beta scheduler
    beta_scheduler = BetaScheduler(
        schedule_type=config.train.kl_schedule,
        max_beta=config.train.beta,
        warmup_epochs=config.train.kl_warmup_epochs
    )
    
    # Setup AMP
    scaler = None
    if config.train.amp:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Using Automatic Mixed Precision (AMP)")
    
    # Setup EMA
    ema = None
    if config.train.ema:
        ema = EMA(model, decay=0.999)
        logger.info("Using Exponential Moving Average (EMA)")
    
    # Training loop
    best_loss = float('inf')
    start_time = time.time()
    
    logger.info("Starting training loop...")
    
    for epoch in range(config.train.epochs):
        epoch_start_time = time.time()
        
        # Train epoch
        train_metrics = train_epoch(
            model, train_loader, optimizer, beta_scheduler, epoch, device,
            config, scaler, ema, logger
        )
        
        # Validate epoch
        beta = beta_scheduler.get_beta(epoch)
        val_metrics = validate_epoch(model, test_loader, beta, device, config)
        
        # Update scheduler
        if scheduler:
            scheduler.step()
        
        # Log metrics
        epoch_time = time.time() - epoch_start_time
        
        logger.info(
            f"Epoch {epoch+1}/{config.train.epochs} ({epoch_time:.2f}s) - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )
        
        # TensorBoard logging
        writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/Validation", val_metrics["loss"], epoch)
        writer.add_scalar("Loss/Train_Reconstruction", train_metrics["recon_loss"], epoch)
        writer.add_scalar("Loss/Train_KL", train_metrics["kl_loss"], epoch)
        writer.add_scalar("Loss/Val_Reconstruction", val_metrics["recon_loss"], epoch)
        writer.add_scalar("Loss/Val_KL", val_metrics["kl_loss"], epoch)
        writer.add_scalar("Training/Beta", beta, epoch)
        writer.add_scalar("Training/Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoints
        is_best = val_metrics['loss'] < best_loss
        if is_best:
            best_loss = val_metrics['loss']
        
        if (epoch + 1) % config.log.save_every == 0 or is_best:
            checkpoint_path = os.path.join(ckpt_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_metrics['loss'],
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                filepath=checkpoint_path,
                config=config,
                beta=beta
            )
            
            if is_best:
                best_path = os.path.join(ckpt_dir, "best.pt")
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    loss=val_metrics['loss'],
                    metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                    filepath=best_path,
                    config=config,
                    beta=beta
                )
                logger.info(f"New best model saved with loss: {best_loss:.4f}")
    
    # Final logging
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")
    logger.info(f"Best validation loss: {best_loss:.4f}")
    
    # Close writer
    writer.close()
    
    # Save final model with EMA weights if enabled
    if ema is not None:
        ema.apply_shadow(model)
        final_path = os.path.join(ckpt_dir, "final_ema.pt")
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=config.train.epochs - 1,
            loss=best_loss,
            metrics={},
            filepath=final_path,
            config=config,
            beta=beta
        )
        logger.info("Final EMA model saved")


if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    
    if len(sys.argv) != 2:
        print("Usage: python -m src.train <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    config = OmegaConf.load(config_path)
    
    train_vae(config)