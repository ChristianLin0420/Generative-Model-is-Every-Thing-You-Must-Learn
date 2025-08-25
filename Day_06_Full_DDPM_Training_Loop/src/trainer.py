"""
DDPM Training Loop with AMP, EMA, grad clipping, LR scheduling, and periodic sampling
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List, Callable
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import torch.nn.functional as F

from .utils import Logger, EMAModel, CheckpointManager, save_image_grid, get_device
from .losses import DDPMLoss
from .ddpm_schedules import DDPMSchedules
from .sampler import DDPMSampler
from .dataset import InfiniteDataLoader


class DDPMTrainer:
    """
    Comprehensive DDPM trainer with all the bells and whistles
    """
    
    def __init__(
        self,
        model: nn.Module,
        schedules: DDPMSchedules,
        loss_fn: DDPMLoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        output_dir: str = "outputs"
    ):
        self.model = model
        self.schedules = schedules
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device if device is not None else get_device()
        self.output_dir = Path(output_dir)
        
        # Move model and schedules to device
        self.model = self.model.to(self.device)
        self.schedules = self.schedules.to(self.device)
        
        # Training config
        train_config = config["training"]
        self.num_epochs = train_config["num_epochs"]
        self.batch_size = train_config["batch_size"]
        self.learning_rate = float(train_config["learning_rate"])
        self.weight_decay = float(train_config.get("weight_decay", 0.0))
        self.grad_clip_norm = float(train_config.get("grad_clip_norm", 1.0))
        
        # AMP config
        self.use_amp = train_config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # EMA config
        ema_config = train_config.get("ema", {})
        self.use_ema = ema_config.get("enabled", True)
        if self.use_ema:
            self.ema = EMAModel(
                model=self.model,
                decay=ema_config.get("decay", 0.999),
                device=self.device
            )
        else:
            self.ema = None
            
        # Setup optimizer
        optimizer_config = train_config.get("optimizer", {"type": "adamw"})
        self.optimizer = self._create_optimizer(optimizer_config)
        
        # Setup scheduler
        scheduler_config = train_config.get("scheduler", {"type": "cosine"})
        self.scheduler = self._create_scheduler(scheduler_config)
        
        # Logging and checkpointing
        self.logger = Logger(self.output_dir / "logs", "ddpm_trainer")
        self.checkpoint_manager = CheckpointManager(self.output_dir / "ckpts")
        
        # Sampling config
        sample_config = train_config.get("sampling", {})
        self.sample_every = sample_config.get("every_epochs", 10)
        self.sample_num_images = sample_config.get("num_images", 64)
        self.sample_ddim_steps = sample_config.get("ddim_steps", 50)
        
        # Create sampler for periodic sampling
        self.sampler = DDPMSampler(self.schedules)
        
        # Training state
        self.epoch = 0
        self.step = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        
        # Create infinite dataloader for step-based training
        self.infinite_train_loader = InfiniteDataLoader(self.train_loader)
        
    def _create_optimizer(self, optimizer_config: Dict[str, Any]) -> optim.Optimizer:
        """Create optimizer from config"""
        optimizer_type = optimizer_config.get("type", "adamw").lower()
        
        if optimizer_type == "adamw":
            betas = optimizer_config.get("betas", (0.9, 0.999))
            if isinstance(betas, list):
                betas = tuple(float(b) for b in betas)
            eps = float(optimizer_config.get("eps", 1e-8))
            
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=betas,
                eps=eps
            )
        elif optimizer_type == "adam":
            betas = optimizer_config.get("betas", (0.9, 0.999))
            if isinstance(betas, list):
                betas = tuple(float(b) for b in betas)
            eps = float(optimizer_config.get("eps", 1e-8))
            
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=betas,
                eps=eps
            )
        elif optimizer_type == "sgd":
            momentum = float(optimizer_config.get("momentum", 0.9))
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=momentum,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
            
    def _create_scheduler(self, scheduler_config: Dict[str, Any]) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler from config"""
        scheduler_type = scheduler_config.get("type", "cosine").lower()
        
        if scheduler_type == "none":
            return None
        elif scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.num_epochs,
                eta_min=float(scheduler_config.get("eta_min", 1e-6))
            )
        elif scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=int(scheduler_config.get("step_size", 50)),
                gamma=float(scheduler_config.get("gamma", 0.1))
            )
        elif scheduler_type == "exponential":
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=float(scheduler_config.get("gamma", 0.95))
            )
        elif scheduler_type == "multistep":
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=scheduler_config.get("milestones", [100, 150]),
                gamma=float(scheduler_config.get("gamma", 0.1))
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
            
    def train_step(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Unpack batch (handle both conditional and unconditional)
        if len(batch) == 2:
            images, labels = batch
            images = images.to(self.device)
            # For now, ignore labels (unconditional training)
        else:
            images = batch[0].to(self.device)
            
        batch_size = images.shape[0]
        
        # Forward pass with AMP
        with autocast(enabled=self.use_amp):
            loss_dict = self.loss_fn(self.model, images, return_dict=True)
            loss = loss_dict["total_loss"]
            
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.grad_clip_norm
                )
                
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.grad_clip_norm
                )
                
            self.optimizer.step()
            
        # Update EMA
        if self.use_ema:
            self.ema.update(self.model)
            
        # Metrics
        metrics = {
            "train_loss": loss.item(),
            "train_simple_loss": loss_dict["simple_loss"].item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "batch_size": batch_size
        }
        
        return metrics
        
    def validate(self) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        
        val_losses = []
        val_simple_losses = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                if len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch[0]
                images = images.to(self.device)
                
                with autocast(enabled=self.use_amp):
                    loss_dict = self.loss_fn(self.model, images, return_dict=True)
                    
                val_losses.append(loss_dict["total_loss"].item())
                val_simple_losses.append(loss_dict["simple_loss"].item())
                
                # Limit validation batches for speed
                if batch_idx >= 10:
                    break
                    
        metrics = {
            "val_loss": sum(val_losses) / len(val_losses),
            "val_simple_loss": sum(val_simple_losses) / len(val_simple_losses)
        }
        
        return metrics
        
    def sample_images(self, use_ema: bool = True) -> torch.Tensor:
        """Generate sample images for visualization"""
        model_to_use = self.model
        
        # Use EMA model if available
        if use_ema and self.ema is not None:
            self.ema.apply_shadow(self.model)
            
        try:
            model_to_use.eval()
            
            with torch.no_grad():
                # Sample shape based on dataset
                if hasattr(self.train_loader.dataset, 'dataset'):
                    # Handle wrapped datasets
                    sample_data = next(iter(self.train_loader))[0][:1]
                else:
                    sample_data = next(iter(self.train_loader))[0][:1]
                    
                sample_shape = (self.sample_num_images, *sample_data.shape[1:])
                
                # Generate samples using DDIM for speed
                samples = self.sampler.ddim_sample(
                    model=model_to_use,
                    shape=sample_shape,
                    num_steps=self.sample_ddim_steps,
                    eta=0.0,
                    device=self.device
                )
                
            return samples
            
        finally:
            # Restore original model weights
            if use_ema and self.ema is not None:
                self.ema.restore(self.model)
                
    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """Save training checkpoint"""
        metadata = {
            "config": self.config,
            "epoch": self.epoch,
            "step": self.step,
            "best_loss": self.best_loss
        }
        
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.epoch,
            step=self.step,
            loss=getattr(self, 'current_loss', 0.0),
            ema=self.ema,
            metadata=metadata,
            filename=filename
        )
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_manager.checkpoint_dir / "best.pt"
            torch.save(torch.load(checkpoint_path), best_path)
            
        # Save EMA model separately
        if self.ema is not None:
            ema_path = self.checkpoint_manager.checkpoint_dir / "ema.pt"
            torch.save({
                "model_state_dict": self.ema.state_dict(),
                "epoch": self.epoch,
                "step": self.step,
                "metadata": metadata
            }, ema_path)
            
        return checkpoint_path
        
    def load_checkpoint(self, filename: str = "latest.pt"):
        """Load training checkpoint"""
        checkpoint_info = self.checkpoint_manager.load_checkpoint(
            filename=filename,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            device=self.device
        )
        
        self.epoch = checkpoint_info["epoch"]
        self.step = checkpoint_info["step"]
        self.best_loss = checkpoint_info.get("metadata", {}).get("best_loss", float('inf'))
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch}, step {self.step}")
        return checkpoint_info
        
    def train_epoch(self) -> Dict[str, float]:
        """Train single epoch"""
        self.model.train()
        
        epoch_metrics = []
        epoch_start_time = time.time()
        
        # Use infinite dataloader for consistent step counting
        for batch_idx, batch in enumerate(self.train_loader):
            step_start_time = time.time()
            
            # Training step
            step_metrics = self.train_step(batch)
            self.step += 1
            
            # Add timing
            step_metrics["step_time"] = time.time() - step_start_time
            epoch_metrics.append(step_metrics)
            
            # Log every N steps
            if batch_idx % 100 == 0:
                self.logger.log_metrics(step_metrics, self.step, prefix="train")
                
        # Aggregate epoch metrics
        epoch_metrics_agg = {}
        for key in epoch_metrics[0].keys():
            if key != "batch_size":
                epoch_metrics_agg[key] = sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
                
        epoch_metrics_agg["epoch_time"] = time.time() - epoch_start_time
        
        return epoch_metrics_agg
        
    def train(self, resume_from: Optional[str] = None):
        """Full training loop"""
        self.logger.info(f"Starting DDPM training for {self.num_epochs} epochs")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Batch size: {self.batch_size}")
        self.logger.info(f"Learning rate: {self.learning_rate}")
        
        # Resume from checkpoint if requested
        if resume_from is not None:
            self.load_checkpoint(resume_from)
            
        # Training loop
        for epoch in range(self.epoch, self.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.train_metrics.append(train_metrics)
            
            # Validation
            val_metrics = self.validate()
            self.val_metrics.append(val_metrics)
            
            # Update scheduler
            if self.scheduler is not None:
                self.scheduler.step()
                
            # Combined metrics for logging
            combined_metrics = {**train_metrics, **val_metrics}
            self.current_loss = combined_metrics["val_loss"]
            
            # Log epoch metrics
            self.logger.log_metrics(combined_metrics, epoch, prefix="epoch")
            
            # Check if best model
            is_best = self.current_loss < self.best_loss
            if is_best:
                self.best_loss = self.current_loss
                
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Sample images periodically
            if (epoch + 1) % self.sample_every == 0 or epoch == 0:
                try:
                    samples = self.sample_images(use_ema=True)
                    
                    # Save sample grid
                    sample_path = self.output_dir / "grids" / f"samples_epoch_{epoch:04d}.png"
                    save_image_grid(samples, str(sample_path), nrow=8, normalize=True, value_range=(-1, 1))
                    
                    self.logger.info(f"Saved {len(samples)} samples to {sample_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to generate samples: {e}")
                    
        self.logger.info("Training completed!")
        
        # Save final metrics
        self.save_training_curves()
        
    def save_training_curves(self):
        """Save training curves to file"""
        import json
        
        # Save metrics as JSON
        metrics_data = {
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "config": self.config
        }
        
        metrics_path = self.output_dir / "logs" / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
            
        self.logger.info(f"Saved training metrics to {metrics_path}")
        
    def compute_model_flops(self) -> int:
        """Estimate model FLOPs (rough approximation)"""
        # This is a very rough estimate
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Assume roughly 2 FLOPs per parameter per forward pass
        # This is a crude approximation
        sample_input = next(iter(self.train_loader))[0][:1].to(self.device)
        H, W = sample_input.shape[-2:]
        
        # Rough estimate: params * image_pixels * 2
        estimated_flops = total_params * H * W * 2
        
        return estimated_flops