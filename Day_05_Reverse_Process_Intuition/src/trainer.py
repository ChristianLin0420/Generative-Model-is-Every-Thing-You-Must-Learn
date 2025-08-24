"""DDPM Training Loop with modern features

Implements comprehensive training with:
- Mixed precision (AMP)
- Exponential Moving Average (EMA) 
- Gradient clipping
- Learning rate scheduling
- Periodic sampling and evaluation
- Tensorboard logging
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import os
import time
from typing import Dict, Any, Optional, List, Callable
from tqdm.auto import tqdm
import numpy as np

from .ddpm_schedules import DDPMScheduler
from .losses import DDPMLoss, compute_training_sample
from .utils import EMA, save_checkpoint, save_image_grid, count_parameters
from .sampler import DDPMSampler


class DDPMTrainer:
    """DDPM training loop with all modern features."""
    
    def __init__(
        self,
        model: nn.Module,
        scheduler: DDPMScheduler,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        device: torch.device = None,
        config: Dict[str, Any] = None
    ):
        """Initialize DDPM trainer.
        
        Args:
            model: Denoising model to train
            scheduler: DDPM noise scheduler
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            device: Training device
            config: Training configuration
        """
        self.model = model
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        
        # Move model and scheduler to device
        self.model.to(self.device)
        self.scheduler.to(self.device)
        
        # Training configuration
        training_config = self.config.get("training", {})
        self.epochs = int(training_config.get("epochs", 100))
        self.learning_rate = float(training_config.get("learning_rate", 2e-4))
        self.weight_decay = float(training_config.get("weight_decay", 0.0))
        self.gradient_clip_val = float(training_config.get("gradient_clip_val", 1.0))
        self.mixed_precision = training_config.get("mixed_precision", True)
        
        # EMA configuration
        self.use_ema = training_config.get("ema_decay", 0.9999) is not None
        if self.use_ema:
            self.ema_decay = float(training_config.get("ema_decay", 0.9999))
            self.ema_model = EMA(self.model, decay=self.ema_decay)
        else:
            self.ema_model = None
        
        # Logging configuration
        self.log_every = int(training_config.get("log_every", 100))
        self.save_every = int(training_config.get("save_every", 5))
        self.sample_every = int(training_config.get("sample_every", 5))
        self.num_sample_images = int(training_config.get("num_sample_images", 16))
        
        # Paths
        paths_config = self.config.get("paths", {})
        self.checkpoint_dir = paths_config.get("checkpoint_dir", "outputs/ckpts")
        self.log_dir = paths_config.get("log_dir", "outputs/logs")
        self.sample_dir = paths_config.get("sample_dir", "outputs/samples")
        
        # Create directories
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.sample_dir, exist_ok=True)
        
        # Setup optimizer and loss
        self._setup_optimizer_and_loss()
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.mixed_precision else None
        
        # Tensorboard logger
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Log model info
        param_info = count_parameters(self.model)
        print(f"Model parameters: {param_info['total']:,} total, {param_info['trainable']:,} trainable")
        self.writer.add_scalar("Model/TotalParams", param_info['total'], 0)
        self.writer.add_scalar("Model/TrainableParams", param_info['trainable'], 0)
    
    def _setup_optimizer_and_loss(self):
        """Setup optimizer, scheduler, and loss function."""
        training_config = self.config.get("training", {})
        ddpm_config = self.config.get("ddpm", {})
        
        # Optimizer
        optimizer_type = training_config.get("optimizer", "adamw").lower()
        if optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
        
        # Learning rate scheduler
        lr_scheduler_type = training_config.get("lr_scheduler", "cosine").lower()
        warmup_steps = training_config.get("warmup_steps", 0)
        
        if lr_scheduler_type == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            total_steps = len(self.train_dataloader) * self.epochs
            self.lr_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=total_steps, eta_min=self.learning_rate * 0.01
            )
        elif lr_scheduler_type == "step":
            from torch.optim.lr_scheduler import StepLR
            step_size = training_config.get("step_size", 30)
            gamma = training_config.get("gamma", 0.1)
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=step_size, gamma=gamma
            )
        elif lr_scheduler_type == "none":
            self.lr_scheduler = None
        else:
            raise ValueError(f"Unknown lr_scheduler: {lr_scheduler_type}")
        
        # Loss function
        prediction_type = ddpm_config.get("prediction_type", "epsilon")
        self.loss_fn = DDPMLoss(
            scheduler=self.scheduler,
            prediction_type=prediction_type,
            loss_type="l2",
            weight_schedule=None  # Start simple
        )
    
    def train_step(self, batch: torch.Tensor) -> Dict[str, float]:
        """Single training step.
        
        Args:
            batch: Batch of clean images
            
        Returns:
            Dictionary with loss values
        """
        # Handle different batch formats
        if isinstance(batch, (list, tuple)):
            x_start = batch[0]  # Ignore labels for now
        else:
            x_start = batch
        
        x_start = x_start.to(self.device)
        batch_size = x_start.shape[0]
        
        # Sample random timesteps and noise
        timesteps = torch.randint(
            0, self.scheduler.num_timesteps, (batch_size,), device=self.device
        )
        noise = torch.randn_like(x_start)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.mixed_precision):
            # Create noisy images
            x_t = self.scheduler.add_noise(x_start, noise, timesteps)
            
            # Predict noise
            model_output = self.model(x_t, timesteps)
            
            # Compute loss
            loss_dict = self.loss_fn(
                model_output, x_start, noise, timesteps, return_dict=True
            )
            loss = loss_dict["loss"]
        
        # Backward pass
        self.optimizer.zero_grad()
        
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            # Gradient clipping
            if self.gradient_clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.gradient_clip_val > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_val)
            self.optimizer.step()
        
        # Update EMA
        if self.ema_model is not None:
            self.ema_model.update(self.model)
        
        # Update learning rate
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        
        # Return metrics
        return {
            "loss": loss.item(),
            "lr": self.optimizer.param_groups[0]["lr"],
            **{k: v for k, v in loss_dict["loss_dict"].items() if isinstance(v, (int, float))}
        }
    
    def validate(self) -> Dict[str, float]:
        """Validation step."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    x_start = batch[0]
                else:
                    x_start = batch
                
                x_start = x_start.to(self.device)
                batch_size = x_start.shape[0]
                
                # Sample random timesteps and noise
                timesteps = torch.randint(
                    0, self.scheduler.num_timesteps, (batch_size,), device=self.device
                )
                noise = torch.randn_like(x_start)
                
                # Forward pass
                x_t = self.scheduler.add_noise(x_start, noise, timesteps)
                model_output = self.model(x_t, timesteps)
                
                # Compute loss
                loss_dict = self.loss_fn(
                    model_output, x_start, noise, timesteps, return_dict=True
                )
                val_losses.append(loss_dict["loss"].item())
        
        self.model.train()
        return {"val_loss": np.mean(val_losses)}
    
    def sample_images(self, num_samples: Optional[int] = None) -> torch.Tensor:
        """Generate sample images."""
        if num_samples is None:
            num_samples = self.num_sample_images
        
        # Use EMA model if available
        model_to_use = self.model
        if self.ema_model is not None:
            self.ema_model.apply_shadow(self.model)
            model_to_use = self.model
        
        model_to_use.eval()
        
        with torch.no_grad():
            sampler = DDPMSampler(self.scheduler)
            
            # Determine shape from dataloader
            sample_batch = next(iter(self.train_dataloader))
            if isinstance(sample_batch, (list, tuple)):
                sample_x = sample_batch[0]
            else:
                sample_x = sample_batch
            
            _, c, h, w = sample_x.shape
            shape = (num_samples, c, h, w)
            
            # Sample
            result = sampler.p_sample_loop(
                model_to_use, shape, 
                num_inference_steps=min(50, self.scheduler.num_timesteps),
                device=self.device, progress=False
            )
            samples = result["images"]
        
        # Restore model if using EMA
        if self.ema_model is not None:
            self.ema_model.restore(self.model)
        
        model_to_use.train()
        return samples
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f"model_epoch_{epoch:03d}.pth" if not is_best else "model_best.pth"
        )
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=0.0,  # Will be filled by validation
            path=checkpoint_path,
            scheduler=self.lr_scheduler,
            ema_model=self.ema_model,
            global_step=self.global_step,
            config=self.config
        )
        
        # Always save latest
        latest_path = os.path.join(self.checkpoint_dir, "model_latest.pth")
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=0.0,
            path=latest_path,
            scheduler=self.lr_scheduler,
            ema_model=self.ema_model,
            global_step=self.global_step,
            config=self.config
        )
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.epochs} epochs on {self.device}")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        if self.val_dataloader:
            print(f"Validation samples: {len(self.val_dataloader.dataset)}")
        
        self.model.train()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_losses = []
            
            # Training loop
            progress_bar = tqdm(
                self.train_dataloader, 
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=False
            )
            
            for batch_idx, batch in enumerate(progress_bar):
                # Training step
                step_metrics = self.train_step(batch)
                epoch_losses.append(step_metrics["loss"])
                self.global_step += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{step_metrics['loss']:.4f}",
                    "lr": f"{step_metrics['lr']:.2e}"
                })
                
                # Log metrics
                if self.global_step % self.log_every == 0:
                    for key, value in step_metrics.items():
                        self.writer.add_scalar(f"Train/{key}", value, self.global_step)
            
            # Epoch statistics
            epoch_loss = np.mean(epoch_losses)
            
            # Validation
            val_metrics = self.validate()
            val_loss = val_metrics.get("val_loss", 0.0)
            
            # Log epoch metrics
            self.writer.add_scalar("Epoch/TrainLoss", epoch_loss, epoch)
            if val_metrics:
                self.writer.add_scalar("Epoch/ValLoss", val_loss, epoch)
            
            print(f"Epoch {epoch+1}: train_loss={epoch_loss:.4f}", end="")
            if val_metrics:
                print(f", val_loss={val_loss:.4f}", end="")
            print()
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss if val_metrics else False
            if is_best:
                self.best_val_loss = val_loss
            
            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best=is_best)
            
            # Generate samples
            if (epoch + 1) % self.sample_every == 0:
                try:
                    samples = self.sample_images()
                    sample_path = os.path.join(
                        self.sample_dir, f"samples_epoch_{epoch+1:03d}.png"
                    )
                    save_image_grid(samples, sample_path, nrow=4, normalize=True)
                    
                    # Log samples to tensorboard
                    self.writer.add_images("Samples", samples, epoch, normalize=True)
                except Exception as e:
                    print(f"Failed to generate samples: {e}")
        
        # Final checkpoint
        self.save_checkpoint(self.epochs, is_best=False)
        
        # Final samples
        try:
            samples = self.sample_images(64)  # More samples for final
            final_sample_path = os.path.join(self.sample_dir, "samples_final.png")
            save_image_grid(samples, final_sample_path, nrow=8, normalize=True)
        except Exception as e:
            print(f"Failed to generate final samples: {e}")
        
        self.writer.close()
        print("Training completed!")


def test_trainer():
    """Test trainer functionality."""
    from .ddpm_schedules import DDPMScheduler
    from .models.unet_tiny import UNetTiny
    from .dataset import create_mnist_dataset, create_dataloader
    
    # Create components
    scheduler = DDPMScheduler(num_timesteps=100)
    model = UNetTiny(in_channels=1, out_channels=1, model_channels=32)
    
    # Create dummy dataloader
    train_dataset, _ = create_mnist_dataset(root="data", download=False, return_labels=False)
    train_loader = create_dataloader(train_dataset, batch_size=4, num_workers=0)
    
    # Test config
    config = {
        "training": {
            "epochs": 2,
            "learning_rate": 1e-3,
            "log_every": 10,
            "save_every": 1,
            "sample_every": 1
        },
        "ddpm": {
            "prediction_type": "epsilon"
        }
    }
    
    # Create trainer
    trainer = DDPMTrainer(model, scheduler, train_loader, config=config)
    
    # Test single step (don't run full training)
    batch = next(iter(train_loader))
    metrics = trainer.train_step(batch)
    print(f"Training step metrics: {metrics}")
    
    print("Trainer test completed!")


if __name__ == "__main__":
    test_trainer()