"""
DDPM trainer with EMA, AMP, learning rate scheduling, and periodic sampling.
Identical training loop across all beta schedules for fair comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import pandas as pd

from .models.unet_small import UNetSmall
from .schedules import get_schedule
from .losses import DDPMLoss, compute_forward_process
from .utils import (
    EMAModel, save_checkpoint, load_checkpoint, save_image_grid,
    setup_logging_dir, count_parameters, compute_model_size
)
from .sampler import DDPMSampler


class DDPMTrainer:
    """
    DDPM trainer with support for different beta schedules.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Setup directories
        self.run_dir = Path(config['log']['out_root']) / config['log']['run_name']
        setup_logging_dir(self.run_dir)
        
        # Initialize components
        self._init_model()
        self._init_schedule()
        self._init_optimizer()
        self._init_loss()
        self._init_logging()
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        # Metrics tracking
        self.train_losses = []
        self.epoch_times = []
        self.lr_history = []
    
    def _init_model(self):
        """Initialize model and EMA."""
        model_config = self.config['model']
        
        self.model = UNetSmall(
            in_channels=model_config['in_ch'],
            out_channels=model_config['in_ch'],  # Same as input for noise prediction
            base_channels=model_config['base_ch'],
            channel_multipliers=model_config['ch_mult'],
            num_res_blocks=model_config.get('num_res_blocks', 2),
            time_embed_dim=model_config['time_embed_dim'],
            dropout=model_config.get('dropout', 0.1),
            attention_resolutions=model_config.get('attn_resolutions', [16])
        ).to(self.device)
        
        # EMA model
        if self.config['train']['ema']:
            self.ema_model = EMAModel(
                self.model, 
                decay=self.config['train']['ema_decay']
            )
        else:
            self.ema_model = None
        
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Model size: {compute_model_size(self.model):.2f} MB")
    
    def _init_schedule(self):
        """Initialize diffusion schedule."""
        diffusion_config = self.config['diffusion']
        
        self.T = diffusion_config['T']
        schedule_name = diffusion_config['schedule']
        
        # Get schedule parameters based on schedule type
        schedule_kwargs = {}
        if schedule_name in ['linear', 'quadratic']:
            if 'beta_min' in diffusion_config:
                schedule_kwargs['beta_min'] = diffusion_config['beta_min']
            if 'beta_max' in diffusion_config:
                schedule_kwargs['beta_max'] = diffusion_config['beta_max']
        elif schedule_name == 'cosine':
            if 'cosine_s' in diffusion_config:
                schedule_kwargs['s'] = diffusion_config['cosine_s']
        
        # Create schedule
        schedule = get_schedule(schedule_name, self.T, **schedule_kwargs)
        
        # Register as buffers
        for key, tensor in schedule.items():
            self.register_buffer(key, tensor.to(self.device))
        
        print(f"Diffusion schedule: {schedule_name} with T={self.T}")
    
    def register_buffer(self, name: str, tensor: torch.Tensor):
        """Register tensor as buffer in model."""
        self.model.register_buffer(name, tensor)
    
    def _init_optimizer(self):
        """Initialize optimizer and scheduler."""
        train_config = self.config['train']
        
        # Optimizer
        if train_config['opt'].lower() == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=train_config['lr'],
                weight_decay=train_config['wd']
            )
        elif train_config['opt'].lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=train_config['lr'],
                weight_decay=train_config['wd']
            )
        else:
            raise ValueError(f"Unknown optimizer: {train_config['opt']}")
        
        # Learning rate scheduler (cosine annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=train_config['epochs']
        )
        
        # Gradient scaler for AMP
        if train_config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def _init_loss(self):
        """Initialize loss function."""
        self.loss_fn = DDPMLoss()
    
    def _init_logging(self):
        """Initialize logging."""
        if self.config['log'].get('tensorboard', True):
            self.writer = SummaryWriter(self.run_dir / 'logs')
        else:
            self.writer = None
        
        # Metrics storage
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'lr': [],
            'epoch_time': [],
            'step': []
        }
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.epoch + 1}")
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)
            batch_size = images.shape[0]
            
            # Sample random timesteps
            timesteps = torch.randint(
                0, self.T, (batch_size,), device=self.device, dtype=torch.long
            )
            
            # Sample noise
            noise = torch.randn_like(images)
            
            # Forward diffusion process
            noisy_images, _ = compute_forward_process(
                images, timesteps, self.model.alpha_bars, noise
            )
            
            # Training step
            if self.scaler is not None:
                # AMP training
                with torch.cuda.amp.autocast():
                    noise_pred = self.model(noisy_images, timesteps)
                    loss = self.loss_fn(noise_pred, noise)
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                if self.config['train'].get('grad_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['train']['grad_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular training
                noise_pred = self.model(noisy_images, timesteps)
                loss = self.loss_fn(noise_pred, noise)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config['train'].get('grad_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['train']['grad_clip']
                    )
                
                self.optimizer.step()
            
            # Update EMA
            if self.ema_model is not None:
                self.ema_model.update()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config['log']['log_every'] == 0:
                if self.writer is not None:
                    self.writer.add_scalar('Loss/Train', loss.item(), self.global_step)
                    self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
        
        return total_loss / num_batches
    
    def sample_images(self, num_images: int = 64) -> torch.Tensor:
        """Sample images using current model."""
        sampler = DDPMSampler(
            self.model,
            betas=self.model.betas,
            alphas=self.model.alphas, 
            alpha_bars=self.model.alpha_bars
        )
        
        if self.ema_model is not None:
            self.ema_model.apply_shadow()
        
        self.model.eval()
        with torch.no_grad():
            # Get image shape from config
            data_config = self.config['data']
            if data_config['dataset'].lower() == 'mnist':
                shape = (num_images, 1, 28, 28)
            elif data_config['dataset'].lower() == 'cifar10':
                shape = (num_images, 3, 32, 32)
            else:
                raise ValueError(f"Unknown dataset: {data_config['dataset']}")
            
            samples = sampler.sample(shape, self.device)
        
        if self.ema_model is not None:
            self.ema_model.restore()
        
        return samples
    
    def save_sample_grid(self, epoch: int, num_images: int = 64):
        """Generate and save sample grid."""
        samples = self.sample_images(num_images)
        
        # Save grid
        grid_path = self.run_dir / 'grids' / f'samples_ep{epoch:03d}.png'
        save_image_grid(samples, grid_path, nrow=8)
        
        print(f"Sample grid saved to {grid_path}")
    
    def save_checkpoint_epoch(self, epoch: int, loss: float):
        """Save checkpoint."""
        # Save regular checkpoint
        ckpt_path = self.run_dir / 'ckpts' / f'model_ep{epoch:03d}.pt'
        save_checkpoint(
            self.model, self.ema_model, self.optimizer, 
            epoch, loss, ckpt_path
        )
        
        # Save EMA checkpoint separately
        if self.ema_model is not None:
            ema_path = self.run_dir / 'ckpts' / f'ema_ep{epoch:03d}.pt'
            
            # Temporarily apply EMA weights
            self.ema_model.apply_shadow()
            save_checkpoint(
                self.model, None, None, 
                epoch, loss, ema_path
            )
            self.ema_model.restore()
        
        # Save best model
        if loss < self.best_loss:
            self.best_loss = loss
            best_path = self.run_dir / 'ckpts' / 'best.pt'
            save_checkpoint(
                self.model, self.ema_model, self.optimizer,
                epoch, loss, best_path
            )
            
            if self.ema_model is not None:
                best_ema_path = self.run_dir / 'ckpts' / 'ema.pt'
                self.ema_model.apply_shadow()
                save_checkpoint(
                    self.model, None, None,
                    epoch, loss, best_ema_path
                )
                self.ema_model.restore()
    
    def save_metrics(self):
        """Save metrics to CSV."""
        df = pd.DataFrame(self.metrics)
        csv_path = self.run_dir / 'logs' / 'metrics.csv'
        df.to_csv(csv_path, index=False)
    
    def train(self, dataloader: DataLoader) -> None:
        """Main training loop."""
        print(f"Starting training for {self.config['train']['epochs']} epochs")
        print(f"Run directory: {self.run_dir}")
        
        for epoch in range(self.config['train']['epochs']):
            self.epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            avg_loss = self.train_epoch(dataloader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.metrics['epoch'].append(epoch)
            self.metrics['train_loss'].append(avg_loss)
            self.metrics['lr'].append(current_lr)
            self.metrics['epoch_time'].append(epoch_time)
            self.metrics['step'].append(self.global_step)
            
            # Logging
            print(f"Epoch {epoch + 1}/{self.config['train']['epochs']} - "
                  f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            if self.writer is not None:
                self.writer.add_scalar('Loss/EpochAvg', avg_loss, epoch)
                self.writer.add_scalar('Time/Epoch', epoch_time, epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.config['train'].get('save_every', 10) == 0:
                self.save_checkpoint_epoch(epoch, avg_loss)
            
            # Generate samples
            if (epoch + 1) % self.config['train'].get('sample_every', 5) == 0:
                self.save_sample_grid(epoch)
        
        # Final checkpoint and samples
        self.save_checkpoint_epoch(self.config['train']['epochs'] - 1, avg_loss)
        self.save_sample_grid(self.config['train']['epochs'] - 1)
        self.save_metrics()
        
        if self.writer is not None:
            self.writer.close()
        
        print("Training completed!")


def test_trainer():
    """Test trainer with minimal config."""
    # Minimal config for testing
    config = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'data': {'dataset': 'mnist', 'normalize': 'minus_one_one'},
        'diffusion': {'T': 100, 'schedule': 'linear'},
        'model': {
            'in_ch': 1, 'base_ch': 32, 'ch_mult': [1, 2], 
            'time_embed_dim': 128, 'dropout': 0.1
        },
        'train': {
            'epochs': 2, 'lr': 1e-3, 'opt': 'adamw', 'wd': 0.01,
            'amp': False, 'ema': True, 'ema_decay': 0.99,
            'grad_clip': 1.0, 'save_every': 1, 'sample_every': 1
        },
        'log': {'out_root': './test_outputs', 'run_name': 'test', 'log_every': 10}
    }
    
    # Create trainer
    trainer = DDPMTrainer(config)
    
    # Create dummy dataloader
    from torch.utils.data import TensorDataset
    dummy_images = torch.randn(100, 1, 28, 28)
    dummy_labels = torch.zeros(100)
    dataset = TensorDataset(dummy_images, dummy_labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Train
    trainer.train(dataloader)
    
    print("Trainer test completed!")


if __name__ == "__main__":
    test_trainer()
