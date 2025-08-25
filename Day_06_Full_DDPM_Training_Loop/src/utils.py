"""
Utility functions: seeding, device management, checkpoint I/O, EMA, logging, etc.
"""

import os
import random
import logging
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.utils as vutils


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get appropriate device (CPU/CUDA/MPS)"""
    if device_str is not None and device_str != "auto":
        return torch.device(device_str)
        
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")  
    else:
        return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class EMAModel:
    """
    Exponential Moving Average of model parameters
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        decay: float = 0.999,
        device: Optional[torch.device] = None
    ):
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create EMA parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
                
    def update(self, model: nn.Module):
        """Update EMA parameters"""
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = (
                        self.decay * self.shadow[name] +
                        (1.0 - self.decay) * param.data.to(self.device)
                    )
                    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA parameters to model"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
                
    def restore(self, model: nn.Module):
        """Restore original parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}
        
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get EMA state dict"""
        return self.shadow
        
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]):
        """Load EMA state dict"""
        self.shadow = {k: v.to(self.device) for k, v in state_dict.items()}


class CheckpointManager:
    """
    Handles saving and loading of model checkpoints
    """
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        epoch: int,
        step: int,
        loss: float,
        ema: Optional[EMAModel] = None,
        metadata: Optional[Dict] = None,
        filename: Optional[str] = None
    ):
        """Save checkpoint"""
        if filename is None:
            filename = f"checkpoint_epoch_{epoch:04d}.pt"
            
        checkpoint = {
            "epoch": epoch,
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "metadata": metadata or {}
        }
        
        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
            
        if ema is not None:
            checkpoint["ema_state_dict"] = ema.state_dict()
            
        filepath = self.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        
        # Also save as latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        return filepath
        
    def load_checkpoint(
        self,
        filename: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        ema: Optional[EMAModel] = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Load checkpoint"""
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Checkpoint not found: {filepath}")
            
        checkpoint = torch.load(filepath, map_location=device)
        
        # Load model
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        # Load scheduler  
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load EMA
        if ema is not None and "ema_state_dict" in checkpoint:
            ema.load_state_dict(checkpoint["ema_state_dict"])
            
        return {
            "epoch": checkpoint["epoch"],
            "step": checkpoint["step"], 
            "loss": checkpoint["loss"],
            "metadata": checkpoint.get("metadata", {})
        }
        
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get latest checkpoint filename"""
        latest_path = self.checkpoint_dir / "latest.pt"
        if latest_path.exists():
            return "latest.pt"
        return None


class Logger:
    """
    Simple logger for training metrics
    """
    
    def __init__(self, log_dir: str, name: str = "training"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / f"{name}.log")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        # Metrics storage
        self.metrics = []
        
    def info(self, msg: str):
        """Log info message"""
        self.logger.info(msg)
        
    def warning(self, msg: str):
        """Log warning message"""
        self.logger.warning(msg)
        
    def error(self, msg: str):
        """Log error message"""
        self.logger.error(msg)
        
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """Log metrics dict"""
        # Add to storage
        metric_entry = {"step": step, **metrics}
        if prefix:
            metric_entry = {f"{prefix}/{k}" if k != "step" else k: v for k, v in metric_entry.items()}
        self.metrics.append(metric_entry)
        
        # Log to console/file
        metric_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
        self.info(f"Step {step} | {metric_str}")
        
    def save_metrics(self, filename: str = "metrics.json"):
        """Save metrics to JSON file"""
        filepath = self.log_dir / filename
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
            
    def load_metrics(self, filename: str = "metrics.json") -> List[Dict]:
        """Load metrics from JSON file"""
        filepath = self.log_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                self.metrics = json.load(f)
        return self.metrics


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_image_grid(
    images: torch.Tensor,
    filepath: str,
    nrow: int = 8,
    normalize: bool = True,
    value_range: Optional[tuple] = None
):
    """Save tensor as image grid"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    if normalize and value_range is None:
        value_range = (-1, 1) if images.min() < 0 else (0, 1)
        
    vutils.save_image(
        images, 
        filepath, 
        nrow=nrow, 
        normalize=normalize, 
        value_range=value_range,
        pad_value=0.5
    )


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    # Assume tensor is [C, H, W] in range [-1, 1] or [0, 1]
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2  # [-1, 1] -> [0, 1]
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    numpy_img = tensor.detach().cpu().numpy()
    
    # Transpose to HWC format
    if numpy_img.shape[0] in [1, 3]:  # CHW format
        numpy_img = np.transpose(numpy_img, (1, 2, 0))
        
    # Convert to uint8
    numpy_img = (numpy_img * 255).astype(np.uint8)
    
    # Handle grayscale
    if numpy_img.shape[2] == 1:
        numpy_img = numpy_img.squeeze(2)
        
    return Image.fromarray(numpy_img)


class WarmupScheduler:
    """
    Learning rate warmup scheduler
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        base_lr: float,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_count = 0
        
    def step(self):
        """Update learning rate"""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            lr = self.min_lr + (self.base_lr - self.min_lr) * self.step_count / self.warmup_steps
        else:
            lr = self.base_lr
            
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr


def create_output_dirs(base_dir: str):
    """Create all necessary output directories"""
    base_path = Path(base_dir)
    
    subdirs = [
        "ckpts", "logs", "curves", "grids", 
        "animations", "reports", "samples"
    ]
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(parents=True, exist_ok=True)