"""
DDPM loss functions with ε-prediction and optional parameterizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable

from .ddpm_schedules import DDPMSchedules


class DDPMLoss(nn.Module):
    """
    DDPM Simple Loss: L = ||ε - ε_θ(x_t, t)||²
    
    Supports different parameterizations:
    - eps: predict noise ε (default)
    - x0: predict x_0 directly  
    - v: predict velocity v = α_t * ε - σ_t * x_0
    """
    
    def __init__(
        self,
        schedules: DDPMSchedules,
        parameterization: str = "eps",
        loss_type: str = "l2",
        lambda_vlb: float = 0.0,  # Variational lower bound weight
        lambda_simple: float = 1.0  # Simple loss weight
    ):
        super().__init__()
        
        self.schedules = schedules
        self.parameterization = parameterization.lower()
        self.loss_type = loss_type.lower()
        self.lambda_vlb = lambda_vlb
        self.lambda_simple = lambda_simple
        
        assert self.parameterization in ["eps", "x0", "v"]
        assert self.loss_type in ["l1", "l2", "huber"]
        
    def _get_target(
        self, 
        x_start: torch.Tensor, 
        noise: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """Get target based on parameterization"""
        if self.parameterization == "eps":
            return noise
        elif self.parameterization == "x0":
            return x_start
        elif self.parameterization == "v":
            # v-parameterization: v = α_t * ε - σ_t * x_0
            sqrt_alphas_cumprod_t = self.schedules.extract(
                self.schedules.sqrt_alphas_cumprod, t, x_start.shape
            )
            sqrt_one_minus_alphas_cumprod_t = self.schedules.extract(
                self.schedules.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            return (sqrt_alphas_cumprod_t * noise - 
                   sqrt_one_minus_alphas_cumprod_t * x_start)
            
    def _get_prediction_x0(
        self, 
        x_t: torch.Tensor, 
        model_output: torch.Tensor, 
        t: torch.Tensor
    ) -> torch.Tensor:
        """Convert model output to x_0 prediction"""
        if self.parameterization == "x0":
            return model_output
        elif self.parameterization == "eps":
            # x_0 = (x_t - σ_t * ε) / α_t
            sqrt_alphas_cumprod_t = self.schedules.extract(
                self.schedules.sqrt_alphas_cumprod, t, x_t.shape
            )
            sqrt_one_minus_alphas_cumprod_t = self.schedules.extract(
                self.schedules.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )
            return (x_t - sqrt_one_minus_alphas_cumprod_t * model_output) / sqrt_alphas_cumprod_t
        elif self.parameterization == "v":
            # From v-parameterization to x_0
            sqrt_alphas_cumprod_t = self.schedules.extract(
                self.schedules.sqrt_alphas_cumprod, t, x_t.shape
            )
            sqrt_one_minus_alphas_cumprod_t = self.schedules.extract(
                self.schedules.sqrt_one_minus_alphas_cumprod, t, x_t.shape
            )
            return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * model_output
            
    def _compute_loss(self, target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        """Compute loss based on loss_type"""
        if self.loss_type == "l1":
            return F.l1_loss(prediction, target, reduction='none')
        elif self.loss_type == "l2":
            return F.mse_loss(prediction, target, reduction='none')
        elif self.loss_type == "huber":
            return F.huber_loss(prediction, target, reduction='none', delta=1.0)
            
    def forward(
        self, 
        model: Callable,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_dict: bool = False
    ) -> torch.Tensor:
        """
        Compute DDPM loss
        
        Args:
            model: denoising model ε_θ(x_t, t)
            x_start: clean images [B, C, H, W]
            t: timesteps [B], if None sample uniformly
            noise: noise tensor [B, C, H, W], if None sample from N(0,I)
            return_dict: whether to return detailed dict
            
        Returns:
            loss: scalar loss value or dict with detailed losses
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample timesteps uniformly
        if t is None:
            t = torch.randint(0, self.schedules.num_timesteps, (batch_size,), device=device)
            
        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Forward process: q(x_t | x_0)
        x_t = self.schedules.q_sample(x_start, t, noise)
        
        # Model prediction
        model_output = model(x_t, t)
        
        # Get target based on parameterization
        target = self._get_target(x_start, noise, t)
        
        # Compute simple loss
        simple_loss = self._compute_loss(target, model_output)
        
        # Reduce over spatial dimensions, keep batch
        simple_loss = simple_loss.flatten(1).mean(dim=1)
        
        total_loss = self.lambda_simple * simple_loss.mean()
        
        if not return_dict:
            return total_loss
            
        # Prepare detailed output
        result = {
            "total_loss": total_loss,
            "simple_loss": simple_loss.mean(),
            "model_output": model_output,
            "target": target,
            "x_t": x_t,
            "t": t,
            "noise": noise
        }
        
        # Add predicted x_0 for analysis
        result["pred_x0"] = self._get_prediction_x0(x_t, model_output, t)
        
        return result


class VLBLoss(nn.Module):
    """
    Variational Lower Bound loss for DDPM
    L_VLB = L_0 + L_1 + ... + L_{T-1} + L_T
    where L_t = D_KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t))
    """
    
    def __init__(self, schedules: DDPMSchedules):
        super().__init__()
        self.schedules = schedules
        
    def forward(
        self, 
        model: Callable,
        x_start: torch.Tensor,
        clip_denoised: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute full VLB loss
        Warning: This is expensive as it requires T forward passes!
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample all timesteps
        t = torch.arange(self.schedules.num_timesteps, device=device)
        t = t.unsqueeze(0).expand(batch_size, -1).contiguous().view(-1)
        
        # Expand x_start to match
        x_start_expanded = x_start.unsqueeze(1).expand(-1, self.schedules.num_timesteps, -1, -1, -1)
        x_start_expanded = x_start_expanded.contiguous().view(-1, *x_start.shape[1:])
        
        # Sample noise
        noise = torch.randn_like(x_start_expanded)
        
        # Forward process for all timesteps
        x_t = self.schedules.q_sample(x_start_expanded, t, noise)
        
        # Model predictions
        model_output = model(x_t, t)
        
        # This is a simplified version - full VLB requires careful handling
        # of the KL divergences at each timestep
        vlb_loss = F.mse_loss(model_output, noise)
        
        return {"vlb_loss": vlb_loss}


class FocalLoss(nn.Module):
    """
    Focal loss for DDPM - puts more weight on hard timesteps
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, loss: torch.Tensor, model_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Apply focal weighting to existing loss
        """
        # Compute prediction error as proxy for "hardness"
        error = F.mse_loss(model_output, target, reduction='none').flatten(1).mean(dim=1)
        
        # Focal weight: (1 - pt)^gamma where pt is "confidence"  
        pt = torch.exp(-error)  # Higher error = lower confidence
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        return (focal_weight * loss).mean()


def get_loss_fn(
    schedules: DDPMSchedules,
    loss_config: Dict
) -> nn.Module:
    """Factory function to create loss function from config"""
    
    loss_type = loss_config.get("type", "simple")
    
    if loss_type == "simple":
        return DDPMLoss(
            schedules=schedules,
            parameterization=loss_config.get("parameterization", "eps"),
            loss_type=loss_config.get("loss_type", "l2"),
            lambda_simple=loss_config.get("lambda_simple", 1.0)
        )
    elif loss_type == "vlb":
        return VLBLoss(schedules=schedules)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")