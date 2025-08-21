"""
Loss functions for Day 2: Denoising Autoencoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ReconstructionLoss(nn.Module):
    """Flexible reconstruction loss supporting multiple types."""
    
    def __init__(self, loss_type: str = "l2", reduction: str = "mean"):
        super().__init__()
        self.loss_type = loss_type.lower()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "l1":
            return F.l1_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == "l2" or self.loss_type == "mse":
            return F.mse_loss(pred, target, reduction=self.reduction)
        elif self.loss_type == "charbonnier":
            return self._charbonnier_loss(pred, target)
        elif self.loss_type == "huber":
            return F.huber_loss(pred, target, reduction=self.reduction)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
    def _charbonnier_loss(self, pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Charbonnier loss: sqrt((pred - target)^2 + eps^2)"""
        diff = pred - target
        loss = torch.sqrt(diff * diff + eps * eps)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    Useful for RGB datasets.
    """
    
    def __init__(
        self,
        feature_layers: list = [0, 5, 10, 19, 28],  # VGG-19 layer indices
        weights: list = [1.0, 1.0, 1.0, 1.0, 1.0],
        normalize: bool = True
    ):
        super().__init__()
        
        # Load pre-trained VGG-19
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # Freeze VGG parameters
        for param in vgg.parameters():
            param.requires_grad = False
        
        # Extract feature layers
        self.feature_extractors = nn.ModuleList()
        current_layers = []
        layer_idx = 0
        
        for i, layer in enumerate(vgg):
            current_layers.append(layer)
            if i in feature_layers:
                self.feature_extractors.append(nn.Sequential(*current_layers))
                current_layers = []
        
        self.weights = weights[:len(self.feature_extractors)]
        self.normalize = normalize
        
        # VGG normalization
        self.register_buffer(
            'vgg_mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'vgg_std', 
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
    
    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input for VGG."""
        # Convert grayscale to RGB if needed
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Normalize for VGG
        if self.normalize:
            x = (x - self.vgg_mean) / self.vgg_std
        
        return x
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = self._preprocess(pred)
        target = self._preprocess(target)
        
        loss = 0.0
        
        for i, extractor in enumerate(self.feature_extractors):
            pred_features = extractor(pred)
            target_features = extractor(target)
            
            layer_loss = F.mse_loss(pred_features, target_features)
            loss += self.weights[i] * layer_loss
        
        return loss


class CombinedLoss(nn.Module):
    """
    Combined loss function with multiple components.
    """
    
    def __init__(
        self,
        reconstruction_type: str = "l2",
        perceptual_weight: float = 0.1,
        use_perceptual: bool = False
    ):
        super().__init__()
        
        self.reconstruction_loss = ReconstructionLoss(reconstruction_type)
        self.use_perceptual = use_perceptual
        self.perceptual_weight = perceptual_weight
        
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Returns dictionary with loss components for logging.
        """
        losses = {}
        
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(pred, target)
        losses['reconstruction'] = recon_loss
        
        total_loss = recon_loss
        
        # Perceptual loss
        if self.use_perceptual:
            perc_loss = self.perceptual_loss(pred, target)
            losses['perceptual'] = perc_loss
            total_loss += self.perceptual_weight * perc_loss
        
        losses['total'] = total_loss
        return losses


class EdgePreservingLoss(nn.Module):
    """
    Loss function that encourages edge preservation.
    """
    
    def __init__(self, edge_weight: float = 0.1):
        super().__init__()
        self.edge_weight = edge_weight
        
        # Sobel filters for edge detection
        self.register_buffer('sobel_x', torch.tensor([
            [[-1, 0, 1],
             [-2, 0, 2], 
             [-1, 0, 1]]
        ], dtype=torch.float32).unsqueeze(0))
        
        self.register_buffer('sobel_y', torch.tensor([
            [[-1, -2, -1],
             [0,  0,  0],
             [1,  2,  1]]
        ], dtype=torch.float32).unsqueeze(0))
    
    def _compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude using Sobel filters."""
        if x.size(1) > 1:
            # Convert to grayscale for multi-channel images
            x = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Apply Sobel filters
        grad_x = F.conv2d(x, self.sobel_x, padding=1)
        grad_y = F.conv2d(x, self.sobel_y, padding=1)
        
        # Compute magnitude
        edges = torch.sqrt(grad_x**2 + grad_y**2)
        return edges
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Standard reconstruction loss
        recon_loss = F.mse_loss(pred, target)
        
        # Edge preservation loss
        pred_edges = self._compute_edges(pred)
        target_edges = self._compute_edges(target)
        edge_loss = F.mse_loss(pred_edges, target_edges)
        
        total_loss = recon_loss + self.edge_weight * edge_loss
        return total_loss


class GradientLoss(nn.Module):
    """
    Loss based on image gradients for sharpness preservation.
    """
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def _gradient(self, x: torch.Tensor) -> tuple:
        """Compute image gradients."""
        # Horizontal gradient
        grad_h = x[:, :, :, 1:] - x[:, :, :, :-1]
        # Vertical gradient  
        grad_v = x[:, :, 1:, :] - x[:, :, :-1, :]
        return grad_h, grad_v
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Image-space loss
        img_loss = F.l1_loss(pred, target)
        
        # Gradient loss
        pred_grad_h, pred_grad_v = self._gradient(pred)
        target_grad_h, target_grad_v = self._gradient(target)
        
        grad_loss = (
            F.l1_loss(pred_grad_h, target_grad_h) +
            F.l1_loss(pred_grad_v, target_grad_v)
        )
        
        return img_loss + self.alpha * grad_loss


def get_loss_function(
    loss_type: str = "l2",
    perceptual_weight: float = 0.0,
    **kwargs
) -> nn.Module:
    """Factory function to create loss functions."""
    
    if loss_type in ["l1", "l2", "mse", "charbonnier", "huber"]:
        if perceptual_weight > 0:
            return CombinedLoss(
                reconstruction_type=loss_type,
                perceptual_weight=perceptual_weight,
                use_perceptual=True
            )
        else:
            return ReconstructionLoss(loss_type)
    
    elif loss_type == "perceptual":
        return PerceptualLoss(**kwargs)
    
    elif loss_type == "edge_preserving":
        return EdgePreservingLoss(**kwargs)
    
    elif loss_type == "gradient":
        return GradientLoss(**kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")