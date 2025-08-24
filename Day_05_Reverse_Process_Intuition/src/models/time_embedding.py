"""Time embedding for diffusion models

Implements sinusoidal positional encoding for timestep conditioning
in neural networks, similar to Transformer positional encodings.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional


def sinusoidal_time_embed(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Create sinusoidal time embeddings.
    
    This function creates time embeddings using sine and cosine functions
    at different frequencies, similar to positional encodings in Transformers.
    
    Args:
        timesteps: Timestep values [B] or [B, 1]
        dim: Embedding dimension (should be even)
    
    Returns:
        Time embeddings [B, dim]
    
    Example:
        >>> t = torch.tensor([0, 10, 50, 100])
        >>> emb = sinusoidal_time_embed(t, 128)
        >>> print(emb.shape)  # torch.Size([4, 128])
    """
    device = timesteps.device
    dtype = timesteps.dtype
    
    # Ensure timesteps is 1D
    if timesteps.dim() > 1:
        timesteps = timesteps.squeeze(-1)
    
    # Calculate frequencies
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=dtype, device=device) * -emb)
    
    # Create embeddings
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    
    # Handle odd dimensions
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    
    return emb


class TimeEmbedding(nn.Module):
    """Time embedding module with learnable projections.
    
    This module:
    1. Creates sinusoidal embeddings from timesteps
    2. Projects them through learned linear layers
    3. Applies activation functions
    
    This allows the model to learn how to use time information effectively.
    """
    
    def __init__(
        self, 
        time_embed_dim: int, 
        hidden_dim: Optional[int] = None,
        activation: str = "silu"
    ):
        """Initialize time embedding.
        
        Args:
            time_embed_dim: Dimension of time embeddings
            hidden_dim: Hidden dimension for MLP (defaults to 4 * time_embed_dim)
            activation: Activation function ('silu', 'relu', 'gelu')
        """
        super().__init__()
        
        self.time_embed_dim = time_embed_dim
        if hidden_dim is None:
            hidden_dim = time_embed_dim * 4
        
        # Activation function
        if activation == "silu":
            act_fn = nn.SiLU()
        elif activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, time_embed_dim),
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            timesteps: Timestep values [B] or [B, 1]
        
        Returns:
            Time embeddings [B, time_embed_dim]
        """
        # Create sinusoidal embeddings
        time_emb = sinusoidal_time_embed(timesteps, self.time_embed_dim)
        
        # Project through MLP
        time_emb = self.time_mlp(time_emb)
        
        return time_emb


class TimestepEmbedSequential(nn.Sequential):
    """Sequential module that passes timestep embeddings to child modules.
    
    This is useful for UNet blocks that need time conditioning.
    Each child module should accept (x, time_emb) as input.
    """
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time embedding.
        
        Args:
            x: Input tensor
            time_emb: Time embeddings
        
        Returns:
            Output tensor
        """
        for layer in self:
            if isinstance(layer, TimestepEmbedding):
                x = layer(x, time_emb)
            else:
                x = layer(x)
        return x


class TimestepEmbedding(nn.Module):
    """Base class for modules that use timestep embeddings.
    
    Modules that inherit from this should implement time-conditioned operations.
    """
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with time embedding.
        
        Args:
            x: Input tensor
            time_emb: Time embeddings
        
        Returns:
            Output tensor
        """
        raise NotImplementedError


class AdaGroupNorm(TimestepEmbedding):
    """Adaptive Group Normalization conditioned on timestep embeddings.
    
    This applies Group Normalization and then modulates the result
    using time embeddings via scale and shift parameters.
    
    Also known as FiLM (Feature-wise Linear Modulation).
    """
    
    def __init__(
        self, 
        num_groups: int, 
        num_channels: int, 
        time_embed_dim: int,
        eps: float = 1e-5
    ):
        """Initialize AdaGroupNorm.
        
        Args:
            num_groups: Number of groups for GroupNorm
            num_channels: Number of channels
            time_embed_dim: Time embedding dimension
            eps: Epsilon for numerical stability
        """
        super().__init__()
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        
        # Group normalization
        self.group_norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=False)
        
        # Time conditioning - predict scale and shift
        self.time_proj = nn.Linear(time_embed_dim, num_channels * 2)
        
        # Initialize time projection to identity transformation
        nn.init.zeros_(self.time_proj.weight)
        nn.init.zeros_(self.time_proj.bias)
        # Set initial scale to 1, shift to 0
        with torch.no_grad():
            self.time_proj.bias[:num_channels] = 1.0
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            time_emb: Time embeddings [B, time_embed_dim]
        
        Returns:
            Normalized and modulated tensor [B, C, H, W]
        """
        # Apply group normalization
        x = self.group_norm(x)
        
        # Get scale and shift from time embedding
        time_out = self.time_proj(time_emb)  # [B, 2*C]
        time_out = time_out.unsqueeze(-1).unsqueeze(-1)  # [B, 2*C, 1, 1]
        
        # Split into scale and shift
        scale, shift = time_out.chunk(2, dim=1)  # Each [B, C, 1, 1]
        
        # Apply FiLM transformation
        return x * scale + shift


def test_time_embedding():
    """Test time embedding functionality."""
    # Test sinusoidal embedding
    timesteps = torch.tensor([0, 10, 50, 100])
    emb = sinusoidal_time_embed(timesteps, 128)
    print(f"Sinusoidal embedding shape: {emb.shape}")
    print(f"Embedding range: [{emb.min():.3f}, {emb.max():.3f}]")
    
    # Test TimeEmbedding module
    time_embed = TimeEmbedding(256)
    time_emb = time_embed(timesteps)
    print(f"TimeEmbedding output shape: {time_emb.shape}")
    
    # Test AdaGroupNorm
    ada_norm = AdaGroupNorm(8, 64, 256)
    x = torch.randn(4, 64, 32, 32)
    x_norm = ada_norm(x, time_emb)
    print(f"AdaGroupNorm output shape: {x_norm.shape}")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_time_embedding()