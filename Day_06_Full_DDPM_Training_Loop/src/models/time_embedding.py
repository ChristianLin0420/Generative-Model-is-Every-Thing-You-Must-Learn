"""
Sinusoidal time embeddings for DDPM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embeddings from Attention is All You Need"""
    
    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: timestep tensor [B] or [B, 1]
        Returns:
            embeddings: [B, dim]
        """
        if x.dim() > 1:
            x = x.squeeze(-1)
            
        device = x.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        
        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
            
        return embeddings


class TimeEmbedding(nn.Module):
    """
    Time embedding module: sinusoidal embedding + MLP
    Maps timestep integers to conditioning vectors
    """
    
    def __init__(
        self, 
        embed_dim: int = 128,
        hidden_dim: Optional[int] = None,
        activation: str = "silu",
        dropout: float = 0.0
    ):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = embed_dim * 4
            
        self.embed_dim = embed_dim
        
        # Sinusoidal position embedding
        self.pos_emb = SinusoidalPosEmb(embed_dim)
        
        # MLP to transform embeddings
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        if name.lower() == "silu":
            return nn.SiLU()
        elif name.lower() == "relu":
            return nn.ReLU()
        elif name.lower() == "gelu":
            return nn.GELU()
        elif name.lower() == "swish":
            return nn.SiLU()  # SiLU is Swish
        else:
            raise ValueError(f"Unknown activation: {name}")
            
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] or [B, 1] timestep indices
        Returns:
            embeddings: [B, embed_dim] time conditioning vectors
        """
        # Get sinusoidal embeddings
        emb = self.pos_emb(timesteps)
        
        # Transform through MLP
        emb = self.mlp(emb)
        
        return emb


class TimestepBlock(nn.Module):
    """
    Abstract base for any module that conditions on timesteps
    """
    
    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor
            emb: time embedding [B, embed_dim]
        Returns:
            output tensor
        """
        raise NotImplementedError
        

class TimestepEmbedSequential(nn.Sequential):
    """
    Sequential module that passes time embeddings to TimestepBlock layers
    """
    
    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x