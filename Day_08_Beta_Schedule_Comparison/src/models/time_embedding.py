"""
Time embedding for DDPM: sinusoidal position encoding + MLP projection.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalPositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for time steps.
    Based on "Attention Is All You Need" transformer positional encoding.
    """
    
    def __init__(self, embed_dim: int, max_timesteps: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_timesteps = max_timesteps
        
        # Create sinusoidal embedding table
        pe = torch.zeros(max_timesteps, embed_dim)
        position = torch.arange(0, max_timesteps, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            (-math.log(10000.0) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Get sinusoidal embeddings for given timesteps.
        
        Args:
            timesteps: Timestep indices [B] or [B, 1]
            
        Returns:
            Embeddings [B, embed_dim]
        """
        if timesteps.dim() == 2:
            timesteps = timesteps.squeeze(-1)
        
        # Clamp to valid range
        timesteps = torch.clamp(timesteps, 0, self.max_timesteps - 1)
        
        return self.pe[timesteps]


class TimeEmbedding(nn.Module):
    """
    Time embedding module: sinusoidal encoding + MLP projection.
    """
    
    def __init__(
        self,
        time_embed_dim: int,
        sinusoidal_dim: Optional[int] = None,
        activation: str = 'silu',
        max_timesteps: int = 10000
    ):
        super().__init__()
        
        if sinusoidal_dim is None:
            sinusoidal_dim = time_embed_dim // 4
        
        self.sinusoidal_dim = sinusoidal_dim
        self.time_embed_dim = time_embed_dim
        
        # Sinusoidal positional embedding
        self.sinusoidal_embedding = SinusoidalPositionalEmbedding(
            sinusoidal_dim, max_timesteps
        )
        
        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(sinusoidal_dim, time_embed_dim),
            get_activation(activation),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Compute time embeddings.
        
        Args:
            timesteps: Timestep indices [B] or [B, 1]
            
        Returns:
            Time embeddings [B, time_embed_dim]
        """
        # Get sinusoidal embeddings
        sin_emb = self.sinusoidal_embedding(timesteps)
        
        # Project through MLP
        time_emb = self.mlp(sin_emb)
        
        return time_emb


def get_activation(name: str) -> nn.Module:
    """Get activation function by name."""
    if name == 'silu' or name == 'swish':
        return nn.SiLU()
    elif name == 'relu':
        return nn.ReLU()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'leaky_relu':
        return nn.LeakyReLU(0.2)
    elif name == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Unknown activation: {name}")


class ConditionalTimeEmbedding(nn.Module):
    """
    Time embedding with optional class conditioning.
    """
    
    def __init__(
        self,
        time_embed_dim: int,
        num_classes: Optional[int] = None,
        class_embed_dim: Optional[int] = None,
        sinusoidal_dim: Optional[int] = None,
        activation: str = 'silu',
        max_timesteps: int = 10000
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Time embedding
        self.time_embedding = TimeEmbedding(
            time_embed_dim, sinusoidal_dim, activation, max_timesteps
        )
        
        # Class embedding (optional)
        if num_classes is not None:
            if class_embed_dim is None:
                class_embed_dim = time_embed_dim
            
            self.class_embedding = nn.Embedding(num_classes, class_embed_dim)
            
            # Combine time and class embeddings
            if class_embed_dim != time_embed_dim:
                self.class_proj = nn.Linear(class_embed_dim, time_embed_dim)
            else:
                self.class_proj = nn.Identity()
        else:
            self.class_embedding = None
            self.class_proj = None
    
    def forward(
        self, 
        timesteps: torch.Tensor, 
        class_labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute conditional time embeddings.
        
        Args:
            timesteps: Timestep indices [B]
            class_labels: Class labels [B] (optional)
            
        Returns:
            Combined embeddings [B, time_embed_dim]
        """
        # Get time embeddings
        time_emb = self.time_embedding(timesteps)
        
        # Add class embeddings if provided
        if self.class_embedding is not None and class_labels is not None:
            class_emb = self.class_embedding(class_labels)
            class_emb = self.class_proj(class_emb)
            time_emb = time_emb + class_emb
        
        return time_emb


def test_time_embedding():
    """Test time embedding modules."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 8
    time_embed_dim = 256
    num_classes = 10
    
    # Test basic time embedding
    time_emb = TimeEmbedding(time_embed_dim).to(device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    
    embeddings = time_emb(timesteps)
    print(f"Time embedding shape: {embeddings.shape}")
    print(f"Time embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
    
    # Test conditional time embedding
    cond_time_emb = ConditionalTimeEmbedding(
        time_embed_dim, num_classes=num_classes
    ).to(device)
    
    class_labels = torch.randint(0, num_classes, (batch_size,), device=device)
    cond_embeddings = cond_time_emb(timesteps, class_labels)
    
    print(f"Conditional embedding shape: {cond_embeddings.shape}")
    print(f"Conditional embedding range: [{cond_embeddings.min():.3f}, {cond_embeddings.max():.3f}]")
    
    # Test that embeddings are different for different timesteps
    t1 = torch.zeros(1, device=device, dtype=torch.long)
    t2 = torch.ones(1, device=device, dtype=torch.long) * 500
    
    emb1 = time_emb(t1)
    emb2 = time_emb(t2)
    
    diff = torch.norm(emb1 - emb2).item()
    print(f"Embedding difference (t=0 vs t=500): {diff:.3f}")
    
    print("Time embedding tests completed!")


if __name__ == "__main__":
    test_time_embedding()
