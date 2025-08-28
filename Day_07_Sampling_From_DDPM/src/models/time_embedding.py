"""
Time embedding for DDPM.
Must match Day 6 implementation exactly for checkpoint compatibility.

Implements sinusoidal positional encoding for timestep information.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding for DDPM timesteps.
    
    Based on the Transformer positional encoding but adapted for timesteps.
    Creates embeddings that allow the model to understand the current
    noise level / denoising timestep.
    """
    
    def __init__(self, embed_dim: int, max_period: int = 10000):
        """
        Initialize time embedding.
        
        Args:
            embed_dim: Dimension of the embedding
            max_period: Maximum period for sinusoidal encoding
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        # Precompute frequency components
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32) / half_dim
        )
        self.register_buffer('freqs', freqs)
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal embeddings for timesteps.
        
        Args:
            timesteps: Tensor of timesteps, shape (batch_size,)
            
        Returns:
            Time embeddings, shape (batch_size, embed_dim)
        """
        # Ensure timesteps are float
        timesteps = timesteps.float()
        
        # Compute arguments for sin/cos: timesteps * freqs
        args = timesteps[:, None] * self.freqs[None, :]
        
        # Apply sin and cos
        embeddings = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        # Handle odd embed_dim by padding with zeros
        if self.embed_dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        
        return embeddings


class TimeEmbeddingMLP(nn.Module):
    """
    MLP to process time embeddings.
    
    Takes sinusoidal embeddings and processes them through a small MLP
    to create richer representations for the UNet.
    """
    
    def __init__(self, time_embed_dim: int, hidden_dim: Optional[int] = None):
        """
        Initialize time embedding MLP.
        
        Args:
            time_embed_dim: Input and output dimension
            hidden_dim: Hidden layer dimension (default: 4 * time_embed_dim)
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = time_embed_dim * 4
        
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),  # Swish activation
            nn.Linear(hidden_dim, time_embed_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process time embeddings through MLP.
        
        Args:
            x: Time embeddings, shape (batch_size, time_embed_dim)
            
        Returns:
            Processed embeddings, shape (batch_size, time_embed_dim)
        """
        return self.mlp(x)


class TimestepEmbedder(nn.Module):
    """
    Complete timestep embedder combining sinusoidal encoding and MLP.
    
    This is the main component used in the UNet to convert integer timesteps
    into rich embedding vectors.
    """
    
    def __init__(self, time_embed_dim: int, max_period: int = 10000):
        """
        Initialize complete timestep embedder.
        
        Args:
            time_embed_dim: Dimension of time embeddings
            max_period: Maximum period for sinusoidal encoding
        """
        super().__init__()
        
        self.time_embed_dim = time_embed_dim
        
        # Sinusoidal embedding
        self.sinusoidal = SinusoidalTimeEmbedding(time_embed_dim, max_period)
        
        # MLP to process embeddings
        self.mlp = TimeEmbeddingMLP(time_embed_dim)
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Convert timesteps to embeddings.
        
        Args:
            timesteps: Integer timesteps, shape (batch_size,)
            
        Returns:
            Time embeddings, shape (batch_size, time_embed_dim)
        """
        # Create sinusoidal embeddings
        emb = self.sinusoidal(timesteps)
        
        # Process through MLP
        emb = self.mlp(emb)
        
        return emb


# Legacy alias for backward compatibility
TimeEmbedding = TimestepEmbedder


def get_timestep_embedding(timesteps: torch.Tensor, embed_dim: int, 
                          max_period: int = 10000) -> torch.Tensor:
    """
    Standalone function to get timestep embeddings.
    
    Args:
        timesteps: Timesteps, shape (batch_size,)
        embed_dim: Embedding dimension
        max_period: Maximum period for encoding
        
    Returns:
        Embeddings, shape (batch_size, embed_dim)
    """
    embedder = SinusoidalTimeEmbedding(embed_dim, max_period)
    embedder = embedder.to(timesteps.device)
    return embedder(timesteps)


# Test functions
def test_sinusoidal_embedding():
    """Test sinusoidal time embedding."""
    embed_dim = 128
    batch_size = 4
    
    embedder = SinusoidalTimeEmbedding(embed_dim)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    embeddings = embedder(timesteps)
    
    assert embeddings.shape == (batch_size, embed_dim)
    assert not torch.isnan(embeddings).any()
    assert not torch.isinf(embeddings).any()
    
    print("✓ Sinusoidal embedding test passed")


def test_timestep_embedder():
    """Test complete timestep embedder."""
    embed_dim = 256
    batch_size = 8
    
    embedder = TimestepEmbedder(embed_dim)
    timesteps = torch.randint(0, 1000, (batch_size,))
    
    embeddings = embedder(timesteps)
    
    assert embeddings.shape == (batch_size, embed_dim)
    assert not torch.isnan(embeddings).any()
    assert not torch.isinf(embeddings).any()
    
    # Test that different timesteps give different embeddings
    emb1 = embedder(torch.tensor([0]))
    emb2 = embedder(torch.tensor([100]))
    assert not torch.allclose(emb1, emb2)
    
    print("✓ Timestep embedder test passed")


def test_embedding_properties():
    """Test mathematical properties of embeddings."""
    embedder = TimestepEmbedder(128)
    
    # Test that same timesteps give same embeddings
    t = torch.tensor([42, 42, 100, 100])
    emb = embedder(t)
    assert torch.allclose(emb[0], emb[1])
    assert torch.allclose(emb[2], emb[3])
    assert not torch.allclose(emb[0], emb[2])
    
    # Test embedding smoothness (nearby timesteps should be similar)
    t1 = torch.tensor([50.0])
    t2 = torch.tensor([51.0])
    emb1 = embedder(t1)
    emb2 = embedder(t2)
    
    # Should be similar but not identical
    similarity = torch.cosine_similarity(emb1, emb2, dim=1)
    assert similarity > 0.9  # Very similar
    assert not torch.allclose(emb1, emb2)  # But not identical
    
    print("✓ Embedding properties test passed")


if __name__ == "__main__":
    test_sinusoidal_embedding()
    test_timestep_embedder()
    test_embedding_properties()
    print("✓ All time embedding tests passed!")
