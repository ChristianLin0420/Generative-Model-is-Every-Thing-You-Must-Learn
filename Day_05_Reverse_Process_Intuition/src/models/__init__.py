"""Model implementations for DDPM

Contains:
- time_embedding: Sinusoidal time embeddings
- unet_tiny: Lightweight UNet with time conditioning
"""

from .time_embedding import sinusoidal_time_embed, TimeEmbedding
from .unet_tiny import UNetTiny

__all__ = ["sinusoidal_time_embed", "TimeEmbedding", "UNetTiny"]