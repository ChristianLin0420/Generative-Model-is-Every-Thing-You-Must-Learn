"""Tiny UNet for DDPM denoising

A lightweight UNet architecture with time conditioning, designed for
educational purposes and fast training on small datasets.

Architecture:
- Encoder: Downsampling with residual blocks  
- Bottleneck: High-level feature processing
- Decoder: Upsampling with skip connections
- Time conditioning via AdaGroupNorm (FiLM)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
import math

from .time_embedding import TimeEmbedding, AdaGroupNorm, TimestepEmbedding


class ResidualBlock(TimestepEmbedding):
    """Residual block with time conditioning.
    
    Structure:
    - GroupNorm + SiLU + Conv
    - Time conditioning (FiLM)  
    - GroupNorm + SiLU + Conv
    - Skip connection
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_groups: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv block
        self.norm1 = AdaGroupNorm(num_groups, in_channels, time_embed_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Second conv block
        self.norm2 = AdaGroupNorm(num_groups, out_channels, time_embed_dim)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Skip connection projection if needed
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_connection = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            time_emb: Time embeddings [B, time_embed_dim]
        
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        residual = x
        
        # First block
        x = self.norm1(x, time_emb)
        x = F.silu(x)
        x = self.conv1(x)
        
        # Second block
        x = self.norm2(x, time_emb)
        x = F.silu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        
        # Skip connection
        return x + self.skip_connection(residual)


class AttentionBlock(nn.Module):
    """Simple self-attention block for UNet.
    
    Uses multi-head attention on flattened spatial dimensions.
    Only applied at lower resolutions to manage computational cost.
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        num_groups: int = 8
    ):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
        # Initialize projection to zero for stable training
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Output tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        q, k, v = qkv.unbind(1)  # Each [B, num_heads, head_dim, H*W]
        
        # Attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum("bhdn,bhdm->bhnm", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum("bhnm,bhdm->bhdn", attn, v)
        out = out.reshape(B, C, H, W)
        
        # Output projection
        out = self.proj_out(out)
        
        return out + residual


class DownBlock(TimestepEmbedding):
    """Downsampling block with residual blocks."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        num_res_blocks: int = 2,
        downsample: bool = True,
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_res_blocks = num_res_blocks
        self.downsample = downsample
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels if i == 0 else out_channels,
                out_channels,
                time_embed_dim,
                dropout=dropout
            ) for i in range(num_res_blocks)
        ])
        
        # Optional attention
        if use_attention:
            self.attention = AttentionBlock(out_channels, num_heads)
        else:
            self.attention = None
        
        # Downsampling
        if downsample:
            self.downsample_conv = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
        else:
            self.downsample_conv = None
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass.
        
        Args:
            x: Input tensor
            time_emb: Time embeddings
        
        Returns:
            Output tensor and skip connections
        """
        skip_connections = []
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x, time_emb)
            skip_connections.append(x)
        
        # Attention
        if self.attention is not None:
            x = self.attention(x)
            skip_connections[-1] = x  # Update last skip connection
        
        # Downsampling
        if self.downsample_conv is not None:
            x = self.downsample_conv(x)
        
        return x, skip_connections


class UpBlock(TimestepEmbedding):
    """Upsampling block with skip connections."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int,
        time_embed_dim: int,
        num_res_blocks: int = 2,
        upsample: bool = True,
        use_attention: bool = False,
        num_heads: int = 4,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.num_res_blocks = num_res_blocks
        self.upsample = upsample
        
        # Upsampling
        if upsample:
            self.upsample_conv = nn.ConvTranspose2d(in_channels, in_channels, 4, stride=2, padding=1)
        else:
            self.upsample_conv = None
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                in_channels + skip_channels if i == 0 else out_channels,
                out_channels,
                time_embed_dim,
                dropout=dropout
            ) for i in range(num_res_blocks)
        ])
        
        # Optional attention
        if use_attention:
            self.attention = AttentionBlock(out_channels, num_heads)
        else:
            self.attention = None
    
    def forward(self, x: torch.Tensor, skip_connections: List[torch.Tensor], time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            skip_connections: Skip connections from encoder
            time_emb: Time embeddings
        
        Returns:
            Output tensor
        """
        # Upsampling
        if self.upsample_conv is not None:
            x = self.upsample_conv(x)
        
        # Process skip connections in reverse order
        for i, res_block in enumerate(self.res_blocks):
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]  # Reverse order
                x = torch.cat([x, skip], dim=1)
            x = res_block(x, time_emb)
        
        # Attention
        if self.attention is not None:
            x = self.attention(x)
        
        return x


class UNetTiny(nn.Module):
    """Tiny UNet for DDPM denoising.
    
    A lightweight UNet with time conditioning, suitable for MNIST and CIFAR-10.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 64,
        num_res_blocks: int = 2,
        channel_mult: List[int] = [1, 2, 2],
        attention_resolutions: List[int] = [16],
        num_heads: int = 4,
        time_embed_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_scale_shift_norm: bool = True
    ):
        """Initialize UNet.
        
        Args:
            in_channels: Input channels (1 for MNIST, 3 for CIFAR-10)
            out_channels: Output channels (same as input for noise prediction)
            model_channels: Base number of channels
            num_res_blocks: Number of residual blocks per level
            channel_mult: Channel multipliers for each level
            attention_resolutions: Resolutions to apply attention at
            num_heads: Number of attention heads
            time_embed_dim: Time embedding dimension
            dropout: Dropout rate
            use_scale_shift_norm: Whether to use scale-shift normalization
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        
        # Time embedding dimension
        if time_embed_dim is None:
            time_embed_dim = model_channels * 4
        self.time_embed_dim = time_embed_dim
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Build encoder
        self.down_blocks = nn.ModuleList()
        ch = model_channels
        input_block_channels = [ch]
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                use_attention = (32 // (2 ** level)) in attention_resolutions
                
                self.down_blocks.append(DownBlock(
                    ch, out_ch, time_embed_dim, 1,
                    downsample=False, use_attention=use_attention,
                    num_heads=num_heads, dropout=dropout
                ))
                ch = out_ch
                input_block_channels.append(ch)
            
            # Downsample except for the last level
            if level != len(channel_mult) - 1:
                self.down_blocks.append(DownBlock(
                    ch, ch, time_embed_dim, 0,
                    downsample=True, use_attention=False,
                    dropout=dropout
                ))
                input_block_channels.append(ch)
        
        # Middle block
        self.middle_block = nn.Sequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout=dropout),
            AttentionBlock(ch, num_heads) if ch >= 32 else nn.Identity(),
            ResidualBlock(ch, ch, time_embed_dim, dropout=dropout),
        )
        
        # Build decoder
        self.up_blocks = nn.ModuleList()
        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):
                skip_ch = input_block_channels.pop()
                use_attention = (32 // (2 ** level)) in attention_resolutions
                
                self.up_blocks.append(UpBlock(
                    ch, out_ch, skip_ch, time_embed_dim, 1,
                    upsample=(i == num_res_blocks and level != 0),
                    use_attention=use_attention,
                    num_heads=num_heads, dropout=dropout
                ))
                ch = out_ch
        
        # Output projection
        self.output_norm = nn.GroupNorm(8, ch)
        self.output_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        # Initialize output layer to zero for stable training
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Noisy input images [B, C, H, W]
            timesteps: Timesteps [B] or [B, 1]
        
        Returns:
            Predicted noise [B, C, H, W]
        """
        # Time embedding
        time_emb = self.time_embedding(timesteps)
        
        # Input projection
        x = self.input_proj(x)
        
        # Encoder
        skip_connections = [x]
        for down_block in self.down_blocks:
            x, skips = down_block(x, time_emb)
            skip_connections.extend(skips)
        
        # Middle
        if isinstance(self.middle_block, nn.Sequential):
            for layer in self.middle_block:
                if isinstance(layer, TimestepEmbedding):
                    x = layer(x, time_emb)
                else:
                    x = layer(x)
        else:
            x = self.middle_block(x, time_emb)
        
        # Decoder
        for up_block in self.up_blocks:
            x = up_block(x, skip_connections, time_emb)
        
        # Output
        x = self.output_norm(x)
        x = F.silu(x)
        x = self.output_conv(x)
        
        return x


def test_unet():
    """Test UNet functionality."""
    # Test MNIST setup
    model = UNetTiny(
        in_channels=1, out_channels=1,
        model_channels=64, channel_mult=[1, 2, 2],
        attention_resolutions=[16]
    )
    
    x = torch.randn(2, 1, 32, 32)
    t = torch.randint(0, 100, (2,))
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"MNIST - Input shape: {x.shape}, Output shape: {out.shape}")
    
    # Test CIFAR-10 setup
    model = UNetTiny(
        in_channels=3, out_channels=3,
        model_channels=128, channel_mult=[1, 2, 2, 2],
        attention_resolutions=[16]
    )
    
    x = torch.randn(2, 3, 32, 32)
    t = torch.randint(0, 200, (2,))
    
    with torch.no_grad():
        out = model(x, t)
    
    print(f"CIFAR-10 - Input shape: {x.shape}, Output shape: {out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    print("UNet tests passed!")


if __name__ == "__main__":
    test_unet()