"""
Compact UNet with time-FiLM conditioning for DDPM
Includes optional attention on 16x16 feature maps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

from .time_embedding import TimeEmbedding, TimestepBlock, TimestepEmbedSequential


class GroupNorm32(nn.GroupNorm):
    """GroupNorm that works with fp16"""
    
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class Attention(nn.Module):
    """Multi-head self-attention for feature maps"""
    
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0
        
        self.norm = GroupNorm32(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for multi-head attention
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)
        
        # Attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        h = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        h = h.reshape(B, C, H, W)
        
        h = self.proj_out(h)
        return x + h


class ResidualBlock(TimestepBlock):
    """
    Residual block with time-FiLM conditioning
    Uses GroupNorm + SiLU activation
    """
    
    def __init__(
        self, 
        in_channels: int,
        out_channels: Optional[int] = None,
        time_emb_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_conv_shortcut: bool = False
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.use_conv_shortcut = use_conv_shortcut
        
        # First conv block
        self.norm1 = GroupNorm32(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
        
        # Time embedding projection (for FiLM)
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Linear(time_emb_dim, self.out_channels * 2)  # scale + shift
        else:
            self.time_emb_proj = None
            
        # Second conv block
        self.norm2 = GroupNorm32(32, self.out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        
        # Shortcut connection
        if in_channels != self.out_channels:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_channels, self.out_channels, 3, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_channels, self.out_channels, 1)
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x
        
        # First conv
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Time conditioning (FiLM)
        if emb is not None and self.time_emb_proj is not None:
            emb_out = self.time_emb_proj(F.silu(emb))[:, :, None, None]
            scale, shift = emb_out.chunk(2, dim=1)
            h = h * (1 + scale) + shift
            
        # Second conv
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class DownsampleBlock(nn.Module):
    """Downsampling block with optional convolution"""
    
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        else:
            self.conv = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_conv:
            return self.conv(x)
        else:
            return F.avg_pool2d(x, kernel_size=2, stride=2)


class UpsampleBlock(nn.Module):
    """Upsampling block with optional convolution"""
    
    def __init__(self, channels: int, use_conv: bool = True):
        super().__init__()
        self.use_conv = use_conv
        
        if use_conv:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        else:
            self.conv = None
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class UNetSmall(nn.Module):
    """
    Compact UNet for DDPM with time conditioning
    
    Architecture:
    - Encoder: [32, 64, 128, 256] channels with 2 residual blocks each
    - Decoder: [256, 128, 64, 32] channels with 2 residual blocks each
    - Optional attention at 16x16 resolution (128 channels)
    - Skip connections between encoder/decoder
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        model_channels: int = 32,
        channel_mult: List[int] = [1, 2, 4, 8],
        num_res_blocks: int = 2,
        attention_resolutions: List[int] = [16],
        dropout: float = 0.0,
        time_embed_dim: int = 128,
        use_attention: bool = True,
        num_heads: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # Input projection
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        # Build encoder
        channels = [model_channels]
        ch = model_channels
        self.encoder_blocks = nn.ModuleList()
        
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResidualBlock(
                        ch, out_ch, time_embed_dim, dropout
                    )
                )
                ch = out_ch
                channels.append(ch)
                
                # Add attention if at correct resolution
                if use_attention and any(ch == model_channels * mult for mult in channel_mult 
                                       if model_channels * mult in [model_channels * 4]):  # 128 channels
                    self.encoder_blocks.append(Attention(ch, num_heads))
                    
            # Downsample (except last level)
            if level != len(channel_mult) - 1:
                self.encoder_blocks.append(DownsampleBlock(ch))
                channels.append(ch)
                
        # Middle blocks
        self.middle_blocks = TimestepEmbedSequential(
            ResidualBlock(ch, ch, time_embed_dim, dropout),
            Attention(ch, num_heads) if use_attention else nn.Identity(),
            ResidualBlock(ch, ch, time_embed_dim, dropout)
        )
        
        # Build decoder
        self.decoder_blocks = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            
            for i in range(num_res_blocks + 1):  # +1 for skip connection
                if len(channels) > 0:
                    skip_ch = channels.pop()
                    self.decoder_blocks.append(
                        ResidualBlock(
                            ch + skip_ch, out_ch, time_embed_dim, dropout
                        )
                    )
                else:
                    # No skip connection available, just use current channels
                    self.decoder_blocks.append(
                        ResidualBlock(
                            ch, out_ch, time_embed_dim, dropout
                        )
                    )
                ch = out_ch
                
                # Add attention if at correct resolution
                if use_attention and any(ch == model_channels * mult for mult in channel_mult 
                                       if model_channels * mult in [model_channels * 4]):  # 128 channels
                    self.decoder_blocks.append(Attention(ch, num_heads))
                    
            # Upsample (except first level)
            if level != 0:
                self.decoder_blocks.append(UpsampleBlock(ch))
                
        # Output projection
        self.out_norm = GroupNorm32(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights with reasonable defaults"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
        # Zero-initialize final conv for stable training
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)
        
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input tensor [B, C, H, W]
            timesteps: timestep indices [B]
        Returns:
            output tensor [B, out_channels, H, W]
        """
        # Get time embeddings
        emb = self.time_embed(timesteps)
        
        # Input projection
        h = self.input_conv(x)
        
        # Encoder with skip connections
        skip_connections = [h]
        
        for block in self.encoder_blocks:
            if isinstance(block, (ResidualBlock, TimestepEmbedSequential)):
                h = block(h, emb)
            else:  # Downsample, Attention
                h = block(h)
            skip_connections.append(h)
            
        # Middle blocks
        h = self.middle_blocks(h, emb)
        
        # Decoder with skip connections
        for block in self.decoder_blocks:
            if isinstance(block, ResidualBlock):
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, emb)
            else:  # Upsample, Attention
                h = block(h)
                
        # Output projection
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)
        
        return h
        
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)