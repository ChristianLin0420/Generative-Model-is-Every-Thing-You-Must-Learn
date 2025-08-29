"""
Small UNet model for DDPM, identical to Day 6/7 for apples-to-apples comparison.
Based on the UNet architecture from "Denoising Diffusion Probabilistic Models".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import math

from .time_embedding import TimeEmbedding


class ResidualBlock(nn.Module):
    """
    Residual block with group normalization and time conditioning.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        num_groups: int = 32
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv block
        # Ensure num_groups divides in_channels evenly
        groups_for_in = min(num_groups, in_channels)
        while in_channels % groups_for_in != 0 and groups_for_in > 1:
            groups_for_in -= 1
        self.norm1 = nn.GroupNorm(
            num_groups=groups_for_in,
            num_channels=in_channels
        )
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        # Second conv block  
        # Ensure num_groups divides out_channels evenly
        groups_for_out = min(num_groups, out_channels)
        while out_channels % groups_for_out != 0 and groups_for_out > 1:
            groups_for_out -= 1
        self.norm2 = nn.GroupNorm(
            num_groups=groups_for_out,
            num_channels=out_channels
        )
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with time conditioning.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            time_emb: Time embedding [B, time_embed_dim]
            
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        h = x
        
        # First conv block
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv1(h)
        
        # Add time embedding
        time_proj = self.time_proj(time_emb)[:, :, None, None]  # [B, out_channels, 1, 1]
        h = h + time_proj
        
        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """
    Self-attention block for UNet.
    """
    
    def __init__(self, channels: int, num_groups: int = 32):
        super().__init__()
        
        self.channels = channels
        # Ensure num_groups divides channels evenly
        self.num_groups = min(num_groups, channels)
        while channels % self.num_groups != 0 and self.num_groups > 1:
            self.num_groups -= 1
        
        self.norm = nn.GroupNorm(self.num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for self-attention.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Output tensor [B, C, H, W]
        """
        b, c, h, w = x.shape
        
        # Normalize input
        h_norm = self.norm(x)
        
        # Compute Q, K, V
        qkv = self.qkv(h_norm)  # [B, 3*C, H, W]
        q, k, v = qkv.chunk(3, dim=1)  # Each [B, C, H, W]
        
        # Reshape for attention computation
        q = q.view(b, c, h * w).transpose(1, 2)  # [B, H*W, C]
        k = k.view(b, c, h * w).transpose(1, 2)  # [B, H*W, C]
        v = v.view(b, c, h * w).transpose(1, 2)  # [B, H*W, C]
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(c)
        attn = torch.bmm(q, k.transpose(1, 2)) * scale  # [B, H*W, H*W]
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(attn, v)  # [B, H*W, C]
        out = out.transpose(1, 2).view(b, c, h, w)  # [B, C, H, W]
        
        # Project and add residual
        out = self.proj_out(out)
        return x + out


class Downsample(nn.Module):
    """Downsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class UNetSmall(nn.Module):
    """
    Small UNet for DDPM with time conditioning.
    Architecture designed for 28x28 (MNIST) and 32x32 (CIFAR) images.
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: Optional[int] = None,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 2],
        num_res_blocks: int = 2,
        time_embed_dim: int = 256,
        dropout: float = 0.1,
        attention_resolutions: List[int] = [16],
        num_groups: int = 32
    ):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_res_blocks = num_res_blocks
        
        # Time embedding
        self.time_embedding = TimeEmbedding(time_embed_dim)
        
        # Calculate channel dimensions
        channels = [base_channels * mult for mult in channel_multipliers]
        self.num_levels = len(channels)
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        
        in_ch = base_channels
        for level, out_ch in enumerate(channels):
            # Residual blocks
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResidualBlock(
                    in_ch, out_ch, time_embed_dim, dropout, num_groups
                ))
                in_ch = out_ch
            
            # Attention block if specified
            if self._get_resolution_at_level(level) in attention_resolutions:
                blocks.append(AttentionBlock(out_ch, num_groups))
            
            self.down_blocks.append(blocks)
            
            # Downsampling (except for last level)
            if level < len(channels) - 1:
                self.down_samples.append(Downsample(out_ch))
            else:
                self.down_samples.append(nn.Identity())
        
        # Middle blocks
        mid_ch = channels[-1]
        self.mid_block1 = ResidualBlock(
            mid_ch, mid_ch, time_embed_dim, dropout, num_groups
        )
        self.mid_attn = AttentionBlock(mid_ch, num_groups)
        self.mid_block2 = ResidualBlock(
            mid_ch, mid_ch, time_embed_dim, dropout, num_groups
        )
        
        # Upsampling path
        self.up_samples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        
        for level in reversed(range(len(channels))):
            out_ch = channels[level]
            in_ch = channels[level] if level == len(channels) - 1 else channels[level + 1]
            
            # Upsampling (except for last level)
            if level < len(channels) - 1:
                self.up_samples.append(Upsample(in_ch))
            else:
                self.up_samples.append(nn.Identity())
            
            # Residual blocks with skip connections
            blocks = nn.ModuleList()
            for i in range(num_res_blocks + 1):  # +1 for skip connection block
                skip_ch = out_ch if i == 0 else 0  # Skip connection from encoder
                blocks.append(ResidualBlock(
                    in_ch + skip_ch, out_ch, time_embed_dim, dropout, num_groups
                ))
                in_ch = out_ch
            
            # Attention block if specified
            if self._get_resolution_at_level(level) in attention_resolutions:
                blocks.append(AttentionBlock(out_ch, num_groups))
            
            self.up_blocks.append(blocks)
        
        # Output convolution
        # Ensure num_groups divides base_channels evenly
        groups_for_base = min(num_groups, base_channels)
        while base_channels % groups_for_base != 0 and groups_for_base > 1:
            groups_for_base -= 1
        self.norm_out = nn.GroupNorm(groups_for_base, base_channels)
        self.conv_out = nn.Conv2d(base_channels, out_channels, 3, padding=1)
    
    def _get_resolution_at_level(self, level: int) -> int:
        """Get spatial resolution at a given level (assuming 32x32 input)."""
        return 32 // (2 ** level)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, in_channels, H, W]
            timesteps: Timestep indices [B]
            
        Returns:
            Output tensor [B, out_channels, H, W]
        """
        # Time embedding
        time_emb = self.time_embedding(timesteps)
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Store skip connections
        skip_connections = []
        
        # Downsampling path
        for level, (blocks, downsample) in enumerate(zip(self.down_blocks, self.down_samples)):
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, time_emb)
                else:  # AttentionBlock
                    h = block(h)
            
            skip_connections.append(h)
            h = downsample(h)
        
        # Middle blocks
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        # Upsampling path
        for level, (upsample, blocks) in enumerate(zip(self.up_samples, self.up_blocks)):
            h = upsample(h)
            
            # Add skip connection
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            
            for block in blocks:
                if isinstance(block, ResidualBlock):
                    h = block(h, time_emb)
                else:  # AttentionBlock
                    h = block(h)
        
        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        
        return h


def test_unet():
    """Test UNet model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test with MNIST-like input
    batch_size = 2
    in_channels = 1
    height = 28
    width = 28
    
    model = UNetSmall(
        in_channels=in_channels,
        base_channels=64,
        channel_multipliers=[1, 2, 2],
        num_res_blocks=2,
        time_embed_dim=256,
        dropout=0.1,
        attention_resolutions=[14]
    ).to(device)
    
    # Test forward pass
    x = torch.randn(batch_size, in_channels, height, width, device=device)
    timesteps = torch.randint(0, 1000, (batch_size,), device=device)
    
    with torch.no_grad():
        output = model(x, timesteps)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2:.2f} MB")
    
    # Test with different timesteps
    t1 = torch.zeros(batch_size, device=device, dtype=torch.long)
    t2 = torch.ones(batch_size, device=device, dtype=torch.long) * 500
    
    with torch.no_grad():
        out1 = model(x, t1)
        out2 = model(x, t2)
    
    diff = torch.norm(out1 - out2).item()
    print(f"Output difference (t=0 vs t=500): {diff:.3f}")
    
    print("UNet test completed!")


if __name__ == "__main__":
    test_unet()
