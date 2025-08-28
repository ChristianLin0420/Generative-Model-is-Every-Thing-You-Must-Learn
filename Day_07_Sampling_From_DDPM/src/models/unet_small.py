"""
Small UNet architecture for DDPM.
Must match Day 6 implementation exactly for checkpoint compatibility.

Based on the UNet architecture from the original DDPM paper
with time conditioning and attention layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math

from .time_embedding import TimestepEmbedder


class Swish(nn.Module):
    """Swish activation function (also known as SiLU)."""
    
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """GroupNorm with 32 channels per group (or fewer if needed)."""
    
    def __init__(self, num_channels: int):
        super().__init__(min(32, num_channels), num_channels)


class ConvBlock(nn.Module):
    """
    Convolutional block with GroupNorm and Swish activation.
    Used as the basic building block in ResNet-style layers.
    """
    
    def __init__(self, in_ch: int, out_ch: int, time_embed_dim: Optional[int] = None,
                 kernel_size: int = 3, padding: int = 1):
        super().__init__()
        
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.norm = GroupNorm32(out_ch)
        self.act = Swish()
        
        # Time embedding projection
        if time_embed_dim is not None:
            self.time_proj = nn.Linear(time_embed_dim, out_ch)
        else:
            self.time_proj = None
    
    def forward(self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        
        # Add time embedding
        if time_emb is not None and self.time_proj is not None:
            time_emb = self.time_proj(time_emb)
            h = h + time_emb[:, :, None, None]
        
        h = self.act(h)
        return h


class ResBlock(nn.Module):
    """
    Residual block with time embedding and optional up/downsampling.
    """
    
    def __init__(self, in_ch: int, out_ch: int, time_embed_dim: int,
                 dropout: float = 0.0, up: bool = False, down: bool = False):
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.up = up
        self.down = down
        
        # First convolution block
        self.conv1 = ConvBlock(in_ch, out_ch, time_embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Second convolution block
        self.conv2 = ConvBlock(out_ch, out_ch)
        
        # Skip connection
        if in_ch != out_ch:
            self.skip_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip_conv = nn.Identity()
        
        # Sampling layers
        if up:
            self.up_layer = nn.Upsample(scale_factor=2, mode='nearest')
        elif down:
            self.down_layer = nn.AvgPool2d(2)
        else:
            self.up_layer = self.down_layer = nn.Identity()
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        # Apply sampling to input
        if self.up:
            x = self.up_layer(x)
        elif self.down:
            x = self.down_layer(x)
        
        # Main path
        h = self.conv1(x, time_emb)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Skip connection
        skip = self.skip_conv(x)
        
        return h + skip


class AttentionBlock(nn.Module):
    """
    Self-attention block for spatial features.
    Applied at lower resolutions to capture global dependencies.
    """
    
    def __init__(self, channels: int, num_heads: int = 1):
        super().__init__()
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        self.norm = GroupNorm32(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        
        # Normalize input
        h_norm = self.norm(x)
        
        # Get Q, K, V
        qkv = self.qkv(h_norm)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention computation
        q = q.view(b, self.num_heads, self.head_dim, h * w)
        k = k.view(b, self.num_heads, self.head_dim, h * w)
        v = v.view(b, self.num_heads, self.head_dim, h * w)
        
        # Compute attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.einsum('bhdt,bhds->bhts', q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        h_attn = torch.einsum('bhts,bhds->bhdt', attn, v)
        h_attn = h_attn.view(b, c, h, w)
        
        # Project back
        h_attn = self.proj_out(h_attn)
        
        # Residual connection
        return x + h_attn


class UNetSmall(nn.Module):
    """
    Small UNet for DDPM with time conditioning.
    
    Architecture:
    - Encoder: downsample with ResBlocks
    - Middle: ResBlocks + Attention
    - Decoder: upsample with skip connections
    """
    
    def __init__(self, in_ch: int = 3, base_ch: int = 64, 
                 ch_mult: List[int] = [1, 2, 2], time_embed_dim: int = 256,
                 dropout: float = 0.0, use_attention: bool = True):
        """
        Initialize UNet.
        
        Args:
            in_ch: Input channels (1 for MNIST, 3 for CIFAR-10)
            base_ch: Base number of channels
            ch_mult: Channel multipliers for each resolution level
            time_embed_dim: Time embedding dimension
            dropout: Dropout probability
            use_attention: Whether to use attention at lowest resolution
        """
        super().__init__()
        
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.ch_mult = ch_mult
        self.time_embed_dim = time_embed_dim
        self.num_resolutions = len(ch_mult)
        
        # Time embedding
        self.time_embed = TimestepEmbedder(time_embed_dim)
        
        # Input convolution
        self.input_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
        
        # Calculate channel sizes for each level
        ch_sizes = [base_ch * mult for mult in ch_mult]
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        
        ch_in = base_ch
        for i, ch_out in enumerate(ch_sizes):
            # ResBlocks at this resolution
            self.encoder_blocks.append(nn.ModuleList([
                ResBlock(ch_in, ch_out, time_embed_dim, dropout),
                ResBlock(ch_out, ch_out, time_embed_dim, dropout)
            ]))
            
            # Downsampling (except for last level)
            if i < len(ch_sizes) - 1:
                self.encoder_downs.append(nn.AvgPool2d(2))
            else:
                self.encoder_downs.append(nn.Identity())
            
            ch_in = ch_out
        
        # Middle blocks
        mid_ch = ch_sizes[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_embed_dim, dropout)
        self.mid_attn = AttentionBlock(mid_ch) if use_attention else nn.Identity()
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_embed_dim, dropout)
        
        # Decoder
        self.decoder_ups = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        ch_in = mid_ch
        for i, ch_out in enumerate(reversed(ch_sizes)):
            # Upsampling (except for first level)
            if i > 0:
                self.decoder_ups.append(nn.Upsample(scale_factor=2, mode='nearest'))
            else:
                self.decoder_ups.append(nn.Identity())
            
            # Input channels: current + skip connection from encoder
            skip_ch = ch_sizes[-(i+1)]  # Corresponding encoder channel size
            if i == 0:
                # First decoder level: middle + skip
                decoder_in_ch = mid_ch + skip_ch
            else:
                # Other levels: previous decoder output + skip
                decoder_in_ch = ch_in + skip_ch
            
            # ResBlocks at this resolution
            self.decoder_blocks.append(nn.ModuleList([
                ResBlock(decoder_in_ch, ch_out, time_embed_dim, dropout),
                ResBlock(ch_out, ch_out, time_embed_dim, dropout)
            ]))
            
            ch_in = ch_out
            
        # Ensure we have the right number of up layers and decoder blocks
        assert len(self.decoder_ups) == len(ch_sizes)
        assert len(self.decoder_blocks) == len(ch_sizes)
        
        # Output convolution
        self.output_conv = nn.Sequential(
            GroupNorm32(base_ch),
            Swish(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through UNet.
        
        Args:
            x: Input images, shape (B, C, H, W)
            timesteps: Timesteps, shape (B,)
            
        Returns:
            Predicted noise, shape (B, C, H, W)
        """
        # Get time embeddings
        time_emb = self.time_embed(timesteps)
        
        # Input convolution
        h = self.input_conv(x)
        
        # Encoder
        skip_connections = []
        
        for blocks, down in zip(self.encoder_blocks, self.encoder_downs):
            # Apply ResBlocks
            for block in blocks:
                h = block(h, time_emb)
            
            # Store skip connection before downsampling
            skip_connections.append(h)
            
            # Downsample
            h = down(h)
        
        # Middle
        h = self.mid_block1(h, time_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, time_emb)
        
        # Decoder
        for i, (up, blocks) in enumerate(zip(self.decoder_ups, self.decoder_blocks)):
            # Upsample
            h = up(h)
            
            # Add skip connection
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]  # Reverse order
                h = torch.cat([h, skip], dim=1)
            
            # Apply ResBlocks
            for block in blocks:
                h = block(h, time_emb)
        
        # Output
        h = self.output_conv(h)
        
        return h


def create_unet_small(config: dict) -> UNetSmall:
    """Create UNet from configuration."""
    model_config = config.get('model', {})
    
    # Check if this configuration matches Day 6 SimpleUNet architecture
    base_ch = model_config.get('base_ch', 64)
    ch_mult = model_config.get('ch_mult', [1, 2, 2])
    
    # Day 6 compatibility: use SimpleUNet for base_ch=32 and ch_mult=[1, 2, 4]
    if base_ch == 32 and ch_mult == [1, 2, 4]:
        from .simple_unet import SimpleUNet
        return SimpleUNet(
            in_channels=model_config.get('in_ch', 3),
            out_channels=model_config.get('in_ch', 3),  # Output same as input
            model_channels=base_ch,
            time_embed_dim=model_config.get('time_embed_dim', 256)
        )
    
    # Default Day 7 UNetSmall
    return UNetSmall(
        in_ch=model_config.get('in_ch', 3),
        base_ch=base_ch,
        ch_mult=ch_mult,
        time_embed_dim=model_config.get('time_embed_dim', 256),
        dropout=model_config.get('dropout', 0.0),
        use_attention=model_config.get('use_attention', True)
    )


# Test functions
def test_unet_shapes():
    """Test UNet input/output shapes."""
    # MNIST config
    model = UNetSmall(in_ch=1, base_ch=32, ch_mult=[1, 2], time_embed_dim=128)
    
    batch_size = 4
    x = torch.randn(batch_size, 1, 32, 32)
    t = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        out = model(x, t)
    
    assert out.shape == x.shape
    print("✓ UNet shape test passed")


def test_unet_forward():
    """Test UNet forward pass."""
    model = UNetSmall(in_ch=3, base_ch=64, ch_mult=[1, 2, 2], time_embed_dim=256)
    
    x = torch.randn(2, 3, 32, 32)
    t = torch.tensor([0, 500])
    
    out = model(x, t)
    
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
    
    print("✓ UNet forward test passed")


if __name__ == "__main__":
    test_unet_shapes()
    test_unet_forward()
    print("✓ All UNet tests passed!")
