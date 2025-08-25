"""
Simple UNet implementation that definitely works
Minimal complexity to get training started
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .time_embedding import TimeEmbedding


class SimpleResBlock(nn.Module):
    """Simple residual block with time conditioning"""
    
    def __init__(self, in_channels, out_channels, time_embed_dim):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_embed_dim, out_channels)
        
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
            
    def forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        
        # Add time conditioning
        time_emb = self.time_proj(F.silu(emb))[:, :, None, None]
        h = h + time_emb
        
        h = self.conv2(F.silu(self.norm2(h)))
        
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    """Very simple UNet that should work"""
    
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        model_channels=64,
        time_embed_dim=256
    ):
        super().__init__()
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_embed_dim)
        
        # Encoder
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)
        
        self.down1 = SimpleResBlock(model_channels, model_channels, time_embed_dim)
        self.down2 = SimpleResBlock(model_channels, model_channels*2, time_embed_dim)
        self.down3 = SimpleResBlock(model_channels*2, model_channels*4, time_embed_dim)
        
        # Middle
        self.middle = SimpleResBlock(model_channels*4, model_channels*4, time_embed_dim)
        
        # Decoder
        self.up3 = SimpleResBlock(model_channels*4 + model_channels*4, model_channels*2, time_embed_dim)
        self.up2 = SimpleResBlock(model_channels*2 + model_channels*2, model_channels, time_embed_dim)
        self.up1 = SimpleResBlock(model_channels + model_channels, model_channels, time_embed_dim)
        
        # Output
        self.output_conv = nn.Conv2d(model_channels, out_channels, 3, padding=1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x, timesteps):
        # Time embedding
        emb = self.time_embed(timesteps)
        
        # Encoder
        h1 = self.input_conv(x)
        h1 = self.down1(h1, emb)
        
        h2 = F.avg_pool2d(h1, 2)
        h2 = self.down2(h2, emb)
        
        h3 = F.avg_pool2d(h2, 2)
        h3 = self.down3(h3, emb)
        
        # Middle
        h = F.avg_pool2d(h3, 2)
        h = self.middle(h, emb)
        
        # Decoder with skip connections
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = torch.cat([h, h3], dim=1)
        h = self.up3(h, emb)
        
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = torch.cat([h, h2], dim=1)
        h = self.up2(h, emb)
        
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = torch.cat([h, h1], dim=1)
        h = self.up1(h, emb)
        
        # Output
        return self.output_conv(h)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)