"""
U-Net Denoising Autoencoder with skip connections
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: Conv -> Norm -> ReLU -> Conv -> Norm -> ReLU."""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        mid_ch: int = None,
        norm_type: str = "group",
        dropout: float = 0.0
    ):
        super().__init__()
        
        if mid_ch is None:
            mid_ch = out_ch
        
        layers = []
        
        # First conv
        layers.append(nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=(norm_type == "none")))
        
        if norm_type == "batch":
            layers.append(nn.BatchNorm2d(mid_ch))
        elif norm_type == "instance":
            layers.append(nn.InstanceNorm2d(mid_ch))
        elif norm_type == "group":
            layers.append(nn.GroupNorm(min(32, mid_ch), mid_ch))
        
        layers.append(nn.ReLU(inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        # Second conv
        layers.append(nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=(norm_type == "none")))
        
        if norm_type == "batch":
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm_type == "instance":
            layers.append(nn.InstanceNorm2d(out_ch))
        elif norm_type == "group":
            layers.append(nn.GroupNorm(min(32, out_ch), out_ch))
        
        layers.append(nn.ReLU(inplace=True))
        
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        norm_type: str = "group",
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, norm_type=norm_type, dropout=dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        bilinear: bool = True,
        norm_type: str = "group",
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Use bilinear upsampling or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After upsampling, we'll concatenate with skip connection, so input will be in_ch + skip_ch
            # For this to work, we need to account for the concatenation
            self.conv = DoubleConv(in_ch, out_ch, norm_type=norm_type, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            # After transpose conv + skip concat: (in_ch // 2) + skip_ch = in_ch total
            self.conv = DoubleConv(in_ch, out_ch, norm_type=norm_type, dropout=dropout)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        
        # Handle input size differences
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        
        return self.conv(x)


class OutConv(nn.Module):
    """Output convolution layer."""
    
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNetDAE(nn.Module):
    """
    U-Net architecture for denoising autoencoder.
    Features skip connections for better detail preservation.
    """
    
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base_ch: int = 64,
        num_downs: int = 4,
        bilinear: bool = True,
        norm_type: str = "group",
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        self.num_downs = num_downs
        self.bilinear = bilinear
        
        # Calculate channel multipliers
        factor = 2 if bilinear else 1
        
        # Input layer
        self.inc = DoubleConv(in_ch, base_ch, norm_type=norm_type, dropout=dropout)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        ch = base_ch
        
        for i in range(num_downs):
            if i == num_downs - 1:
                # Last down block
                self.down_blocks.append(Down(ch, ch * 2 // factor, norm_type=norm_type, dropout=dropout))
                ch = ch * 2 // factor
            else:
                self.down_blocks.append(Down(ch, ch * 2, norm_type=norm_type, dropout=dropout))
                ch *= 2
        
        # Upsampling path - need to track channels carefully for skip connections
        self.up_blocks = nn.ModuleList()
        
        # Calculate channels for upsampling path
        up_channels = []
        temp_ch = base_ch
        for i in range(num_downs - 1):
            up_channels.append(temp_ch)
            temp_ch *= 2
        up_channels.reverse()  # Reverse to match upsampling order
        
        for i in range(num_downs):
            if i == 0:
                # First up block: from bottleneck (ch) + skip connection (ch//2) 
                skip_ch = ch // 2
                input_ch = ch + skip_ch  # Concatenated channels
                output_ch = skip_ch
                self.up_blocks.append(Up(input_ch, output_ch, bilinear, norm_type=norm_type, dropout=dropout))
                ch = output_ch
            else:
                # Subsequent up blocks
                skip_ch = up_channels[i-1] if i-1 < len(up_channels) else base_ch
                input_ch = ch + skip_ch  # Current + skip connection
                output_ch = skip_ch
                self.up_blocks.append(Up(input_ch, output_ch, bilinear, norm_type=norm_type, dropout=dropout))
                ch = output_ch
        
        # Output layer
        self.outc = OutConv(base_ch, out_ch)
        
        # Output activation
        self.output_activation = nn.Sigmoid()  # For [0, 1] output range
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store skip connections
        skip_connections = []
        
        # Input conv
        x = self.inc(x)
        skip_connections.append(x)
        
        # Downsampling path
        for down_block in self.down_blocks[:-1]:
            x = down_block(x)
            skip_connections.append(x)
        
        # Bottom of U-Net
        x = self.down_blocks[-1](x)
        
        # Upsampling path with skip connections
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_connections[-(i+1)]  # Reverse order
            x = up_block(x, skip)
        
        # Output layer
        x = self.outc(x)
        x = self.output_activation(x)
        
        return x
    
    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """Get intermediate feature maps for visualization."""
        features = {}
        
        # Input
        x = self.inc(x)
        features['input_conv'] = x.clone()
        skip_connections = [x]
        
        # Downsampling
        for i, down_block in enumerate(self.down_blocks[:-1]):
            x = down_block(x)
            features[f'down_{i}'] = x.clone()
            skip_connections.append(x)
        
        # Bottom
        x = self.down_blocks[-1](x)
        features['bottleneck'] = x.clone()
        
        # Upsampling
        for i, up_block in enumerate(self.up_blocks):
            skip = skip_connections[-(i+1)]
            x = up_block(x, skip)
            features[f'up_{i}'] = x.clone()
        
        # Output
        x = self.outc(x)
        features['output'] = x.clone()
        
        return features


class AttentionUNet(UNetDAE):
    """
    U-Net with attention gates for better feature selection.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Add attention gates (simplified implementation)
        # This is a placeholder for attention mechanism
        # Could be expanded with proper attention gates
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For now, just use standard U-Net forward
        # Can be extended with attention mechanisms
        return super().forward(x)