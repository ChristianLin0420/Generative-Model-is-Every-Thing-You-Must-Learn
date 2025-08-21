"""
Convolutional Denoising Autoencoder - Lightweight baseline
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Basic convolutional block with normalization and activation."""
    
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        norm_type: str = "batch",
        activation: str = "relu",
        dropout: float = 0.0
    ):
        super().__init__()
        
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=(norm_type == "none"))
        ]
        
        # Normalization
        if norm_type == "batch":
            layers.append(nn.BatchNorm2d(out_ch))
        elif norm_type == "instance":
            layers.append(nn.InstanceNorm2d(out_ch))
        elif norm_type == "group":
            layers.append(nn.GroupNorm(min(32, out_ch), out_ch))
        
        # Activation
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "gelu":
            layers.append(nn.GELU())
        
        # Dropout
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvDAE(nn.Module):
    """
    Convolutional Denoising Autoencoder with encoder-decoder structure.
    Lightweight baseline for comparison with UNet.
    """
    
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        base_ch: int = 64,
        num_downs: int = 3,
        norm_type: str = "batch",
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.base_ch = base_ch
        self.num_downs = num_downs
        
        # Encoder
        encoder_layers = []
        ch = base_ch
        
        # Input layer
        encoder_layers.append(
            ConvBlock(in_ch, ch, norm_type=norm_type, activation=activation, dropout=dropout)
        )
        
        # Downsampling layers
        for i in range(num_downs):
            encoder_layers.append(
                ConvBlock(ch, ch * 2, stride=2, norm_type=norm_type, activation=activation, dropout=dropout)
            )
            ch *= 2
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Bottleneck
        self.bottleneck = ConvBlock(ch, ch, norm_type=norm_type, activation=activation, dropout=dropout)
        
        # Decoder
        decoder_layers = []
        
        # Upsampling layers - use adaptive approach to preserve input size
        for i in range(num_downs):
            # Use stride=2 but adjust kernel and padding to preserve size
            decoder_layers.append(
                nn.ConvTranspose2d(ch, ch // 2, kernel_size=4, stride=2, padding=1, bias=(norm_type == "none"))
            )
            ch //= 2
            
            if norm_type == "batch":
                decoder_layers.append(nn.BatchNorm2d(ch))
            elif norm_type == "instance":
                decoder_layers.append(nn.InstanceNorm2d(ch))
            elif norm_type == "group":
                decoder_layers.append(nn.GroupNorm(min(32, ch), ch))
            
            decoder_layers.append(nn.ReLU(inplace=True) if activation == "relu" else nn.GELU())
            
            if dropout > 0:
                decoder_layers.append(nn.Dropout2d(dropout))
        
        # Output layer
        decoder_layers.append(
            nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)
        )
        decoder_layers.append(nn.Sigmoid())  # Output in [0, 1] range
        
        self.decoder = nn.Sequential(*decoder_layers)
        
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
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        z = self.bottleneck(z)
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass through autoencoder."""
        input_size = x.shape[-2:]  # Store original H, W
        
        # Encode
        z = self.encode(x)
        
        # Decode
        recon = self.decode(z)
        
        # Ensure output matches input size exactly
        if recon.shape[-2:] != input_size:
            recon = torch.nn.functional.interpolate(
                recon, size=input_size, mode='bilinear', align_corners=False
            )
        
        return recon
    
    def get_latent_dim(self, input_shape: tuple) -> tuple:
        """Compute latent representation dimensions."""
        with torch.no_grad():
            x = torch.randn(1, *input_shape)
            z = self.encode(x)
            return z.shape[1:]


class SimpleConvDAE(nn.Module):
    """
    Ultra-lightweight convolutional DAE for fast prototyping.
    """
    
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        hidden_ch: int = 32
    ):
        super().__init__()
        
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        # Simple encoder: conv -> pool -> conv -> pool
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_ch, hidden_ch * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        # Simple decoder: upsample -> conv -> upsample -> conv
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_ch * 2, hidden_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(hidden_ch, out_ch, 3, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon