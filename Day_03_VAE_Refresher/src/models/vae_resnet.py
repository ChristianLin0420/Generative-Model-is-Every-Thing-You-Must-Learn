"""
ResNet-based Variational Autoencoder implementation.
Uses residual blocks for deeper, more stable training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Basic residual block with batch normalization."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResidualTransposeBlock(nn.Module):
    """Residual block with transposed convolution for decoder."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, 
                                       stride=stride, padding=1, output_padding=stride-1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, 
                                  stride=stride, output_padding=stride-1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetEncoder(nn.Module):
    """ResNet-based encoder for VAE."""
    
    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, 
                              stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        
        # Residual layers
        self.layer1 = self._make_layer(base_channels, base_channels, 2, stride=1)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, 2, stride=2)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, 2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Linear layers for latent parameters
        self.fc_mu = nn.Linear(base_channels * 8, latent_dim)
        self.fc_logvar = nn.Linear(base_channels * 8, latent_dim)
    
    def _make_layer(self, in_channels: int, out_channels: int, 
                   num_blocks: int, stride: int) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input images [batch_size, channels, height, width]
        
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        # Initial convolution
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        # Global pooling and flatten
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        
        # Latent parameters
        mu = self.fc_mu(out)
        logvar = self.fc_logvar(out)
        
        return mu, logvar


class ResNetDecoder(nn.Module):
    """ResNet-based decoder for VAE."""
    
    def __init__(self, latent_dim: int, out_channels: int, base_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # Determine initial size based on output channels
        if out_channels == 1:  # MNIST
            self.init_size = 4
            self.img_size = 28
        else:  # CIFAR-10
            self.init_size = 4
            self.img_size = 32
        
        # Initial linear layer
        self.fc = nn.Linear(latent_dim, base_channels * 8 * self.init_size * self.init_size)
        
        # Residual transpose layers
        self.layer1 = self._make_transpose_layer(base_channels * 8, base_channels * 4, 2, stride=2)
        self.layer2 = self._make_transpose_layer(base_channels * 4, base_channels * 2, 2, stride=2)
        self.layer3 = self._make_transpose_layer(base_channels * 2, base_channels, 2, stride=2)
        
        # Final output layer
        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def _make_transpose_layer(self, in_channels: int, out_channels: int, 
                            num_blocks: int, stride: int) -> nn.Sequential:
        """Create a layer with multiple residual transpose blocks."""
        layers = []
        layers.append(ResidualTransposeBlock(in_channels, out_channels, stride))
        
        for _ in range(num_blocks - 1):
            layers.append(ResidualTransposeBlock(out_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
        
        Returns:
            x_hat: Reconstructed images [batch_size, channels, height, width]
        """
        # Linear transformation and reshape
        out = self.fc(z)
        out = out.view(out.size(0), self.base_channels * 8, self.init_size, self.init_size)
        
        # Residual transpose layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Final convolution
        out = self.conv_out(out)
        
        # Ensure correct output size
        if out.size(-1) != self.img_size:
            out = F.interpolate(out, size=(self.img_size, self.img_size), 
                              mode='bilinear', align_corners=False)
        
        return out


class VAEResNet(nn.Module):
    """ResNet-based Variational Autoencoder."""
    
    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        
        # Encoder and decoder
        self.encoder = ResNetEncoder(in_channels, latent_dim, base_channels)
        self.decoder = ResNetDecoder(latent_dim, in_channels, base_channels)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * eps
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            z: Sampled latent vectors
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through VAE.
        
        Args:
            x: Input images [batch_size, channels, height, width]
        
        Returns:
            x_hat: Reconstructed images
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        # Encode
        mu, logvar = self.encode(x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_hat = self.decode(z)
        
        return x_hat, mu, logvar
    
    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """
        Sample new images from prior distribution.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
        
        Returns:
            Generated images
        """
        self.eval()
        with torch.no_grad():
            # Sample from prior N(0, I)
            z = torch.randn(num_samples, self.latent_dim, device=device)
            
            # Decode to images
            samples = self.decode(z)
            
        return samples
    
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct input images.
        
        Args:
            x: Input images
        
        Returns:
            Reconstructed images
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            # Use mean (no sampling) for deterministic reconstruction
            x_hat = self.decode(mu)
        return x_hat
    
    def interpolate(self, x1: torch.Tensor, x2: torch.Tensor, num_steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two images in latent space.
        
        Args:
            x1: First image
            x2: Second image
            num_steps: Number of interpolation steps
        
        Returns:
            Interpolated images
        """
        self.eval()
        with torch.no_grad():
            # Encode both images
            mu1, _ = self.encode(x1)
            mu2, _ = self.encode(x2)
            
            # Create interpolation coefficients
            alphas = torch.linspace(0, 1, num_steps, device=x1.device)
            
            # Interpolate in latent space
            interpolated_z = []
            for alpha in alphas:
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                interpolated_z.append(z_interp)
            
            # Decode all interpolated vectors
            z_stack = torch.cat(interpolated_z, dim=0)
            interpolated_images = self.decode(z_stack)
            
        return interpolated_images