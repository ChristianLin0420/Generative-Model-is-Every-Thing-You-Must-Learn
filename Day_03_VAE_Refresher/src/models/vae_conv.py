"""
Convolutional Variational Autoencoder (VAE) implementation.
Default architecture with CNN encoder and decoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Encoder(nn.Module):
    """CNN Encoder that outputs mean and log-variance of latent distribution."""
    
    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Second conv block
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Third conv block
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Fourth conv block (optional, for larger images)
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )
        
        # Calculate the size after convolutions
        # For 28x28 (MNIST): 28 -> 14 -> 7 -> 3 -> 1 (with padding adjustments)
        # For 32x32 (CIFAR): 32 -> 16 -> 8 -> 4 -> 2
        self.conv_output_size = self._get_conv_output_size()
        
        # Linear layers for latent parameters
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)
        
    def _get_conv_output_size(self) -> int:
        """Calculate the output size of convolutional layers."""
        # Create a dummy input to calculate output size
        # Use batch size > 1 to avoid BatchNorm issues
        if self.in_channels == 1:  # MNIST
            dummy_input = torch.zeros(2, self.in_channels, 28, 28)
        else:  # CIFAR-10
            dummy_input = torch.zeros(2, self.in_channels, 32, 32)
        
        # Set to eval mode temporarily to avoid BatchNorm issues
        training_mode = self.conv_layers.training
        self.conv_layers.eval()
        
        with torch.no_grad():
            conv_output = self.conv_layers(dummy_input)
            output_size = conv_output.numel() // 2  # Divide by batch size
        
        # Restore training mode
        self.conv_layers.train(training_mode)
        
        return output_size
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Input images [batch_size, channels, height, width]
        
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        # Convolutional feature extraction
        conv_out = self.conv_layers(x)
        
        # Flatten for linear layers
        flattened = conv_out.view(conv_out.size(0), -1)
        
        # Compute latent parameters
        mu = self.fc_mu(flattened)
        logvar = self.fc_logvar(flattened)
        
        return mu, logvar


class Decoder(nn.Module):
    """CNN Decoder that reconstructs images from latent vectors."""
    
    def __init__(self, latent_dim: int, out_channels: int, base_channels: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.base_channels = base_channels
        
        # Calculate the size after convolutions (should match encoder)
        if out_channels == 1:  # MNIST
            self.init_size = 2  # 2x2 for MNIST after 4 conv layers
            self.img_size = 28
        else:  # CIFAR-10
            self.init_size = 2  # 2x2 for CIFAR after 4 conv layers  
            self.img_size = 32
        
        # Initial linear layer
        self.fc = nn.Linear(latent_dim, base_channels * 8 * self.init_size * self.init_size)
        
        # Transposed convolutional layers (decoder)
        self.deconv_layers = nn.Sequential(
            # First deconv block
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 
                             kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            
            # Second deconv block
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 
                             kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            
            # Third deconv block
            nn.ConvTranspose2d(base_channels * 2, base_channels, 
                             kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            
            # Final layer
            nn.ConvTranspose2d(base_channels, out_channels, 
                             kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
        
        Returns:
            x_hat: Reconstructed images [batch_size, channels, height, width]
        """
        # Linear transformation
        fc_out = self.fc(z)
        
        # Reshape to feature maps
        deconv_input = fc_out.view(fc_out.size(0), self.base_channels * 8, 
                                 self.init_size, self.init_size)
        
        # Transposed convolutions
        x_hat = self.deconv_layers(deconv_input)
        
        # Ensure correct output size
        if x_hat.size(-1) != self.img_size:
            x_hat = F.interpolate(x_hat, size=(self.img_size, self.img_size), 
                                mode='bilinear', align_corners=False)
        
        return x_hat


class VAEConv(nn.Module):
    """Convolutional Variational Autoencoder."""
    
    def __init__(self, in_channels: int, latent_dim: int, base_channels: int = 64):
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        
        # Encoder and decoder
        self.encoder = Encoder(in_channels, latent_dim, base_channels)
        self.decoder = Decoder(latent_dim, in_channels, base_channels)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.xavier_uniform_(module.weight)
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