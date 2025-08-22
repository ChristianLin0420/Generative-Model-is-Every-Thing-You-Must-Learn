"""
MLP-based Variational Autoencoder implementation.
Simple fully-connected architecture, primarily for MNIST (28x28).
"""

import torch
import torch.nn as nn
from typing import Tuple


class MLPEncoder(nn.Module):
    """MLP Encoder for VAE."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = [512, 256]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
        
        self.encoder = nn.Sequential(*layers)
        
        # Output layers for latent parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through encoder.
        
        Args:
            x: Flattened input [batch_size, input_dim]
        
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar


class MLPDecoder(nn.Module):
    """MLP Decoder for VAE."""
    
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: list = [256, 512]):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        dims = [latent_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
        
        # Final output layer
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through decoder.
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
        
        Returns:
            x_hat: Reconstructed flattened images [batch_size, output_dim]
        """
        return self.decoder(z)


class VAEMLP(nn.Module):
    """MLP-based Variational Autoencoder."""
    
    def __init__(
        self, 
        in_channels: int, 
        img_size: int, 
        latent_dim: int,
        hidden_dims: list = [512, 256]
    ):
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        
        # Calculate input/output dimensions
        self.input_dim = in_channels * img_size * img_size
        
        # Encoder and decoder
        self.encoder = MLPEncoder(self.input_dim, latent_dim, hidden_dims)
        self.decoder = MLPDecoder(latent_dim, self.input_dim, hidden_dims[::-1])
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        """Flatten image tensor."""
        return x.view(x.size(0), -1)
    
    def _unflatten(self, x: torch.Tensor) -> torch.Tensor:
        """Unflatten tensor back to image format."""
        return x.view(x.size(0), self.in_channels, self.img_size, self.img_size)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent parameters."""
        x_flat = self._flatten(x)
        return self.encoder(x_flat)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        x_flat = self.decoder(z)
        return self._unflatten(x_flat)
    
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