"""
Test suite for VAE models.
Tests model architectures, forward passes, and reparameterization.
"""

import pytest
import torch
import torch.nn as nn

from src.models.vae_conv import VAEConv
from src.models.vae_resnet import VAEResNet
from src.models.vae_mlp import VAEMLP


class TestVAEModels:
    """Test VAE model implementations."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")  # Use CPU for tests
    
    @pytest.fixture
    def mnist_batch(self, device):
        """Sample MNIST batch."""
        return torch.randn(4, 1, 28, 28, device=device)
    
    @pytest.fixture
    def cifar_batch(self, device):
        """Sample CIFAR-10 batch."""
        return torch.randn(4, 3, 32, 32, device=device)
    
    def test_vae_conv_mnist_shapes(self, mnist_batch, device):
        """Test VAEConv with MNIST input shapes."""
        model = VAEConv(in_channels=1, latent_dim=32, base_channels=32).to(device)
        
        # Test forward pass
        x_hat, mu, logvar = model(mnist_batch)
        
        # Check shapes
        assert x_hat.shape == mnist_batch.shape
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)
        
        # Test encode/decode separately
        mu_enc, logvar_enc = model.encode(mnist_batch)
        assert mu_enc.shape == (4, 32)
        assert logvar_enc.shape == (4, 32)
        
        z = torch.randn(4, 32, device=device)
        decoded = model.decode(z)
        assert decoded.shape == mnist_batch.shape
    
    def test_vae_conv_cifar_shapes(self, cifar_batch, device):
        """Test VAEConv with CIFAR-10 input shapes."""
        model = VAEConv(in_channels=3, latent_dim=64, base_channels=32).to(device)
        
        # Test forward pass
        x_hat, mu, logvar = model(cifar_batch)
        
        # Check shapes
        assert x_hat.shape == cifar_batch.shape
        assert mu.shape == (4, 64)
        assert logvar.shape == (4, 64)
    
    def test_vae_resnet_shapes(self, mnist_batch, device):
        """Test VAEResNet shapes."""
        model = VAEResNet(in_channels=1, latent_dim=32, base_channels=32).to(device)
        
        # Test forward pass
        x_hat, mu, logvar = model(mnist_batch)
        
        # Check shapes
        assert x_hat.shape == mnist_batch.shape
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)
    
    def test_vae_mlp_shapes(self, mnist_batch, device):
        """Test VAEMLP shapes."""
        model = VAEMLP(in_channels=1, img_size=28, latent_dim=32).to(device)
        
        # Test forward pass
        x_hat, mu, logvar = model(mnist_batch)
        
        # Check shapes
        assert x_hat.shape == mnist_batch.shape
        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)
    
    def test_reparameterization_trick(self, device):
        """Test reparameterization trick implementation."""
        model = VAEConv(in_channels=1, latent_dim=10, base_channels=16).to(device)
        
        mu = torch.randn(4, 10, device=device)
        logvar = torch.randn(4, 10, device=device)
        
        # Sample multiple times
        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)
        
        # Check shapes
        assert z1.shape == (4, 10)
        assert z2.shape == (4, 10)
        
        # Should be different due to randomness (with very high probability)
        assert not torch.allclose(z1, z2, atol=1e-6)
        
        # With zero variance, should be deterministic
        z_det1 = model.reparameterize(mu, torch.full_like(logvar, -10))  # Very small variance
        z_det2 = model.reparameterize(mu, torch.full_like(logvar, -10))
        
        # Should be very close to mu and to each other
        assert torch.allclose(z_det1, mu, atol=1e-3)
        assert torch.allclose(z_det1, z_det2, atol=1e-6)
    
    def test_sampling_from_prior(self, device):
        """Test sampling from prior distribution."""
        model = VAEConv(in_channels=1, latent_dim=16, base_channels=16).to(device)
        model.eval()
        
        # Sample from prior
        samples = model.sample(num_samples=8, device=device)
        
        # Check shape
        assert samples.shape == (8, 1, 28, 28)
        
        # Samples should be different
        assert not torch.allclose(samples[0], samples[1], atol=1e-6)
    
    def test_reconstruction(self, mnist_batch, device):
        """Test reconstruction functionality."""
        model = VAEConv(in_channels=1, latent_dim=16, base_channels=16).to(device)
        model.eval()
        
        # Get reconstructions
        reconstructions = model.reconstruct(mnist_batch)
        
        # Check shape
        assert reconstructions.shape == mnist_batch.shape
    
    def test_interpolation(self, mnist_batch, device):
        """Test latent space interpolation."""
        model = VAEConv(in_channels=1, latent_dim=16, base_channels=16).to(device)
        model.eval()
        
        x1 = mnist_batch[0:1]  # First image
        x2 = mnist_batch[1:2]  # Second image
        
        # Interpolate
        interpolated = model.interpolate(x1, x2, num_steps=5)
        
        # Check shape
        assert interpolated.shape == (5, 1, 28, 28)
        
        # First and last should be close to originals (after encoding/decoding)
        recon1 = model.reconstruct(x1)
        recon2 = model.reconstruct(x2)
        
        # Note: We compare with reconstructions, not originals, due to VAE stochasticity
        assert torch.allclose(interpolated[0:1], recon1, atol=0.1)
        assert torch.allclose(interpolated[-1:], recon2, atol=0.1)
    
    def test_model_parameter_counts(self, device):
        """Test that models have reasonable parameter counts."""
        models = [
            VAEConv(in_channels=1, latent_dim=32, base_channels=32),
            VAEResNet(in_channels=1, latent_dim=32, base_channels=32),
            VAEMLP(in_channels=1, img_size=28, latent_dim=32)
        ]
        
        for model in models:
            model = model.to(device)
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Should have at least 1000 parameters but not too many for test models
            assert 1000 < param_count < 10_000_000
    
    def test_gradient_flow(self, mnist_batch, device):
        """Test that gradients flow properly through the model."""
        model = VAEConv(in_channels=1, latent_dim=16, base_channels=16).to(device)
        model.train()
        
        # Forward pass
        x_hat, mu, logvar = model(mnist_batch)
        
        # Compute a simple loss
        recon_loss = nn.MSELoss()(x_hat, mnist_batch)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + 0.1 * kl_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check that gradients exist and are not zero for all parameters
        grad_norms = []
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
                grad_norms.append(param.grad.norm().item())
        
        # At least some gradients should be non-zero
        assert sum(grad_norms) > 0
    
    def test_eval_mode_consistency(self, mnist_batch, device):
        """Test that model outputs are consistent in eval mode."""
        model = VAEConv(in_channels=1, latent_dim=16, base_channels=16).to(device)
        model.eval()
        
        with torch.no_grad():
            # Reconstruction should be deterministic in eval mode (using mu)
            recon1 = model.reconstruct(mnist_batch)
            recon2 = model.reconstruct(mnist_batch)
            
            assert torch.allclose(recon1, recon2, atol=1e-6)
    
    @pytest.mark.parametrize("latent_dim", [2, 16, 64, 128])
    def test_different_latent_dimensions(self, mnist_batch, device, latent_dim):
        """Test models with different latent dimensions."""
        model = VAEConv(in_channels=1, latent_dim=latent_dim, base_channels=16).to(device)
        
        x_hat, mu, logvar = model(mnist_batch)
        
        assert x_hat.shape == mnist_batch.shape
        assert mu.shape == (4, latent_dim)
        assert logvar.shape == (4, latent_dim)
    
    def test_model_state_dict_save_load(self, device):
        """Test model state dict saving and loading."""
        model1 = VAEConv(in_channels=1, latent_dim=16, base_channels=16).to(device)
        model2 = VAEConv(in_channels=1, latent_dim=16, base_channels=16).to(device)
        
        # Initialize with different weights
        with torch.no_grad():
            for p1, p2 in zip(model1.parameters(), model2.parameters()):
                p1.normal_()
                p2.normal_()
        
        # Ensure they're different
        input_tensor = torch.randn(2, 1, 28, 28, device=device)
        out1 = model1(input_tensor)[0]
        out2 = model2(input_tensor)[0]
        assert not torch.allclose(out1, out2, atol=1e-6)
        
        # Save and load state dict
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)
        
        # Should now be identical
        out1_new = model1(input_tensor)[0]
        out2_new = model2(input_tensor)[0]
        assert torch.allclose(out1_new, out2_new, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])