"""
Test suite for visualization functions.
Tests that visualizations can be created without errors.
"""

import pytest
import torch
import os
import tempfile
import matplotlib.pyplot as plt

from src.visualize import (
    create_reconstruction_grid, create_latent_traversal,
    create_multi_dim_traversal, plot_latent_distribution_comparison
)
from src.models.vae_conv import VAEConv
from src.dataset import create_unlabeled_dataloaders


class TestVisualization:
    """Test visualization functionality."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    @pytest.fixture
    def model(self, device):
        """Small VAE model for testing."""
        return VAEConv(in_channels=1, latent_dim=8, base_channels=8).to(device)
    
    @pytest.fixture
    def sample_images(self, device):
        """Sample images for testing."""
        return torch.randn(16, 1, 28, 28, device=device)
    
    def test_create_reconstruction_grid(self, model, sample_images, device):
        """Test reconstruction grid creation."""
        model.eval()
        
        original, reconstructed = create_reconstruction_grid(
            model, sample_images, device, num_images=8
        )
        
        # Check shapes
        assert original.shape == (8, 1, 28, 28)
        assert reconstructed.shape == (8, 1, 28, 28)
        
        # Check that they're different (reconstruction should differ from original)
        # Note: With random weights, they should be quite different
        mse = torch.mean((original - reconstructed)**2).item()
        assert mse > 0.01  # Should have significant reconstruction error with random weights
    
    def test_create_latent_traversal(self, model, sample_images, device):
        """Test latent traversal creation."""
        model.eval()
        
        base_image = sample_images[0:1]
        traversal = create_latent_traversal(
            model, base_image, device, dim_to_traverse=0, num_steps=5
        )
        
        # Check shape
        assert traversal.shape == (5, 1, 28, 28)
        
        # Images should be different across traversal
        assert not torch.allclose(traversal[0], traversal[-1], atol=1e-3)
    
    def test_create_multi_dim_traversal(self, model, sample_images, device):
        """Test multi-dimensional traversal."""
        model.eval()
        
        base_image = sample_images[0:1]
        traversals = create_multi_dim_traversal(
            model, base_image, device, dims_to_traverse=[0, 1, 2], num_steps=5
        )
        
        # Check that we get traversals for all requested dimensions
        assert len(traversals) == 3
        assert 0 in traversals
        assert 1 in traversals
        assert 2 in traversals
        
        # Each traversal should have correct shape
        for traversal in traversals.values():
            assert traversal.shape == (5, 1, 28, 28)
    
    def test_plot_latent_distribution_comparison(self, model, device):
        """Test latent distribution plotting."""
        model.eval()
        
        # Create a small dataset for testing
        sample_data = torch.randn(32, 1, 28, 28, device=device)
        
        # Create a simple dataloader-like object
        class SimpleDataLoader:
            def __init__(self, data):
                self.data = data
            
            def __iter__(self):
                batch_size = 8
                for i in range(0, len(self.data), batch_size):
                    yield self.data[i:i+batch_size]
        
        dataloader = SimpleDataLoader(sample_data)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = os.path.join(temp_dir, "test_plot.png")
            
            # This should not raise an exception
            fig = plot_latent_distribution_comparison(
                model, dataloader, device, num_batches=2, save_path=save_path
            )
            
            # Check that figure was created
            assert isinstance(fig, plt.Figure)
            
            # Check that file was saved
            assert os.path.exists(save_path)
            
            plt.close(fig)  # Clean up
    
    def test_model_2d_latent_space(self, device):
        """Test visualization with 2D latent space."""
        # Create model with 2D latent space
        model = VAEConv(in_channels=1, latent_dim=2, base_channels=8).to(device)
        model.eval()
        
        sample_images = torch.randn(8, 1, 28, 28, device=device)
        
        # Test latent traversal with 2D space
        traversal = create_latent_traversal(
            model, sample_images[0:1], device, dim_to_traverse=0, num_steps=3
        )
        
        assert traversal.shape == (3, 1, 28, 28)
        
        # Test multi-dim traversal with both dimensions
        traversals = create_multi_dim_traversal(
            model, sample_images[0:1], device, dims_to_traverse=[0, 1], num_steps=3
        )
        
        assert len(traversals) == 2
    
    def test_visualization_with_different_models(self, device):
        """Test visualization works with different model architectures."""
        from src.models.vae_mlp import VAEMLP
        from src.models.vae_resnet import VAEResNet
        
        models = [
            VAEConv(in_channels=1, latent_dim=4, base_channels=8),
            VAEMLP(in_channels=1, img_size=28, latent_dim=4),
            VAEResNet(in_channels=1, latent_dim=4, base_channels=8)
        ]
        
        sample_images = torch.randn(4, 1, 28, 28, device=device)
        
        for model in models:
            model = model.to(device).eval()
            
            # Test reconstruction grid
            original, reconstructed = create_reconstruction_grid(
                model, sample_images, device, num_images=2
            )
            
            assert original.shape == (2, 1, 28, 28)
            assert reconstructed.shape == (2, 1, 28, 28)
            
            # Test latent traversal
            traversal = create_latent_traversal(
                model, sample_images[0:1], device, dim_to_traverse=0, num_steps=3
            )
            
            assert traversal.shape == (3, 1, 28, 28)
    
    def test_visualization_edge_cases(self, model, device):
        """Test visualization with edge cases."""
        model.eval()
        
        # Single image
        single_image = torch.randn(1, 1, 28, 28, device=device)
        
        original, reconstructed = create_reconstruction_grid(
            model, single_image, device, num_images=1
        )
        
        assert original.shape == (1, 1, 28, 28)
        assert reconstructed.shape == (1, 1, 28, 28)
        
        # Single step traversal
        traversal = create_latent_traversal(
            model, single_image, device, dim_to_traverse=0, num_steps=1
        )
        
        assert traversal.shape == (1, 1, 28, 28)
        
        # Empty dimensions list
        traversals = create_multi_dim_traversal(
            model, single_image, device, dims_to_traverse=[], num_steps=3
        )
        
        assert len(traversals) == 0
    
    def test_traversal_parameter_ranges(self, model, sample_images, device):
        """Test latent traversal with different parameter ranges."""
        model.eval()
        base_image = sample_images[0:1]
        
        # Different step sizes should produce different results
        traversal_small = create_latent_traversal(
            model, base_image, device, dim_to_traverse=0, 
            num_steps=3, step_size=1.0
        )
        
        traversal_large = create_latent_traversal(
            model, base_image, device, dim_to_traverse=0, 
            num_steps=3, step_size=5.0
        )
        
        # With larger step size, the difference should be more pronounced
        diff_small = torch.mean((traversal_small[0] - traversal_small[-1])**2)
        diff_large = torch.mean((traversal_large[0] - traversal_large[-1])**2)
        
        # This might not always be true due to decoder nonlinearity, but generally should hold
        assert diff_large.item() >= diff_small.item() * 0.5  # Allow some tolerance
    
    @pytest.mark.parametrize("latent_dim", [2, 8, 16])
    def test_different_latent_dimensions(self, device, latent_dim):
        """Test visualization with different latent dimensions."""
        model = VAEConv(in_channels=1, latent_dim=latent_dim, base_channels=8).to(device)
        model.eval()
        
        sample_images = torch.randn(4, 1, 28, 28, device=device)
        
        # Test traversal for first dimension
        traversal = create_latent_traversal(
            model, sample_images[0:1], device, dim_to_traverse=0, num_steps=3
        )
        
        assert traversal.shape == (3, 1, 28, 28)
        
        # Test multi-dim traversal for first few dimensions
        max_dims = min(latent_dim, 3)
        dims_to_test = list(range(max_dims))
        
        traversals = create_multi_dim_traversal(
            model, sample_images[0:1], device, dims_to_traverse=dims_to_test, num_steps=3
        )
        
        assert len(traversals) == max_dims


if __name__ == "__main__":
    pytest.main([__file__])