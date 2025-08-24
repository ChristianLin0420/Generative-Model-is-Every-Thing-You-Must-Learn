"""Tests for UNet tiny model

Verifies:
- Model architecture correctness
- Input/output shapes
- Time embedding functionality
- Parameter count sanity checks
"""

import pytest
import torch
from src.models.unet_tiny import UNetTiny
from src.utils import count_parameters


class TestUNetTiny:
    """Test UNet tiny model."""
    
    def test_model_initialization(self):
        """Test model can be initialized with different configurations."""
        # MNIST configuration
        model_mnist = UNetTiny(
            in_channels=1,
            out_channels=1,
            model_channels=64,
            channel_mult=[1, 2, 2]
        )
        assert model_mnist is not None
        
        # CIFAR-10 configuration
        model_cifar = UNetTiny(
            in_channels=3,
            out_channels=3,
            model_channels=128,
            channel_mult=[1, 2, 2, 2]
        )
        assert model_cifar is not None
    
    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        batch_size = 4
        
        # Test MNIST
        model_mnist = UNetTiny(in_channels=1, out_channels=1, model_channels=32)
        x_mnist = torch.randn(batch_size, 1, 32, 32)
        t_mnist = torch.randint(0, 100, (batch_size,))
        
        with torch.no_grad():
            out_mnist = model_mnist(x_mnist, t_mnist)
        
        assert out_mnist.shape == x_mnist.shape, f"Expected {x_mnist.shape}, got {out_mnist.shape}"
        
        # Test CIFAR-10
        model_cifar = UNetTiny(in_channels=3, out_channels=3, model_channels=32)
        x_cifar = torch.randn(batch_size, 3, 32, 32)
        t_cifar = torch.randint(0, 200, (batch_size,))
        
        with torch.no_grad():
            out_cifar = model_cifar(x_cifar, t_cifar)
        
        assert out_cifar.shape == x_cifar.shape, f"Expected {x_cifar.shape}, got {out_cifar.shape}"
    
    def test_time_embedding_broadcast(self):
        """Test time embeddings work with different batch sizes."""
        model = UNetTiny(in_channels=3, out_channels=3, model_channels=32)
        
        # Test different batch sizes
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 3, 32, 32)
            t = torch.randint(0, 100, (batch_size,))
            
            with torch.no_grad():
                out = model(x, t)
            
            assert out.shape == x.shape
    
    def test_different_image_sizes(self):
        """Test model works with different image sizes."""
        model = UNetTiny(in_channels=1, out_channels=1, model_channels=32)
        
        # Test different sizes (must be multiples of 2^num_levels)
        sizes = [16, 32, 64]
        
        for size in sizes:
            x = torch.randn(2, 1, size, size)
            t = torch.randint(0, 100, (2,))
            
            with torch.no_grad():
                out = model(x, t)
            
            assert out.shape == x.shape
    
    def test_parameter_count_reasonable(self):
        """Test parameter count is reasonable for tiny model."""
        # Small MNIST model
        model_small = UNetTiny(
            in_channels=1, out_channels=1,
            model_channels=32, channel_mult=[1, 2]
        )
        
        params_small = count_parameters(model_small)
        assert params_small['total'] < 500_000, "Small model should have < 500K parameters"
        
        # Larger CIFAR model
        model_large = UNetTiny(
            in_channels=3, out_channels=3,
            model_channels=128, channel_mult=[1, 2, 2, 2]
        )
        
        params_large = count_parameters(model_large)
        assert params_large['total'] < 50_000_000, "Large model should have < 50M parameters"
        assert params_large['total'] > params_small['total'], "Large model should have more parameters"
    
    def test_gradient_flow(self):
        """Test gradients flow properly through the model."""
        model = UNetTiny(in_channels=3, out_channels=3, model_channels=32)
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        t = torch.randint(0, 100, (2,))
        
        out = model(x, t)
        loss = out.mean()
        loss.backward()
        
        # Check input gradients exist
        assert x.grad is not None, "Gradients should flow to input"
        
        # Check model parameter gradients
        param_grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(param_grads) > 0, "Some model parameters should have gradients"
    
    def test_deterministic_output(self):
        """Test model gives same output for same input (with fixed weights)."""
        model = UNetTiny(in_channels=1, out_channels=1, model_channels=32)
        model.eval()  # Set to eval mode to disable dropout
        
        x = torch.randn(2, 1, 32, 32)
        t = torch.randint(0, 100, (2,))
        
        with torch.no_grad():
            out1 = model(x, t)
            out2 = model(x, t)
        
        assert torch.allclose(out1, out2, atol=1e-6), "Model should be deterministic in eval mode"
    
    def test_attention_blocks(self):
        """Test model with attention blocks."""
        model = UNetTiny(
            in_channels=3, out_channels=3,
            model_channels=64,
            attention_resolutions=[16, 32],
            num_heads=4
        )
        
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 100, (2,))
        
        with torch.no_grad():
            out = model(x, t)
        
        assert out.shape == x.shape
    
    def test_device_consistency(self):
        """Test model works on different devices."""
        devices = [torch.device('cpu')]
        
        # Add CUDA if available
        if torch.cuda.is_available():
            devices.append(torch.device('cuda'))
        
        for device in devices:
            model = UNetTiny(in_channels=1, out_channels=1, model_channels=32)
            model = model.to(device)
            
            x = torch.randn(2, 1, 32, 32, device=device)
            t = torch.randint(0, 100, (2,), device=device)
            
            with torch.no_grad():
                out = model(x, t)
            
            assert out.device == device, f"Output should be on {device}"
            assert out.shape == x.shape


def test_model_smoke():
    """Quick smoke test to ensure model can be created and run."""
    model = UNetTiny(in_channels=1, out_channels=1, model_channels=16)
    x = torch.randn(1, 1, 32, 32)
    t = torch.randint(0, 10, (1,))
    
    with torch.no_grad():
        out = model(x, t)
    
    assert out.shape == x.shape
    print("✓ UNet smoke test passed")


if __name__ == "__main__":
    test_model_smoke()
    print("Run with: python -m pytest tests/test_unet_tiny.py -v")