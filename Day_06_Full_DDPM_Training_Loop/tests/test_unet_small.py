"""
Test suite for UNet model: forward shapes, parameter count, time conditioning
"""

import pytest
import torch
import torch.nn as nn
from src.models.unet_small import UNetSmall
from src.models.time_embedding import TimeEmbedding


class TestUNetSmall:
    """Test UNet small model"""
    
    def test_model_initialization(self):
        """Test model can be initialized with default parameters"""
        model = UNetSmall()
        
        # Check it's a nn.Module
        assert isinstance(model, nn.Module)
        
        # Check default values
        assert model.in_channels == 3
        assert model.out_channels == 3
        assert model.model_channels == 32
        
    def test_forward_shapes_rgb(self):
        """Test forward pass shapes for RGB images"""
        batch_size = 4
        height, width = 32, 32
        
        model = UNetSmall(
            in_channels=3,
            out_channels=3,
            model_channels=32
        )
        model.eval()
        
        # Input tensor and timesteps
        x = torch.randn(batch_size, 3, height, width)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        # Forward pass
        with torch.no_grad():
            output = model(x, timesteps)
            
        # Check output shape
        assert output.shape == x.shape
        assert output.dtype == x.dtype
        
    def test_forward_shapes_grayscale(self):
        """Test forward pass shapes for grayscale images"""
        batch_size = 2
        height, width = 28, 28
        
        model = UNetSmall(
            in_channels=1,
            out_channels=1,
            model_channels=32,
            channel_mult=[1, 2, 2]
        )
        model.eval()
        
        # Input tensor and timesteps
        x = torch.randn(batch_size, 1, height, width)
        timesteps = torch.randint(0, 1000, (batch_size,))
        
        # Forward pass
        with torch.no_grad():
            output = model(x, timesteps)
            
        # Check output shape matches input
        assert output.shape == (batch_size, 1, height, width)
        
    def test_different_image_sizes(self):
        """Test model works with different image sizes"""
        model = UNetSmall(model_channels=16)  # Smaller for speed
        model.eval()
        
        batch_size = 1
        timesteps = torch.tensor([500])
        
        # Test various sizes
        for size in [16, 32, 64]:
            x = torch.randn(batch_size, 3, size, size)
            
            with torch.no_grad():
                output = model(x, timesteps)
                
            assert output.shape == x.shape
            
    def test_time_conditioning(self):
        """Test time conditioning is properly applied"""
        model = UNetSmall(model_channels=16)
        model.eval()
        
        x = torch.randn(2, 3, 32, 32)
        
        # Different timesteps should produce different outputs
        t1 = torch.tensor([100, 100])
        t2 = torch.tensor([900, 900])
        
        with torch.no_grad():
            out1 = model(x, t1)
            out2 = model(x, t2)
            
        # Outputs should be different due to time conditioning
        assert not torch.allclose(out1, out2, atol=1e-3)
        
    def test_batch_size_consistency(self):
        """Test model works with different batch sizes"""
        model = UNetSmall(model_channels=16)
        model.eval()
        
        x_single = torch.randn(1, 3, 32, 32)
        x_batch = torch.randn(8, 3, 32, 32)
        
        t_single = torch.tensor([500])
        t_batch = torch.tensor([500] * 8)
        
        with torch.no_grad():
            out_single = model(x_single, t_single)
            out_batch = model(x_batch, t_batch)
            
        assert out_single.shape == (1, 3, 32, 32)
        assert out_batch.shape == (8, 3, 32, 32)
        
        # First item of batch should match single item (with same input)
        x_batch_first = x_batch[:1]
        t_batch_first = t_batch[:1]
        
        # Set same input
        x_batch_first.data = x_single.data
        
        with torch.no_grad():
            out_batch_first = model(x_batch_first, t_batch_first)
            
        assert torch.allclose(out_single, out_batch_first, atol=1e-5)
        
    def test_parameter_count(self):
        """Test parameter count is reasonable"""
        # Small model
        small_model = UNetSmall(
            model_channels=16,
            channel_mult=[1, 2],
            num_res_blocks=1,
            use_attention=False
        )
        small_params = small_model.count_parameters()
        
        # Larger model
        large_model = UNetSmall(
            model_channels=64,
            channel_mult=[1, 2, 4, 8],
            num_res_blocks=2,
            use_attention=True
        )
        large_params = large_model.count_parameters()
        
        # Larger model should have more parameters
        assert large_params > small_params
        
        # Parameter counts should be reasonable (not too small or huge)
        assert small_params > 1000  # At least some parameters
        assert small_params < 10_000_000  # Not too many
        assert large_params > small_params
        assert large_params < 100_000_000  # Still reasonable
        
    def test_attention_configuration(self):
        """Test attention can be enabled/disabled"""
        # Model without attention
        model_no_attn = UNetSmall(
            model_channels=32,
            use_attention=False
        )
        
        # Model with attention
        model_with_attn = UNetSmall(
            model_channels=32,
            use_attention=True,
            attention_resolutions=[16]
        )
        
        # Both should work
        x = torch.randn(1, 3, 32, 32)
        t = torch.tensor([500])
        
        with torch.no_grad():
            out_no_attn = model_no_attn(x, t)
            out_with_attn = model_with_attn(x, t)
            
        assert out_no_attn.shape == x.shape
        assert out_with_attn.shape == x.shape
        
        # Model with attention should have more parameters
        assert model_with_attn.count_parameters() > model_no_attn.count_parameters()
        
    def test_channel_multipliers(self):
        """Test different channel multiplier configurations"""
        base_channels = 32
        
        # Test various configurations
        configs = [
            [1, 2],
            [1, 2, 4],
            [1, 2, 4, 8],
            [1, 1, 2, 2]  # Non-standard
        ]
        
        for channel_mult in configs:
            model = UNetSmall(
                model_channels=base_channels,
                channel_mult=channel_mult,
                num_res_blocks=1  # Keep small for speed
            )
            
            x = torch.randn(1, 3, 32, 32)
            t = torch.tensor([100])
            
            with torch.no_grad():
                output = model(x, t)
                
            assert output.shape == x.shape
            
    def test_gradient_flow(self):
        """Test gradients flow properly through the model"""
        model = UNetSmall(model_channels=16)
        model.train()
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        t = torch.tensor([100, 500])
        
        output = model(x, t)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist and are non-zero for most parameters
        non_zero_grads = 0
        total_params = 0
        
        for param in model.parameters():
            if param.grad is not None:
                total_params += 1
                if torch.any(param.grad != 0):
                    non_zero_grads += 1
                    
        # Most parameters should have gradients
        assert non_zero_grads / total_params > 0.5
        
    def test_time_embedding_dimensions(self):
        """Test time embedding dimensions are consistent"""
        time_embed_dims = [64, 128, 256, 512]
        
        for embed_dim in time_embed_dims:
            model = UNetSmall(
                model_channels=16,
                time_embed_dim=embed_dim
            )
            
            # Check time embedding layer
            assert model.time_embed.embed_dim == embed_dim
            
            # Model should still work
            x = torch.randn(1, 3, 32, 32)
            t = torch.tensor([500])
            
            with torch.no_grad():
                output = model(x, t)
                
            assert output.shape == x.shape
            
    def test_deterministic_output(self):
        """Test model produces deterministic output with same inputs"""
        model = UNetSmall(model_channels=16)
        model.eval()
        
        x = torch.randn(2, 3, 32, 32)
        t = torch.tensor([100, 500])
        
        with torch.no_grad():
            out1 = model(x, t)
            out2 = model(x, t)
            
        # Should be identical
        assert torch.allclose(out1, out2, atol=1e-6)
        
    def test_extreme_timesteps(self):
        """Test model handles extreme timestep values"""
        model = UNetSmall(model_channels=16)
        model.eval()
        
        x = torch.randn(3, 3, 32, 32)
        
        # Test extreme timesteps
        extreme_timesteps = [
            torch.tensor([0, 0, 0]),      # Minimum
            torch.tensor([999, 999, 999]), # Maximum
            torch.tensor([0, 500, 999])   # Mixed
        ]
        
        for timesteps in extreme_timesteps:
            with torch.no_grad():
                output = model(x, timesteps)
                
            # Should produce valid output
            assert output.shape == x.shape
            assert torch.all(torch.isfinite(output))
            
    def test_memory_efficiency(self):
        """Test model doesn't use excessive memory"""
        model = UNetSmall(model_channels=32)
        model.eval()
        
        # Large batch to test memory usage
        large_batch_size = 16
        x = torch.randn(large_batch_size, 3, 32, 32)
        t = torch.randint(0, 1000, (large_batch_size,))
        
        # Should complete without memory error
        with torch.no_grad():
            output = model(x, t)
            
        assert output.shape == x.shape
        
        # Clean up
        del x, t, output