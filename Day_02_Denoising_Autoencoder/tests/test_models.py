"""
Tests for model architectures in Day 2: Denoising Autoencoder
"""

import pytest
import torch

from src.models import create_model, ConvDAE, UNetDAE
from src.models.dae_conv import SimpleConvDAE


class TestModelCreation:
    """Test model factory function."""
    
    def test_create_conv_model(self):
        """Test creating ConvDAE model."""
        model = create_model("conv", in_ch=1, out_ch=1)
        assert isinstance(model, ConvDAE)
    
    def test_create_unet_model(self):
        """Test creating UNet model."""
        model = create_model("unet", in_ch=3, out_ch=3)
        assert isinstance(model, UNetDAE)
    
    def test_invalid_model_name(self):
        """Test that invalid model name raises error."""
        with pytest.raises(ValueError):
            create_model("invalid", in_ch=1, out_ch=1)


class TestConvDAE:
    """Test ConvDAE architecture."""
    
    def test_conv_dae_forward_pass(self):
        """Test forward pass through ConvDAE."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=32, num_downs=2)
        x = torch.randn(2, 1, 28, 28)
        
        output = model(x)
        
        assert output.shape == x.shape
        assert torch.all(output >= 0)  # Should have sigmoid activation
        assert torch.all(output <= 1)
    
    def test_conv_dae_multi_channel(self):
        """Test ConvDAE with multi-channel input."""
        model = ConvDAE(in_ch=3, out_ch=3, base_ch=64, num_downs=3)
        x = torch.randn(4, 3, 32, 32)
        
        output = model(x)
        
        assert output.shape == x.shape
    
    def test_conv_dae_encode_decode(self):
        """Test separate encode/decode functions."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=32, num_downs=2)
        x = torch.randn(2, 1, 16, 16)
        
        # Test encoding
        z = model.encode(x)
        assert z.ndim == 4  # Should be [B, C, H, W]
        assert z.shape[0] == x.shape[0]  # Same batch size
        
        # Test decoding
        recon = model.decode(z)
        assert recon.shape == x.shape
    
    def test_conv_dae_latent_dim(self):
        """Test latent dimension computation."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=32, num_downs=2)
        
        latent_dim = model.get_latent_dim((1, 28, 28))
        
        assert len(latent_dim) == 3  # (C, H, W)
        assert all(isinstance(d, int) for d in latent_dim)
    
    def test_conv_dae_parameters(self):
        """Test that model has trainable parameters."""
        model = ConvDAE(in_ch=1, out_ch=1)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params == total_params  # All parameters should be trainable


class TestUNetDAE:
    """Test UNetDAE architecture."""
    
    def test_unet_dae_forward_pass(self):
        """Test forward pass through UNetDAE."""
        model = UNetDAE(in_ch=1, out_ch=1, base_ch=32, num_downs=3)
        x = torch.randn(2, 1, 32, 32)
        
        output = model(x)
        
        assert output.shape == x.shape
        assert torch.all(output >= 0)  # Should have sigmoid activation
        assert torch.all(output <= 1)
    
    def test_unet_dae_different_sizes(self):
        """Test UNet with different input sizes."""
        model = UNetDAE(in_ch=1, out_ch=1, base_ch=64, num_downs=2)
        
        # Test different sizes
        sizes = [(16, 16), (32, 32), (64, 64)]
        
        for h, w in sizes:
            x = torch.randn(1, 1, h, w)
            output = model(x)
            assert output.shape == x.shape, f"Failed for size {h}x{w}"
    
    def test_unet_dae_feature_maps(self):
        """Test feature map extraction if available."""
        model = UNetDAE(in_ch=1, out_ch=1, base_ch=32, num_downs=2)
        x = torch.randn(1, 1, 28, 28)
        
        if hasattr(model, 'get_feature_maps'):
            features = model.get_feature_maps(x)
            
            assert isinstance(features, dict)
            assert len(features) > 0
            
            # Check that all features are tensors
            for name, feature in features.items():
                assert isinstance(feature, torch.Tensor)
                assert feature.ndim == 4  # [B, C, H, W]
    
    def test_unet_dae_bilinear_vs_transpose(self):
        """Test both upsampling methods."""
        x = torch.randn(1, 1, 32, 32)
        
        # Bilinear upsampling
        model_bilinear = UNetDAE(in_ch=1, out_ch=1, base_ch=32, bilinear=True)
        output_bilinear = model_bilinear(x)
        
        # Transposed convolution
        model_transpose = UNetDAE(in_ch=1, out_ch=1, base_ch=32, bilinear=False)
        output_transpose = model_transpose(x)
        
        assert output_bilinear.shape == x.shape
        assert output_transpose.shape == x.shape
    
    def test_unet_dae_normalization_types(self):
        """Test different normalization types."""
        x = torch.randn(2, 1, 16, 16)
        
        for norm_type in ['batch', 'group', 'instance']:
            model = UNetDAE(in_ch=1, out_ch=1, norm_type=norm_type, base_ch=16)
            
            output = model(x)
            assert output.shape == x.shape


class TestSimpleConvDAE:
    """Test SimpleConvDAE architecture."""
    
    def test_simple_conv_dae_forward(self):
        """Test forward pass through SimpleConvDAE."""
        model = SimpleConvDAE(in_ch=1, out_ch=1, hidden_ch=16)
        x = torch.randn(2, 1, 28, 28)
        
        output = model(x)
        
        assert output.shape == x.shape
        assert torch.all(output >= 0)
        assert torch.all(output <= 1)
    
    def test_simple_conv_dae_parameters(self):
        """Test parameter count is reasonable."""
        model = SimpleConvDAE(in_ch=1, out_ch=1, hidden_ch=32)
        
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should be much smaller than full models
        assert total_params < 100000  # Less than 100k parameters


class TestModelComparison:
    """Compare different model architectures."""
    
    def test_parameter_counts(self):
        """Compare parameter counts across models."""
        input_shape = (1, 28, 28)
        
        conv_model = ConvDAE(in_ch=1, out_ch=1, base_ch=64, num_downs=2)
        unet_model = UNetDAE(in_ch=1, out_ch=1, base_ch=64, num_downs=2)
        simple_model = SimpleConvDAE(in_ch=1, out_ch=1, hidden_ch=64)
        
        conv_params = sum(p.numel() for p in conv_model.parameters())
        unet_params = sum(p.numel() for p in unet_model.parameters())
        simple_params = sum(p.numel() for p in simple_model.parameters())
        
        # UNet should have more parameters due to skip connections
        assert unet_params > conv_params
        
        # Simple model should have fewer parameters
        assert simple_params < conv_params
        
        print(f"Parameter counts:")
        print(f"  ConvDAE: {conv_params:,}")
        print(f"  UNetDAE: {unet_params:,}")
        print(f"  SimpleConvDAE: {simple_params:,}")
    
    def test_output_consistency(self):
        """Test that all models produce valid outputs."""
        x = torch.randn(2, 1, 32, 32)
        
        models = [
            ConvDAE(in_ch=1, out_ch=1, base_ch=32, num_downs=2),
            UNetDAE(in_ch=1, out_ch=1, base_ch=32, num_downs=2),
            SimpleConvDAE(in_ch=1, out_ch=1, hidden_ch=32)
        ]
        
        for model in models:
            output = model(x)
            
            # All should preserve shape
            assert output.shape == x.shape
            
            # All should output in valid range
            assert torch.all(output >= 0)
            assert torch.all(output <= 1)
            
            # Should not be all zeros or all ones
            assert torch.any(output > 0.1)
            assert torch.any(output < 0.9)


class TestModelEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_channel_to_multi_channel(self):
        """Test model with different input/output channels."""
        model = UNetDAE(in_ch=1, out_ch=3, base_ch=32)  # Grayscale to RGB
        x = torch.randn(1, 1, 32, 32)
        
        output = model(x)
        
        assert output.shape == (1, 3, 32, 32)
    
    def test_very_small_input(self):
        """Test with very small input size."""
        model = UNetDAE(in_ch=1, out_ch=1, base_ch=16, num_downs=1)  # Shallow network
        x = torch.randn(1, 1, 8, 8)
        
        output = model(x)
        assert output.shape == x.shape
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        model = ConvDAE(in_ch=1, out_ch=1)
        x = torch.randn(1, 1, 28, 28)
        
        output = model(x)
        assert output.shape == x.shape
    
    def test_model_device_placement(self):
        """Test model device placement."""
        model = UNetDAE(in_ch=1, out_ch=1)
        
        # Test CPU
        x_cpu = torch.randn(1, 1, 16, 16)
        output_cpu = model(x_cpu)
        assert output_cpu.device == x_cpu.device
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            x_gpu = x_cpu.cuda()
            output_gpu = model_gpu(x_gpu)
            assert output_gpu.device == x_gpu.device


if __name__ == '__main__':
    pytest.main([__file__])