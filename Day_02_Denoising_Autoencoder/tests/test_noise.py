"""
Tests for noise functionality in Day 2: Denoising Autoencoder
"""

import pytest
import torch
import numpy as np

from src.noise import (
    add_gaussian_noise,
    sigma_schedule, 
    get_train_test_mismatch_sigmas,
    batch_add_different_noise,
    compute_noise_statistics,
    AdaptiveNoiseScheduler
)


class TestGaussianNoise:
    """Test Gaussian noise addition functionality."""
    
    def test_add_gaussian_noise_shape(self):
        """Test that noise addition preserves tensor shape."""
        x = torch.randn(2, 3, 4, 4)
        sigma = 0.1
        
        noisy_x = add_gaussian_noise(x, sigma)
        
        assert noisy_x.shape == x.shape
    
    def test_add_gaussian_noise_zero_sigma(self):
        """Test that zero sigma returns original tensor."""
        x = torch.randn(2, 3, 4, 4)
        
        noisy_x = add_gaussian_noise(x, 0.0)
        
        assert torch.allclose(noisy_x, x)
    
    def test_add_gaussian_noise_reproducibility(self):
        """Test that noise addition is reproducible with same generator."""
        x = torch.randn(2, 3, 4, 4)
        sigma = 0.5
        
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        
        noisy_x1 = add_gaussian_noise(x, sigma, generator=gen1)
        noisy_x2 = add_gaussian_noise(x, sigma, generator=gen2)
        
        assert torch.allclose(noisy_x1, noisy_x2)
    
    def test_add_gaussian_noise_variance(self):
        """Test that added noise has approximately correct variance."""
        x = torch.zeros(1000, 1, 28, 28)  # Large tensor for statistics
        sigma = 0.3
        
        noisy_x = add_gaussian_noise(x, sigma)
        noise = noisy_x - x
        
        # Check that noise std is approximately sigma
        noise_std = torch.std(noise).item()
        assert abs(noise_std - sigma) < 0.05  # Within 5% tolerance
    
    def test_clipping_range(self):
        """Test that clipping range is respected."""
        x = torch.ones(10, 1, 4, 4)  # All ones
        sigma = 2.0  # Large noise
        clip_range = (0, 1)
        
        noisy_x = add_gaussian_noise(x, sigma, clip_range=clip_range)
        
        assert torch.all(noisy_x >= clip_range[0])
        assert torch.all(noisy_x <= clip_range[1])


class TestSigmaSchedules:
    """Test noise schedule generation."""
    
    def test_linear_schedule(self):
        """Test linear schedule properties."""
        schedule = sigma_schedule('linear', 5, 0.0, 1.0)
        
        assert len(schedule) == 5
        assert schedule[0] == 0.0
        assert schedule[-1] == 1.0
        
        # Check monotonic increase
        for i in range(1, len(schedule)):
            assert schedule[i] > schedule[i-1]
    
    def test_cosine_schedule(self):
        """Test cosine schedule properties."""
        schedule = sigma_schedule('cosine', 5, 0.0, 1.0)
        
        assert len(schedule) == 5
        assert schedule[0] == 0.0
        assert abs(schedule[-1] - 1.0) < 1e-6  # Close to 1.0
        
        # Check monotonic increase
        for i in range(1, len(schedule)):
            assert schedule[i] > schedule[i-1]
    
    def test_invalid_schedule(self):
        """Test that invalid schedule raises error."""
        with pytest.raises(ValueError):
            sigma_schedule('invalid', 5, 0.0, 1.0)


class TestTrainTestMismatch:
    """Test train/test sigma mismatch functionality."""
    
    def test_get_train_test_mismatch_sigmas(self):
        """Test train/test sigma generation."""
        train_sigmas, test_sigmas = get_train_test_mismatch_sigmas(
            train_range=(0.1, 0.5),
            test_range=(0.1, 1.0),
            num_train=4,
            num_test=6
        )
        
        assert len(train_sigmas) == 4
        assert len(test_sigmas) == 6
        assert min(train_sigmas) >= 0.1
        assert max(train_sigmas) <= 0.5
        assert min(test_sigmas) >= 0.1
        assert max(test_sigmas) <= 1.0
        
        # Test should have wider range than train
        assert max(test_sigmas) > max(train_sigmas)


class TestBatchNoise:
    """Test batch noise functionality."""
    
    def test_batch_add_different_noise(self):
        """Test adding different noise levels to batch."""
        batch = torch.randn(4, 1, 8, 8)
        sigmas = [0.1, 0.2, 0.3, 0.4]
        
        noisy_batch = batch_add_different_noise(batch, sigmas)
        
        assert noisy_batch.shape == batch.shape
        
        # Test that different samples have different noise levels
        # (This is statistical, so we can't test exact values)
        noise1 = noisy_batch[0] - batch[0]
        noise2 = noisy_batch[1] - batch[1]
        
        # Different noise levels should give different variances
        var1 = torch.var(noise1).item()
        var2 = torch.var(noise2).item()
        assert abs(var1 - 0.1**2) < abs(var1 - 0.2**2)  # noise1 should be closer to sigma1^2
    
    def test_batch_noise_cycling(self):
        """Test that sigmas cycle when batch is larger."""
        batch = torch.randn(6, 1, 4, 4)
        sigmas = [0.1, 0.2]  # Only 2 sigmas for 6 samples
        
        noisy_batch = batch_add_different_noise(batch, sigmas)
        
        assert noisy_batch.shape == batch.shape


class TestNoiseStatistics:
    """Test noise statistics computation."""
    
    def test_compute_noise_statistics(self):
        """Test noise statistics computation."""
        clean = torch.randn(5, 1, 8, 8)
        
        # Add known noise
        noise_std = 0.2
        noise = torch.randn_like(clean) * noise_std
        noisy = clean + noise
        
        stats = compute_noise_statistics(clean, noisy)
        
        # Check that statistics are reasonable
        assert abs(stats['noise_std'] - noise_std) < 0.1
        assert abs(stats['noise_mean']) < 0.1  # Should be close to zero
        assert stats['empirical_snr_db'] > 0  # Should be positive for reasonable SNR


class TestAdaptiveNoiseScheduler:
    """Test adaptive noise scheduler."""
    
    def test_adaptive_scheduler_curriculum(self):
        """Test curriculum learning scheduler."""
        initial_sigmas = [0.1, 0.2, 0.3, 0.4]
        scheduler = AdaptiveNoiseScheduler(
            initial_sigmas,
            adaptation_type="curriculum",
            curriculum_epochs=10
        )
        
        # Test progression
        early_sigmas = scheduler.step(2)
        late_sigmas = scheduler.step(8)
        
        # Later epochs should have higher noise levels
        assert max(late_sigmas) > max(early_sigmas)
    
    def test_adaptive_scheduler_fixed(self):
        """Test fixed scheduler."""
        initial_sigmas = [0.1, 0.2, 0.3]
        scheduler = AdaptiveNoiseScheduler(
            initial_sigmas,
            adaptation_type="fixed"
        )
        
        sigmas1 = scheduler.step(1)
        sigmas2 = scheduler.step(10)
        
        # Should be identical for fixed scheduler
        assert sigmas1 == sigmas2 == initial_sigmas
    
    def test_get_current_sigmas(self):
        """Test getting current sigmas."""
        initial_sigmas = [0.1, 0.2]
        scheduler = AdaptiveNoiseScheduler(initial_sigmas)
        
        current = scheduler.get_current_sigmas()
        assert current == initial_sigmas


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_tensor(self):
        """Test behavior with empty tensor."""
        x = torch.empty(0, 1, 4, 4)
        sigma = 0.1
        
        noisy_x = add_gaussian_noise(x, sigma)
        assert noisy_x.shape == x.shape
    
    def test_single_pixel_image(self):
        """Test with minimal image size."""
        x = torch.randn(1, 1, 1, 1)
        sigma = 0.5
        
        noisy_x = add_gaussian_noise(x, sigma)
        assert noisy_x.shape == x.shape
    
    def test_very_large_sigma(self):
        """Test with very large noise level."""
        x = torch.randn(2, 1, 4, 4)
        sigma = 10.0  # Very large noise
        
        noisy_x = add_gaussian_noise(x, sigma, clip_range=(0, 1))
        
        # Should still respect clipping
        assert torch.all(noisy_x >= 0)
        assert torch.all(noisy_x <= 1)
    
    def test_negative_sigma(self):
        """Test that negative sigma works (just inverts noise)."""
        x = torch.randn(2, 1, 4, 4)
        sigma = -0.1
        
        # Should not raise error, just apply negative noise
        noisy_x = add_gaussian_noise(x, sigma)
        assert noisy_x.shape == x.shape


if __name__ == '__main__':
    pytest.main([__file__])