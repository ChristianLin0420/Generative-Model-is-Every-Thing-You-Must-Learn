"""
Tests for noise functionality
"""

import pytest
import torch
import numpy as np

from src.noise import (
    add_gaussian_noise,
    sigma_schedule, 
    NoiseScheduler,
    compute_snr,
    progressive_noise_sequence,
    analyze_noise_impact
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
    
    def test_custom_schedule(self):
        """Test custom (quadratic) schedule."""
        schedule = sigma_schedule('custom', 4, 0.0, 1.0)
        
        assert len(schedule) == 4
        assert schedule[0] == 0.0
        assert schedule[-1] == 1.0
    
    def test_invalid_schedule(self):
        """Test that invalid schedule raises error."""
        with pytest.raises(ValueError):
            sigma_schedule('invalid', 5, 0.0, 1.0)


class TestNoiseScheduler:
    """Test NoiseScheduler class."""
    
    def test_scheduler_initialization(self):
        """Test scheduler creates correct schedule."""
        scheduler = NoiseScheduler('linear', 5, 0.0, 1.0)
        
        assert len(scheduler) == 5
        assert scheduler[0] == 0.0
        assert scheduler[-1] == 1.0
        
        sigmas = scheduler.get_sigmas()
        assert len(sigmas) == 5
    
    def test_scheduler_snr_calculation(self):
        """Test SNR calculation in scheduler."""
        scheduler = NoiseScheduler('linear', 3, 0.1, 0.5)
        snrs = scheduler.get_snrs(data_std=1.0)
        
        assert len(snrs) == 3
        # SNR should decrease as sigma increases
        assert snrs[0] > snrs[1] > snrs[2]


class TestSNRComputation:
    """Test SNR computation functions."""
    
    def test_compute_snr_basic(self):
        """Test basic SNR computation."""
        sigma = 0.1
        data_std = 1.0
        
        snr = compute_snr(sigma, data_std)
        
        # SNR = 10 * log10(data_std^2 / sigma^2)
        expected_snr = 10 * np.log10(1.0 / 0.01)  # 20 dB
        assert abs(snr - expected_snr) < 1e-6
    
    def test_compute_snr_zero_noise(self):
        """Test SNR with zero noise."""
        snr = compute_snr(0.0, 1.0)
        assert snr == float('inf')
    
    def test_compute_snr_different_data_std(self):
        """Test SNR with different data standard deviations."""
        sigma = 0.1
        
        snr1 = compute_snr(sigma, 1.0)
        snr2 = compute_snr(sigma, 2.0)
        
        # Higher data std should give higher SNR
        assert snr2 > snr1


class TestProgressiveNoise:
    """Test progressive noise sequence generation."""
    
    def test_progressive_noise_sequence_shape(self):
        """Test that progressive sequence has correct shape."""
        image = torch.randn(1, 28, 28)  # Single image
        sigmas = [0.0, 0.1, 0.2]
        
        sequence = progressive_noise_sequence(image, sigmas)
        
        assert sequence.shape == (3, 1, 28, 28)
    
    def test_progressive_noise_reproducibility(self):
        """Test reproducibility of progressive sequence."""
        image = torch.randn(1, 28, 28)
        sigmas = [0.0, 0.1, 0.2]
        
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        
        seq1 = progressive_noise_sequence(image, sigmas, generator=gen1)
        seq2 = progressive_noise_sequence(image, sigmas, generator=gen2)
        
        assert torch.allclose(seq1, seq2)


class TestNoiseAnalysis:
    """Test noise impact analysis functions."""
    
    def test_analyze_noise_impact(self):
        """Test noise impact analysis."""
        original = torch.randn(2, 1, 4, 4)
        
        # Add known noise
        noise = torch.randn_like(original) * 0.1
        noisy = original + noise
        
        results = analyze_noise_impact(original, noisy)
        
        # Check that results contain expected keys
        expected_keys = ['mse', 'mae', 'noise_std', 'noise_mean', 
                        'original_std', 'noisy_std', 'snr_empirical']
        for key in expected_keys:
            assert key in results
        
        # Check that MSE > 0 (since we added noise)
        assert results['mse'] > 0
    
    def test_analyze_no_noise(self):
        """Test analysis with identical images (no noise)."""
        original = torch.randn(2, 1, 4, 4)
        
        results = analyze_noise_impact(original, original)
        
        assert results['mse'] == 0.0
        assert results['mae'] == 0.0
        assert results['noise_std'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__])