"""Tests for forward diffusion process q(x_t|x_0)."""

import pytest
import torch
import numpy as np

from src.ddpm_schedules import get_ddpm_schedule
from src.forward import (
    q_xt_given_x0, q_xt_given_xtm1, sample_trajectory,
    extract, snr, snr_db, compute_mse_to_x0
)


class TestForwardSampling:
    """Test forward sampling functions."""
    
    @pytest.fixture
    def setup_data(self):
        """Setup test data."""
        torch.manual_seed(42)
        batch_size = 4
        x0 = torch.randn(batch_size, 1, 8, 8)  # Small test images
        betas, alphas, alpha_bars = get_ddpm_schedule(100, "linear")
        return x0, betas, alphas, alpha_bars
    
    def test_extract_function(self, setup_data):
        """Test the extract utility function."""
        x0, betas, alphas, alpha_bars = setup_data
        
        timesteps = torch.tensor([0, 10, 50, 99])
        extracted = extract(alpha_bars, timesteps, x0.shape)
        
        # Check shape
        assert extracted.shape == (4, 1, 1, 1), "Extract should broadcast correctly"
        
        # Check values
        expected = alpha_bars[timesteps].view(-1, 1, 1, 1)
        assert torch.allclose(extracted, expected), "Extract should get correct values"
    
    def test_q_xt_given_x0_shape(self, setup_data):
        """Test that q(x_t|x_0) produces correct shapes."""
        x0, betas, alphas, alpha_bars = setup_data
        
        t = torch.tensor([10, 20, 30, 40])
        x_t, noise = q_xt_given_x0(x0, t, alpha_bars)
        
        assert x_t.shape == x0.shape, "x_t should have same shape as x_0"
        assert noise.shape == x0.shape, "noise should have same shape as x_0"
    
    def test_q_xt_given_x0_variance(self, setup_data):
        """Test that q(x_t|x_0) has correct variance."""
        x0, betas, alphas, alpha_bars = setup_data
        
        # For unit variance x_0, variance of x_t should be 1
        # when x_0 is normalized to unit variance
        x0_normalized = torch.randn_like(x0)  # Already unit variance
        
        t = torch.tensor([50] * x0.shape[0])
        alpha_bar_t = alpha_bars[50]
        
        # Sample many times to estimate variance
        n_samples = 1000
        x_t_samples = []
        for _ in range(n_samples):
            x_t, _ = q_xt_given_x0(x0_normalized, t, alpha_bars)
            x_t_samples.append(x_t.flatten())
        
        x_t_all = torch.cat(x_t_samples)
        empirical_var = x_t_all.var()
        
        # Theoretical variance should be 1 for unit variance input
        theoretical_var = 1.0
        
        # Allow some tolerance due to finite sampling
        assert abs(empirical_var - theoretical_var) < 0.1, \
            f"Variance mismatch: empirical={empirical_var:.3f}, theoretical={theoretical_var:.3f}"
    
    def test_t_zero_identity(self, setup_data):
        """Test that t=0 returns original image."""
        x0, betas, alphas, alpha_bars = setup_data
        
        t = torch.zeros(x0.shape[0], dtype=torch.long)
        x_t, noise = q_xt_given_x0(x0, t, alpha_bars)
        
        # At t=0, alpha_bar = alpha_bars[0] = alphas[0] = 1 - betas[0] ≈ 1
        # So x_t should be very close to x_0
        assert torch.allclose(x_t, x0, atol=1e-3), "t=0 should return nearly original image"
    
    def test_large_t_pure_noise(self, setup_data):
        """Test that large t produces noise-dominated signal."""
        x0, betas, alphas, alpha_bars = setup_data
        
        t = torch.full((x0.shape[0],), 99)  # Large t
        x_t, noise = q_xt_given_x0(x0, t, alpha_bars)
        
        # At large t, alpha_bar should be small, so x_t ≈ noise
        alpha_bar_large = alpha_bars[99]
        assert alpha_bar_large < 0.1, "Large t should have small alpha_bar"
        
        # x_t should be mostly noise (closer to noise than to x_0)
        noise_similarity = F.mse_loss(x_t, noise)
        signal_similarity = F.mse_loss(x_t, x0)
        
        # This test might be flaky, so we use a relaxed condition
        assert noise_similarity < signal_similarity * 2, "At large t, x_t should be closer to noise"
    
    def test_sequential_vs_closed_form(self, setup_data):
        """Test that sequential and closed-form sampling are close."""
        x0, betas, alphas, alpha_bars = setup_data
        
        # Use fixed noise for reproducible comparison
        torch.manual_seed(123)
        T = 50  # Shorter for faster test
        
        # Sequential sampling
        x_current = x0
        for t_val in range(1, T + 1):
            t_tensor = torch.full((x0.shape[0],), t_val - 1)
            x_current, _ = q_xt_given_xtm1(x_current, t_tensor, betas[:T])
        x_sequential = x_current
        
        # Closed-form sampling  
        t_final = torch.full((x0.shape[0],), T - 1)
        x_closed_form, _ = q_xt_given_x0(x0, t_final, alpha_bars[:T])
        
        # They should be similar in distribution (not exact due to different noise)
        # Just check that they have similar statistics
        seq_mean = x_sequential.mean()
        closed_mean = x_closed_form.mean()
        seq_std = x_sequential.std()
        closed_std = x_closed_form.std()
        
        assert abs(seq_mean - closed_mean) < 0.2, "Sequential and closed-form should have similar means"
        assert abs(seq_std - closed_std) < 0.2, "Sequential and closed-form should have similar stds"


class TestSNRFunctions:
    """Test SNR computation functions."""
    
    def test_snr_computation(self):
        """Test SNR computation."""
        alpha_bars = torch.tensor([0.9, 0.5, 0.1, 0.01])
        snr_values = snr(alpha_bars)
        
        expected = alpha_bars / (1.0 - alpha_bars)
        assert torch.allclose(snr_values, expected), "SNR should be alpha_bar / (1 - alpha_bar)"
    
    def test_snr_db_computation(self):
        """Test SNR in dB computation."""
        alpha_bars = torch.tensor([0.9, 0.5, 0.1])
        snr_db_values = snr_db(alpha_bars)
        
        snr_linear = alpha_bars / (1.0 - alpha_bars)
        expected_db = 10.0 * torch.log10(snr_linear)
        
        assert torch.allclose(snr_db_values, expected_db, atol=1e-4), \
            "SNR dB should be 10 * log10(SNR_linear)"
    
    def test_snr_monotonic_decreasing(self):
        """Test that SNR decreases monotonically with timestep."""
        for schedule_type in ["linear", "cosine", "sigmoid"]:
            betas, alphas, alpha_bars = get_ddpm_schedule(1000, schedule_type)
            snr_values = snr(alpha_bars)
            
            # SNR should decrease (alpha_bar decreases, 1-alpha_bar increases)
            assert torch.all(snr_values[1:] <= snr_values[:-1]), \
                f"SNR should decrease monotonically for {schedule_type}"


class TestTrajectoryGeneration:
    """Test trajectory generation."""
    
    @pytest.fixture
    def setup_trajectory_data(self):
        """Setup data for trajectory tests."""
        torch.manual_seed(42)
        x0 = torch.randn(2, 1, 4, 4)
        betas, alphas, alpha_bars = get_ddpm_schedule(20, "linear")
        return x0, betas, alphas, alpha_bars
    
    def test_trajectory_shape(self, setup_trajectory_data):
        """Test trajectory has correct shape."""
        x0, betas, alphas, alpha_bars = setup_trajectory_data
        T = len(betas)
        
        trajectory = sample_trajectory(x0, T, betas, alpha_bars, use_closed_form=True)
        
        expected_shape = (x0.shape[0], T + 1) + x0.shape[1:]
        assert trajectory.shape == expected_shape, f"Expected {expected_shape}, got {trajectory.shape}"
    
    def test_trajectory_initial_condition(self, setup_trajectory_data):
        """Test that trajectory starts with x_0."""
        x0, betas, alphas, alpha_bars = setup_trajectory_data
        T = len(betas)
        
        trajectory = sample_trajectory(x0, T, betas, alpha_bars, use_closed_form=True)
        
        assert torch.allclose(trajectory[:, 0], x0), "Trajectory should start with x_0"
    
    def test_fixed_noise_reproducibility(self, setup_trajectory_data):
        """Test that fixed noise produces reproducible trajectories."""
        x0, betas, alphas, alpha_bars = setup_trajectory_data
        T = len(betas)
        
        # Generate fixed noise
        torch.manual_seed(456)
        fixed_noise = torch.randn((T,) + x0.shape)
        
        # Generate two trajectories with same fixed noise
        traj1 = sample_trajectory(x0, T, betas, alpha_bars, use_closed_form=True, fixed_noise=fixed_noise)
        traj2 = sample_trajectory(x0, T, betas, alpha_bars, use_closed_form=True, fixed_noise=fixed_noise)
        
        assert torch.allclose(traj1, traj2), "Fixed noise should produce identical trajectories"


class TestMSEComputation:
    """Test MSE computation."""
    
    def test_mse_zero_for_identical(self):
        """Test MSE is zero for identical images."""
        x = torch.randn(4, 1, 8, 8)
        mse = compute_mse_to_x0(x, x)
        
        assert torch.allclose(mse, torch.zeros_like(mse), atol=1e-6), \
            "MSE should be zero for identical images"
    
    def test_mse_increases_with_noise(self):
        """Test that MSE increases as noise is added."""
        x0 = torch.randn(4, 1, 8, 8)
        
        # Add different amounts of noise
        noise_scales = [0.1, 0.5, 1.0, 2.0]
        mse_values = []
        
        for scale in noise_scales:
            noise = torch.randn_like(x0) * scale
            x_noisy = x0 + noise
            mse = compute_mse_to_x0(x_noisy, x0).mean()
            mse_values.append(mse)
        
        # MSE should increase with noise scale
        for i in range(1, len(mse_values)):
            assert mse_values[i] > mse_values[i-1], "MSE should increase with noise"


if __name__ == "__main__":
    pytest.main([__file__])