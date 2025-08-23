"""Tests for noise schedules."""

import pytest
import torch
import numpy as np

from src.ddpm_schedules import (
    make_beta_schedule, compute_alpha_schedule, get_ddpm_schedule,
    get_schedule_stats
)


class TestBetaSchedules:
    """Test beta schedule generation."""
    
    def test_linear_schedule_monotonic(self):
        """Test that linear schedule is monotonically increasing."""
        betas = make_beta_schedule(1000, "linear")
        
        # Check monotonicity
        assert torch.all(betas[1:] >= betas[:-1]), "Linear schedule should be monotonic"
        
        # Check range
        assert torch.all(betas > 0), "All betas should be positive"
        assert torch.all(betas < 1), "All betas should be less than 1"
    
    def test_cosine_schedule_properties(self):
        """Test cosine schedule properties."""
        betas = make_beta_schedule(1000, "cosine")
        
        # Check range
        assert torch.all(betas > 0), "All betas should be positive"
        assert torch.all(betas < 1), "All betas should be less than 1"
        
        # Check that it starts small and ends large
        assert betas[0] < betas[-1], "Cosine schedule should increase overall"
        assert betas[0] < 0.01, "Cosine schedule should start small"
    
    def test_sigmoid_schedule_properties(self):
        """Test sigmoid schedule properties."""
        betas = make_beta_schedule(1000, "sigmoid")
        
        # Check range
        assert torch.all(betas > 0), "All betas should be positive"
        assert torch.all(betas < 1), "All betas should be less than 1"
        
        # Check S-curve property (derivative should first increase then decrease)
        diff = betas[1:] - betas[:-1]
        assert torch.max(diff) > torch.mean(diff), "Should have steepest part in middle"
    
    def test_schedule_clamping(self):
        """Test that schedules are properly clamped."""
        for schedule_type in ["linear", "cosine", "sigmoid"]:
            betas = make_beta_schedule(1000, schedule_type)
            
            assert torch.all(betas >= 1e-6), f"{schedule_type} betas below minimum"
            assert torch.all(betas <= 0.999), f"{schedule_type} betas above maximum"


class TestAlphaSchedules:
    """Test alpha and alpha_bar computation."""
    
    def test_alpha_computation(self):
        """Test alpha = 1 - beta."""
        betas = make_beta_schedule(100, "linear")
        alphas, alpha_bars = compute_alpha_schedule(betas)
        
        expected_alphas = 1.0 - betas
        assert torch.allclose(alphas, expected_alphas), "Alphas should be 1 - betas"
    
    def test_alpha_bar_computation(self):
        """Test alpha_bar cumulative product."""
        betas = make_beta_schedule(100, "linear")
        alphas, alpha_bars = compute_alpha_schedule(betas)
        
        expected_alpha_bars = torch.cumprod(alphas, dim=0)
        assert torch.allclose(alpha_bars, expected_alpha_bars), "Alpha_bars should be cumulative product"
    
    def test_alpha_bar_monotonic_decreasing(self):
        """Test that alpha_bar is monotonically decreasing."""
        for schedule_type in ["linear", "cosine", "sigmoid"]:
            betas, alphas, alpha_bars = get_ddpm_schedule(1000, schedule_type)
            
            # Should be monotonically decreasing
            assert torch.all(alpha_bars[1:] <= alpha_bars[:-1]), \
                f"{schedule_type} alpha_bars should be monotonically decreasing"
    
    def test_boundary_conditions(self):
        """Test boundary conditions for alpha_bar."""
        for schedule_type in ["linear", "cosine", "sigmoid"]:
            betas, alphas, alpha_bars = get_ddpm_schedule(1000, schedule_type)
            
            # At t=0, alpha_bar should be close to 1 (actually alpha_bars[0] = alphas[0])
            assert alpha_bars[0] > 0.9, f"{schedule_type} should start with high alpha_bar"
            
            # At final timestep, alpha_bar should be small
            assert alpha_bars[-1] < 0.1, f"{schedule_type} should end with small alpha_bar"
    
    def test_alpha_bar_range(self):
        """Test that alpha_bar stays in valid range."""
        for schedule_type in ["linear", "cosine", "sigmoid"]:
            betas, alphas, alpha_bars = get_ddpm_schedule(1000, schedule_type)
            
            assert torch.all(alpha_bars > 0), f"{schedule_type} alpha_bars should be positive"
            assert torch.all(alpha_bars <= 1), f"{schedule_type} alpha_bars should be <= 1"


class TestScheduleStats:
    """Test schedule statistics computation."""
    
    def test_schedule_stats_structure(self):
        """Test that schedule stats have correct structure."""
        betas, alphas, alpha_bars = get_ddpm_schedule(100, "linear")
        stats = get_schedule_stats(betas, alphas, alpha_bars)
        
        required_keys = ['T', 'beta_min', 'beta_max', 'alpha_bar_min', 'alpha_bar_max', 'alpha_bar_final']
        for key in required_keys:
            assert key in stats, f"Missing key: {key}"
    
    def test_schedule_stats_values(self):
        """Test that schedule stats have reasonable values."""
        betas, alphas, alpha_bars = get_ddpm_schedule(1000, "cosine")
        stats = get_schedule_stats(betas, alphas, alpha_bars)
        
        assert stats['T'] == 1000, "T should match input"
        assert stats['beta_min'] > 0, "Beta min should be positive"
        assert stats['beta_max'] < 1, "Beta max should be less than 1"
        assert stats['alpha_bar_min'] > 0, "Alpha bar min should be positive"
        assert stats['alpha_bar_max'] <= 1, "Alpha bar max should be <= 1"
        assert stats['alpha_bar_final'] == stats['alpha_bar_min'], "Final should be minimum"


if __name__ == "__main__":
    pytest.main([__file__])