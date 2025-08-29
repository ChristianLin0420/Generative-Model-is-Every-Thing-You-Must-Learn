"""
Test beta schedule shapes and validity: β∈(0,1), ᾱ monotone↓, shapes OK for all schedules.
"""

import pytest
import torch
import numpy as np

from src.schedules import (
    beta_linear, beta_cosine, beta_quadratic, 
    get_schedule, validate_schedule
)


class TestScheduleShapes:
    """Test schedule tensor shapes and basic properties."""
    
    @pytest.mark.parametrize("T", [10, 100, 1000])
    def test_linear_schedule_shapes(self, T):
        """Test linear schedule produces correct shapes."""
        schedule = beta_linear(T)
        
        assert 'betas' in schedule
        assert 'alphas' in schedule
        assert 'alpha_bars' in schedule
        assert 'snr' in schedule
        
        for key, tensor in schedule.items():
            assert tensor.shape == (T,), f"{key} has wrong shape: {tensor.shape}"
            assert tensor.dtype == torch.float32
    
    @pytest.mark.parametrize("T", [10, 100, 1000])
    def test_cosine_schedule_shapes(self, T):
        """Test cosine schedule produces correct shapes."""
        schedule = beta_cosine(T)
        
        for key, tensor in schedule.items():
            assert tensor.shape == (T,), f"{key} has wrong shape: {tensor.shape}"
            assert tensor.dtype == torch.float32
    
    @pytest.mark.parametrize("T", [10, 100, 1000])
    def test_quadratic_schedule_shapes(self, T):
        """Test quadratic schedule produces correct shapes."""
        schedule = beta_quadratic(T)
        
        for key, tensor in schedule.items():
            assert tensor.shape == (T,), f"{key} has wrong shape: {tensor.shape}"
            assert tensor.dtype == torch.float32


class TestScheduleValidity:
    """Test schedule validity constraints."""
    
    @pytest.mark.parametrize("schedule_name", ["linear", "cosine", "quadratic"])
    @pytest.mark.parametrize("T", [10, 100, 1000])
    def test_beta_range(self, schedule_name, T):
        """Test that β ∈ (0, 1) for all schedules."""
        schedule = get_schedule(schedule_name, T)
        betas = schedule['betas']
        
        assert torch.all(betas > 0), f"Some betas <= 0 in {schedule_name}"
        assert torch.all(betas < 1), f"Some betas >= 1 in {schedule_name}"
    
    @pytest.mark.parametrize("schedule_name", ["linear", "cosine", "quadratic"])
    @pytest.mark.parametrize("T", [10, 100, 1000])
    def test_alpha_bar_monotonic(self, schedule_name, T):
        """Test that ᾱ is monotonically decreasing."""
        schedule = get_schedule(schedule_name, T)
        alpha_bars = schedule['alpha_bars']
        
        # Check monotonic decrease
        diffs = alpha_bars[1:] - alpha_bars[:-1]
        assert torch.all(diffs <= 0), f"α_bar not monotonic in {schedule_name}"
        
        # Check range
        assert torch.all(alpha_bars > 0), f"Some α_bar <= 0 in {schedule_name}"
        assert torch.all(alpha_bars <= 1), f"Some α_bar > 1 in {schedule_name}"
    
    @pytest.mark.parametrize("schedule_name", ["linear", "cosine", "quadratic"])
    def test_alpha_relationship(self, schedule_name):
        """Test that α = 1 - β."""
        T = 100
        schedule = get_schedule(schedule_name, T)
        
        betas = schedule['betas']
        alphas = schedule['alphas']
        
        expected_alphas = 1.0 - betas
        assert torch.allclose(alphas, expected_alphas, atol=1e-6)
    
    @pytest.mark.parametrize("schedule_name", ["linear", "cosine", "quadratic"])
    def test_alpha_bar_relationship(self, schedule_name):
        """Test that ᾱ = ∏α."""
        T = 100
        schedule = get_schedule(schedule_name, T)
        
        alphas = schedule['alphas']
        alpha_bars = schedule['alpha_bars']
        
        expected_alpha_bars = torch.cumprod(alphas, dim=0)
        assert torch.allclose(alpha_bars, expected_alpha_bars, atol=1e-6)
    
    @pytest.mark.parametrize("schedule_name", ["linear", "cosine", "quadratic"])
    def test_snr_relationship(self, schedule_name):
        """Test that SNR = ᾱ / (1 - ᾱ)."""
        T = 100
        schedule = get_schedule(schedule_name, T)
        
        alpha_bars = schedule['alpha_bars']
        snr = schedule['snr']
        
        expected_snr = alpha_bars / (1.0 - alpha_bars)
        assert torch.allclose(snr, expected_snr, atol=1e-6)


class TestScheduleComparison:
    """Test relative properties between different schedules."""
    
    def test_schedule_differences(self):
        """Test that different schedules produce different values."""
        T = 1000
        
        linear_schedule = get_schedule("linear", T)
        cosine_schedule = get_schedule("cosine", T)
        quad_schedule = get_schedule("quadratic", T)
        
        # Schedules should be different
        assert not torch.allclose(linear_schedule['betas'], cosine_schedule['betas'])
        assert not torch.allclose(linear_schedule['betas'], quad_schedule['betas'])
        assert not torch.allclose(cosine_schedule['betas'], quad_schedule['betas'])
    
    def test_schedule_endpoints(self):
        """Test schedule behavior at endpoints."""
        T = 1000
        
        for schedule_name in ["linear", "cosine", "quadratic"]:
            schedule = get_schedule(schedule_name, T)
            
            # First alpha_bar should be close to 1
            assert schedule['alpha_bars'][0] > 0.9, f"{schedule_name} α_bar[0] too small"
            
            # Last alpha_bar should be small
            assert schedule['alpha_bars'][-1] < 0.1, f"{schedule_name} α_bar[-1] too large"
            
            # SNR should start high and end low
            assert schedule['snr'][0] > schedule['snr'][-1], f"{schedule_name} SNR not decreasing"


class TestScheduleParameters:
    """Test schedule parameter handling."""
    
    def test_linear_parameters(self):
        """Test linear schedule with different parameters."""
        T = 100
        
        # Test default parameters
        schedule1 = beta_linear(T)
        schedule2 = beta_linear(T, beta_min=1e-4, beta_max=0.02)
        assert torch.allclose(schedule1['betas'], schedule2['betas'])
        
        # Test different parameters
        schedule3 = beta_linear(T, beta_min=1e-3, beta_max=0.01)
        assert not torch.allclose(schedule1['betas'], schedule3['betas'])
        
        # Check parameter effects
        assert schedule3['betas'][0] > schedule1['betas'][0]  # Higher min
        assert schedule3['betas'][-1] < schedule1['betas'][-1]  # Lower max
    
    def test_cosine_parameters(self):
        """Test cosine schedule with different s parameter."""
        T = 100
        
        schedule1 = beta_cosine(T, s=0.008)
        schedule2 = beta_cosine(T, s=0.01)
        
        assert not torch.allclose(schedule1['betas'], schedule2['betas'])
    
    def test_quadratic_parameters(self):
        """Test quadratic schedule with different parameters."""
        T = 100
        
        schedule1 = beta_quadratic(T, beta_min=1e-4, beta_max=0.02)
        schedule2 = beta_quadratic(T, beta_min=1e-3, beta_max=0.01)
        
        assert not torch.allclose(schedule1['betas'], schedule2['betas'])


class TestScheduleValidation:
    """Test schedule validation function."""
    
    def test_validate_good_schedule(self):
        """Test validation passes for good schedules."""
        for schedule_name in ["linear", "cosine", "quadratic"]:
            schedule = get_schedule(schedule_name, 100)
            assert validate_schedule(schedule), f"Valid {schedule_name} schedule failed validation"
    
    def test_validate_bad_beta_range(self):
        """Test validation fails for bad beta range."""
        schedule = beta_linear(100)
        
        # Set some betas out of range
        schedule['betas'][0] = -0.1  # Negative beta
        assert not validate_schedule(schedule)
        
        schedule = beta_linear(100)
        schedule['betas'][-1] = 1.5  # Beta > 1
        assert not validate_schedule(schedule)
    
    def test_validate_non_monotonic(self):
        """Test validation fails for non-monotonic alpha_bars."""
        schedule = beta_linear(100)
        
        # Make alpha_bars non-monotonic
        schedule['alpha_bars'][50] = schedule['alpha_bars'][49] + 0.1
        assert not validate_schedule(schedule)


class TestScheduleEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_small_T(self):
        """Test schedules work with small T."""
        for schedule_name in ["linear", "cosine", "quadratic"]:
            schedule = get_schedule(schedule_name, T=1)
            assert validate_schedule(schedule)
            
            schedule = get_schedule(schedule_name, T=2)
            assert validate_schedule(schedule)
    
    def test_invalid_schedule_name(self):
        """Test error for invalid schedule name."""
        with pytest.raises(ValueError):
            get_schedule("invalid_schedule", 100)
    
    def test_zero_T(self):
        """Test error for T=0."""
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            beta_linear(0)
            
        # Also test that T=0 raises an error for other schedules
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            beta_cosine(0)
            
        with pytest.raises((ValueError, RuntimeError, AssertionError)):
            beta_quadratic(0)


if __name__ == "__main__":
    # Run tests manually
    import sys
    
    print("Running schedule shape and validity tests...")
    
    # Test basic functionality
    for T in [10, 100, 1000]:
        for schedule_name in ["linear", "cosine", "quadratic"]:
            schedule = get_schedule(schedule_name, T)
            assert validate_schedule(schedule), f"Failed: {schedule_name} T={T}"
    
    print("✓ All basic tests passed")
    
    # Test some specific properties
    schedule = get_schedule("linear", 1000)
    assert torch.all(schedule['betas'] > 0)
    assert torch.all(schedule['betas'] < 1)
    assert torch.all(schedule['alpha_bars'][1:] <= schedule['alpha_bars'][:-1])
    
    print("✓ Validation tests passed")
    print("All schedule tests completed successfully!")
