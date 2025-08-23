"""Tests for visualization functions."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

from src.ddpm_schedules import get_ddpm_schedule
from src.visualize import (
    plot_schedules, create_trajectory_grid, create_trajectory_animation,
    plot_pixel_histograms, plot_mse_and_kl_curves
)
from src.stats import compute_forward_stats


class TestVisualizationSmokeTests:
    """Smoke tests to ensure visualization functions don't crash."""
    
    @pytest.fixture
    def setup_vis_data(self):
        """Setup test data for visualization."""
        torch.manual_seed(42)
        x0 = torch.randn(4, 1, 16, 16)  # Small test images
        betas, alphas, alpha_bars = get_ddpm_schedule(50, "cosine")  # Short schedule for speed
        return x0, betas, alphas, alpha_bars
    
    def test_plot_schedules_smoke(self, setup_vis_data):
        """Test that plot_schedules runs without error."""
        x0, betas, alphas, alpha_bars = setup_vis_data
        
        # Create mock schedules data
        schedules_data = {
            'cosine': {
                'timesteps': torch.arange(len(betas)),
                'betas': betas,
                'alpha_bars': alpha_bars,
                'snr_db': 10.0 * torch.log10(alpha_bars / (1.0 - alpha_bars + 1e-8))
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = Path(tmp.name)
        
        try:
            # Should not raise any exceptions
            plot_schedules(schedules_data, save_path=temp_path)
            
            # Check that file was created
            assert temp_path.exists(), "Plot file should be created"
            assert temp_path.stat().st_size > 0, "Plot file should not be empty"
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def test_trajectory_grid_smoke(self, setup_vis_data):
        """Test that create_trajectory_grid runs without error."""
        x0, betas, alphas, alpha_bars = setup_vis_data
        
        timesteps_to_show = [0, 10, 25, 49]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = Path(tmp.name)
        
        try:
            # Should not raise any exceptions
            trajectory = create_trajectory_grid(
                x0,
                timesteps_to_show,
                alpha_bars,
                save_path=temp_path,
                normalize=True
            )
            
            # Check output shape
            expected_shape = (x0.shape[0], len(timesteps_to_show)) + x0.shape[1:]
            assert trajectory.shape == expected_shape, f"Expected {expected_shape}, got {trajectory.shape}"
            
            # Check that file was created
            assert temp_path.exists(), "Grid file should be created"
            assert temp_path.stat().st_size > 0, "Grid file should not be empty"
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def test_trajectory_animation_smoke(self, setup_vis_data):
        """Test that create_trajectory_animation runs without error."""
        x0, betas, alphas, alpha_bars = setup_vis_data
        
        # Use just one image
        x0_single = x0[:1]
        T = len(betas)
        
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as tmp:
            temp_path = Path(tmp.name)
        
        try:
            # Should not raise any exceptions
            trajectory = create_trajectory_animation(
                x0_single,
                T,
                alpha_bars,
                save_path=temp_path,
                normalize=True,
                duration=0.1
            )
            
            # Check output shape: (1, T+1, C, H, W)
            expected_shape = (1, T + 1) + x0_single.shape[1:]
            assert trajectory.shape == expected_shape, f"Expected {expected_shape}, got {trajectory.shape}"
            
            # Check that file was created
            assert temp_path.exists(), "Animation file should be created"
            assert temp_path.stat().st_size > 0, "Animation file should not be empty"
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def test_pixel_histograms_smoke(self, setup_vis_data):
        """Test that plot_pixel_histograms runs without error."""
        x0, betas, alphas, alpha_bars = setup_vis_data
        
        timesteps_to_show = [0, 25, 49]
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = Path(tmp.name)
        
        try:
            # Should not raise any exceptions
            plot_pixel_histograms(
                x0,
                timesteps_to_show,
                alpha_bars,
                save_path=temp_path
            )
            
            # Check that file was created
            assert temp_path.exists(), "Histogram file should be created"
            assert temp_path.stat().st_size > 0, "Histogram file should not be empty"
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()
    
    def test_mse_kl_curves_smoke(self, setup_vis_data):
        """Test that plot_mse_and_kl_curves runs without error."""
        x0, betas, alphas, alpha_bars = setup_vis_data
        
        # Compute minimal stats for testing
        device = torch.device("cpu")
        stats = compute_forward_stats(x0, betas, alpha_bars, device)
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            temp_path = Path(tmp.name)
        
        try:
            # Should not raise any exceptions
            plot_mse_and_kl_curves(stats, save_path=temp_path)
            
            # Check that file was created
            assert temp_path.exists(), "MSE/KL curves file should be created"
            assert temp_path.stat().st_size > 0, "MSE/KL curves file should not be empty"
            
        finally:
            # Clean up
            if temp_path.exists():
                temp_path.unlink()


class TestVisualizationOutputs:
    """Test visualization output correctness."""
    
    def test_trajectory_grid_timestep_consistency(self):
        """Test that trajectory grid shows correct timesteps."""
        torch.manual_seed(42)
        x0 = torch.randn(2, 1, 8, 8)
        betas, alphas, alpha_bars = get_ddpm_schedule(100, "linear")
        
        timesteps_to_show = [0, 50, 99]
        
        trajectory = create_trajectory_grid(
            x0,
            timesteps_to_show,
            alpha_bars,
            save_path=None,  # Don't save during test
            normalize=False
        )
        
        # First timestep should be original
        assert torch.allclose(trajectory[:, 0], x0), "First timestep should be x_0"
        
        # Last timestep should be very noisy (different from original)
        final_mse = torch.mean((trajectory[:, -1] - x0) ** 2)
        assert final_mse > 0.5, "Final timestep should be significantly different from x_0"
    
    def test_trajectory_animation_length(self):
        """Test that animation has correct number of frames."""
        torch.manual_seed(42)
        x0 = torch.randn(1, 1, 8, 8)
        T = 20
        betas, alphas, alpha_bars = get_ddpm_schedule(T, "linear")
        
        trajectory = create_trajectory_animation(
            x0,
            T,
            alpha_bars,
            save_path=None,  # Don't save during test
            normalize=False
        )
        
        # Should have T+1 frames (including x_0)
        assert trajectory.shape[1] == T + 1, f"Expected {T+1} frames, got {trajectory.shape[1]}"


if __name__ == "__main__":
    pytest.main([__file__])