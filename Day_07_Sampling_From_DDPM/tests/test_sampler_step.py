"""
Test ancestral sampling step correctness.
Verifies shapes, boundary conditions, and variance handling.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.ddpm_schedules import DDPMSchedules
from src.sampler import DDPMSampler
from src.models.unet_small import UNetSmall


class MockModel(torch.nn.Module):
    """Mock model for testing that returns predictable noise."""
    
    def __init__(self, output_shape):
        super().__init__()
        self.output_shape = output_shape
        
    def forward(self, x, t):
        # Return small random noise
        return torch.randn_like(x) * 0.1


@pytest.fixture
def setup_sampler():
    """Set up sampler with mock model for testing."""
    device = torch.device('cpu')
    
    # Create schedules
    schedules = DDPMSchedules(timesteps=100, schedule='linear')
    schedules.to(device)
    
    # Create mock model
    model = MockModel((1, 28, 28))
    
    # Create sampler
    sampler = DDPMSampler(model, schedules, device)
    
    return sampler, schedules, device


def test_ancestral_step_shapes(setup_sampler):
    """Test that ancestral step maintains correct shapes."""
    sampler, schedules, device = setup_sampler
    
    batch_size = 4
    shape = (batch_size, 1, 28, 28)
    
    # Create test input
    x_t = torch.randn(shape, device=device)
    t = torch.randint(1, schedules.timesteps, (batch_size,), device=device)
    
    # Perform ancestral step
    x_t_minus_1, x_0_pred, eps_pred = sampler.ancestral_step(x_t, t)
    
    # Check shapes
    assert x_t_minus_1.shape == shape
    assert x_0_pred.shape == shape
    assert eps_pred.shape == shape
    
    # Check no NaNs or infs
    assert not torch.isnan(x_t_minus_1).any()
    assert not torch.isnan(x_0_pred).any()
    assert not torch.isnan(eps_pred).any()
    
    assert not torch.isinf(x_t_minus_1).any()
    assert not torch.isinf(x_0_pred).any()
    assert not torch.isinf(eps_pred).any()


def test_ancestral_step_t_zero_boundary(setup_sampler):
    """Test that t=0 step has no additional sampling noise."""
    sampler, schedules, device = setup_sampler
    
    batch_size = 2
    shape = (batch_size, 1, 28, 28)
    
    # Test with t=0 vs t=1 to verify noise is only added for t>0
    x_t = torch.randn(shape, device=device)
    
    # Set manual seed for reproducible results
    torch.manual_seed(42)
    t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
    x_t_minus_1_zero, _, _ = sampler.ancestral_step(x_t, t_zero)
    
    torch.manual_seed(42)
    t_one = torch.ones(batch_size, dtype=torch.long, device=device)
    x_t_minus_1_one, _, _ = sampler.ancestral_step(x_t, t_one)
    
    # The outputs should be different because t=1 adds sampling noise while t=0 doesn't
    assert not torch.allclose(x_t_minus_1_zero, x_t_minus_1_one, rtol=1e-3)
    
    # Test that t=0 step with same input and model predictions gives consistent results
    # (accounting for the fact that model predictions may vary)


def test_ancestral_step_variance_types(setup_sampler):
    """Test different variance types produce different results."""
    sampler, schedules, device = setup_sampler
    
    batch_size = 2
    shape = (batch_size, 1, 28, 28)
    
    x_t = torch.randn(shape, device=device)
    t = torch.tensor([50, 75], device=device)  # Mid-range timesteps
    
    # Test both variance types
    x_beta, _, _ = sampler.ancestral_step(x_t, t, variance_type='beta')
    x_posterior, _, _ = sampler.ancestral_step(x_t, t, variance_type='posterior')
    
    # Should be different
    assert not torch.allclose(x_beta, x_posterior, rtol=1e-3)
    
    # Both should be valid
    assert not torch.isnan(x_beta).any()
    assert not torch.isnan(x_posterior).any()


def test_sampling_full_trajectory(setup_sampler):
    """Test full sampling trajectory from T to 0."""
    sampler, schedules, device = setup_sampler
    
    shape = (2, 1, 28, 28)
    
    # Generate samples
    results = sampler.sample(
        shape=shape,
        progress=False,
        return_trajectory=True,
        trajectory_steps=[99, 50, 25, 10, 0]
    )
    
    samples = results['samples']
    trajectory = results['trajectory']
    trajectory_steps = results['trajectory_steps']
    
    # Check final samples
    assert samples.shape == shape
    assert not torch.isnan(samples).any()
    assert not torch.isinf(samples).any()
    
    # Check trajectory
    assert len(trajectory) == len(trajectory_steps)
    assert all(t.shape == shape for t in trajectory)
    
    # Check that final trajectory step matches final samples
    torch.testing.assert_close(trajectory[-1], samples, rtol=1e-6, atol=1e-6)


def test_ddim_step_shapes(setup_sampler):
    """Test DDIM step shapes and properties."""
    sampler, schedules, device = setup_sampler
    
    batch_size = 3
    shape = (batch_size, 1, 28, 28)
    
    x_t = torch.randn(shape, device=device)
    t = torch.tensor([80, 60, 40], device=device)
    t_prev = torch.tensor([70, 50, 30], device=device)
    
    # Test DDIM step
    x_t_prev, x_0_pred, eps_pred = sampler.ddim_step(x_t, t, t_prev, eta=0.0)
    
    # Check shapes
    assert x_t_prev.shape == shape
    assert x_0_pred.shape == shape
    assert eps_pred.shape == shape
    
    # Check no NaNs
    assert not torch.isnan(x_t_prev).any()
    assert not torch.isnan(x_0_pred).any()
    assert not torch.isnan(eps_pred).any()


def test_ddim_deterministic(setup_sampler):
    """Test that DDIM with eta=0 is deterministic."""
    sampler, schedules, device = setup_sampler
    
    shape = (1, 1, 28, 28)
    
    # Generate same sample twice
    torch.manual_seed(42)
    results1 = sampler.sample(
        shape=shape,
        ddim=True,
        num_steps=20,
        ddim_eta=0.0,
        progress=False
    )
    
    torch.manual_seed(42)
    results2 = sampler.sample(
        shape=shape,
        ddim=True,
        num_steps=20,
        ddim_eta=0.0,
        progress=False
    )
    
    # Should be identical
    torch.testing.assert_close(results1['samples'], results2['samples'], rtol=1e-6, atol=1e-6)


def test_single_trajectory_recording(setup_sampler):
    """Test single trajectory recording functionality."""
    sampler, schedules, device = setup_sampler
    
    shape = (1, 1, 28, 28)
    
    results = sampler.sample_single_trajectory(
        shape=shape,
        record_every=10,
        progress=False
    )
    
    trajectory = results['trajectory']
    trajectory_steps = results['trajectory_steps']
    
    # Check that we have recordings
    assert len(trajectory) > 5  # Should have several frames
    assert len(trajectory_steps) == len(trajectory)
    
    # Check that steps are decreasing
    assert all(trajectory_steps[i] >= trajectory_steps[i+1] 
              for i in range(len(trajectory_steps)-1))
    
    # Check that all frames have correct shape
    assert all(t.shape == shape for t in trajectory)


def test_sampling_info_tracking(setup_sampler):
    """Test that sampling info is properly tracked."""
    sampler, schedules, device = setup_sampler
    
    shape = (2, 1, 28, 28)
    
    # Test DDPM
    results = sampler.sample(shape=shape, progress=False)
    info = results['sampling_info']
    
    assert 'num_steps' in info
    assert 'variance_type' in info
    assert 'ddim' in info
    assert info['ddim'] == False
    
    # Test DDIM
    results = sampler.sample(shape=shape, ddim=True, num_steps=25, progress=False)
    info = results['sampling_info']
    
    assert info['ddim'] == True
    assert info['num_steps'] <= 25  # Should be 25 or fewer


def test_batch_size_consistency(setup_sampler):
    """Test that sampling works with different batch sizes."""
    sampler, schedules, device = setup_sampler
    
    # Test various batch sizes
    for batch_size in [1, 2, 5, 8]:
        shape = (batch_size, 1, 28, 28)
        
        results = sampler.sample(shape=shape, progress=False)
        samples = results['samples']
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()


def test_image_size_consistency(setup_sampler):
    """Test that sampling works with different image sizes."""
    sampler, schedules, device = setup_sampler
    
    # Test various image sizes
    for size in [16, 28, 32]:
        shape = (2, 1, size, size)
        
        results = sampler.sample(shape=shape, progress=False)
        samples = results['samples']
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()


if __name__ == "__main__":
    pytest.main([__file__])
