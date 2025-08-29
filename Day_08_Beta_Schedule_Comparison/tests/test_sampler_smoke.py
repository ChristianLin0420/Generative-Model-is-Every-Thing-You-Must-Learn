"""
Sampler smoke tests: verify reverse pass shapes, no NaNs, proper sampling functionality.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.unet_small import UNetSmall
from src.schedules import get_schedule
from src.sampler import DDPMSampler, DDIMSampler
from src.utils import set_seed


class TestSamplerSmoke:
    """Smoke tests for sampling functionality."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def dummy_model(self, device):
        """Create a small model for testing."""
        model = UNetSmall(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_multipliers=[1, 2],
            time_embed_dim=64,
            num_res_blocks=1
        ).to(device)
        model.eval()
        return model
    
    @pytest.fixture
    def short_schedule(self, device):
        """Create a short schedule for fast testing."""
        schedule = get_schedule("linear", T=10)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        return schedule
    
    @pytest.mark.parametrize("schedule_name", ["linear", "cosine", "quadratic"])
    def test_ddpm_sampler_initialization(self, schedule_name, device):
        """Test DDPM sampler can be initialized."""
        set_seed(42)
        
        model = UNetSmall(in_channels=1, base_channels=16).to(device)
        schedule = get_schedule(schedule_name, T=10)
        
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        sampler = DDPMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars']
        )
        
        assert sampler.model == model
        assert sampler.T == 10
        assert torch.allclose(sampler.betas, schedule['betas'])
    
    @pytest.mark.parametrize("schedule_name", ["linear", "cosine", "quadratic"])
    def test_ddim_sampler_initialization(self, schedule_name, device):
        """Test DDIM sampler can be initialized."""
        set_seed(42)
        
        model = UNetSmall(in_channels=1, base_channels=16).to(device)
        schedule = get_schedule(schedule_name, T=10)
        
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        sampler = DDIMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars'],
            eta=0.0
        )
        
        assert sampler.model == model
        assert sampler.T == 10
        assert sampler.eta == 0.0
    
    def test_ddpm_single_step(self, dummy_model, short_schedule, device):
        """Test DDPM single reverse step works correctly."""
        set_seed(42)
        
        sampler = DDPMSampler(
            dummy_model,
            short_schedule['betas'],
            short_schedule['alphas'],
            short_schedule['alpha_bars']
        )
        
        # Test single step
        batch_size = 2
        x_t = torch.randn(batch_size, 1, 28, 28, device=device)
        t = torch.full((batch_size,), 5, device=device, dtype=torch.long)
        
        with torch.no_grad():
            x_prev = sampler.sample_step(x_t, t)
        
        # Check output properties
        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()
        assert not torch.isinf(x_prev).any()
        assert x_prev.device == device
    
    def test_ddim_single_step(self, dummy_model, short_schedule, device):
        """Test DDIM single reverse step works correctly."""
        set_seed(42)
        
        sampler = DDIMSampler(
            dummy_model,
            short_schedule['betas'],
            short_schedule['alphas'],
            short_schedule['alpha_bars'],
            eta=0.0
        )
        
        # Test single step
        batch_size = 2
        x_t = torch.randn(batch_size, 1, 28, 28, device=device)
        t = torch.full((batch_size,), 5, device=device, dtype=torch.long)
        t_prev = torch.full((batch_size,), 4, device=device, dtype=torch.long)
        
        with torch.no_grad():
            x_prev = sampler.sample_step(x_t, t, t_prev)
        
        # Check output properties
        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()
        assert not torch.isinf(x_prev).any()
        assert x_prev.device == device
    
    @pytest.mark.parametrize("shape", [
        (1, 1, 28, 28),  # Single MNIST-like
        (4, 1, 28, 28),  # Batch MNIST-like
        (2, 3, 32, 32),  # CIFAR-like
    ])
    def test_ddpm_full_sampling(self, shape, device):
        """Test DDPM full sampling process."""
        set_seed(42)
        
        # Create model with appropriate channels
        in_channels = shape[1]
        model = UNetSmall(
            in_channels=in_channels,
            out_channels=in_channels,
            base_channels=16,
            channel_multipliers=[1, 2]
        ).to(device)
        
        # Short schedule for speed
        schedule = get_schedule("linear", T=5)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        sampler = DDPMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars']
        )
        
        # Full sampling
        with torch.no_grad():
            samples = sampler.sample(shape, device)
        
        # Check output properties
        assert samples.shape == shape
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()
        assert samples.device == device
        
        print(f"DDPM sampling {shape}: range [{samples.min():.3f}, {samples.max():.3f}]")
    
    @pytest.mark.parametrize("num_steps", [5, 3, 1])
    def test_ddim_accelerated_sampling(self, num_steps, device):
        """Test DDIM accelerated sampling."""
        set_seed(42)
        
        model = UNetSmall(in_channels=1, base_channels=16).to(device)
        
        # Longer schedule to test acceleration
        schedule = get_schedule("linear", T=10)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        sampler = DDIMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars'],
            eta=0.0
        )
        
        shape = (2, 1, 28, 28)
        
        with torch.no_grad():
            samples = sampler.sample(shape, device, num_steps=num_steps)
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()
        
        print(f"DDIM {num_steps} steps: range [{samples.min():.3f}, {samples.max():.3f}]")
    
    @pytest.mark.parametrize("eta", [0.0, 0.5, 1.0])
    def test_ddim_eta_parameter(self, eta, device):
        """Test DDIM with different eta values."""
        set_seed(42)
        
        model = UNetSmall(in_channels=1, base_channels=16).to(device)
        schedule = get_schedule("linear", T=5)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        sampler = DDIMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars'],
            eta=eta
        )
        
        shape = (2, 1, 28, 28)
        
        with torch.no_grad():
            samples = sampler.sample(shape, device, num_steps=3)
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()
        
        print(f"DDIM η={eta}: range [{samples.min():.3f}, {samples.max():.3f}]")
    
    def test_ddpm_trajectory_sampling(self, dummy_model, short_schedule, device):
        """Test DDPM trajectory sampling returns full trajectory."""
        set_seed(42)
        
        sampler = DDPMSampler(
            dummy_model,
            short_schedule['betas'],
            short_schedule['alphas'],
            short_schedule['alpha_bars']
        )
        
        shape = (2, 1, 28, 28)
        
        with torch.no_grad():
            trajectory = sampler.sample(shape, device, return_trajectory=True)
        
        # Trajectory should have T+1 timesteps (including initial noise)
        expected_shape = (shape[0], short_schedule['betas'].shape[0] + 1) + shape[1:]
        assert trajectory.shape == expected_shape
        assert not torch.isnan(trajectory).any()
        assert not torch.isinf(trajectory).any()
        
        print(f"Trajectory shape: {trajectory.shape}")
    
    def test_ddim_trajectory_sampling(self, dummy_model, short_schedule, device):
        """Test DDIM trajectory sampling."""
        set_seed(42)
        
        sampler = DDIMSampler(
            dummy_model,
            short_schedule['betas'],
            short_schedule['alphas'],
            short_schedule['alpha_bars'],
            eta=0.0
        )
        
        shape = (2, 1, 28, 28)
        num_steps = 3
        
        with torch.no_grad():
            trajectory = sampler.sample(shape, device, num_steps=num_steps, return_trajectory=True)
        
        # Trajectory should have num_steps+1 timesteps
        expected_shape = (shape[0], num_steps + 1) + shape[1:]
        assert trajectory.shape == expected_shape
        assert not torch.isnan(trajectory).any()
        
        print(f"DDIM trajectory shape: {trajectory.shape}")
    
    def test_sampling_determinism(self, device):
        """Test that sampling is deterministic with same seed."""
        model = UNetSmall(in_channels=1, base_channels=16).to(device)
        schedule = get_schedule("linear", T=5)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        sampler = DDPMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars']
        )
        
        shape = (2, 1, 28, 28)
        
        # Sample twice with same seed
        set_seed(42)
        torch.manual_seed(42)
        with torch.no_grad():
            samples1 = sampler.sample(shape, device)
        
        set_seed(42)
        torch.manual_seed(42)
        with torch.no_grad():
            samples2 = sampler.sample(shape, device)
        
        # Should be identical (within floating point precision)
        assert torch.allclose(samples1, samples2, atol=1e-6)
    
    def test_ddim_determinism(self, device):
        """Test DDIM determinism with eta=0."""
        model = UNetSmall(in_channels=1, base_channels=16).to(device)
        schedule = get_schedule("linear", T=5)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        sampler = DDIMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars'],
            eta=0.0  # Deterministic
        )
        
        shape = (2, 1, 28, 28)
        
        # Sample twice with same seed
        set_seed(42)
        torch.manual_seed(42)
        with torch.no_grad():
            samples1 = sampler.sample(shape, device, num_steps=3)
        
        set_seed(42)
        torch.manual_seed(42)
        with torch.no_grad():
            samples2 = sampler.sample(shape, device, num_steps=3)
        
        # Should be identical for eta=0
        assert torch.allclose(samples1, samples2, atol=1e-6)


class TestSamplerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_t_equals_zero_step(self, device):
        """Test sampling step at t=0 (final step)."""
        model = UNetSmall(in_channels=1, base_channels=16).to(device)
        schedule = get_schedule("linear", T=5)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        sampler = DDPMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars']
        )
        
        x_t = torch.randn(2, 1, 28, 28, device=device)
        t = torch.zeros(2, device=device, dtype=torch.long)  # t=0
        
        with torch.no_grad():
            x_prev = sampler.sample_step(x_t, t)
        
        # Should not add noise at final step
        assert x_prev.shape == x_t.shape
        assert not torch.isnan(x_prev).any()
    
    def test_single_timestep_schedule(self, device):
        """Test sampling with T=1."""
        model = UNetSmall(in_channels=1, base_channels=16).to(device)
        schedule = get_schedule("linear", T=1)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        sampler = DDPMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars']
        )
        
        shape = (2, 1, 28, 28)
        
        with torch.no_grad():
            samples = sampler.sample(shape, device)
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()


if __name__ == "__main__":
    # Run tests manually
    print("Running sampler smoke tests...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test basic sampling functionality
    for schedule_name in ["linear", "cosine", "quadratic"]:
        print(f"\nTesting {schedule_name} schedule...")
        
        set_seed(42)
        
        # Create model and schedule
        model = UNetSmall(in_channels=1, base_channels=16).to(device)
        schedule = get_schedule(schedule_name, T=5)  # Short for speed
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        # Test DDPM sampler
        ddpm_sampler = DDPMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars']
        )
        
        shape = (2, 1, 28, 28)
        
        with torch.no_grad():
            samples = ddpm_sampler.sample(shape, device)
        
        assert samples.shape == shape
        assert not torch.isnan(samples).any()
        print(f"  ✓ DDPM sampling: {samples.shape}, range [{samples.min():.3f}, {samples.max():.3f}]")
        
        # Test DDIM sampler
        ddim_sampler = DDIMSampler(
            model,
            schedule['betas'],
            schedule['alphas'],
            schedule['alpha_bars'],
            eta=0.0
        )
        
        with torch.no_grad():
            ddim_samples = ddim_sampler.sample(shape, device, num_steps=3)
        
        assert ddim_samples.shape == shape
        assert not torch.isnan(ddim_samples).any()
        print(f"  ✓ DDIM sampling: {ddim_samples.shape}, range [{ddim_samples.min():.3f}, {ddim_samples.max():.3f}]")
    
    print("\n✓ All sampler smoke tests passed!")
    print("Sampling components are working correctly.")
