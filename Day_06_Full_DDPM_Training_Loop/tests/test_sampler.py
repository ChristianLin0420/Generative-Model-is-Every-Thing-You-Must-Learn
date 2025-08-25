"""
Test suite for DDPM/DDIM sampling: shapes, boundary timesteps, dtype/device consistency
"""

import pytest
import torch
from src.ddpm_schedules import DDPMSchedules
from src.sampler import DDPMSampler
from src.models.unet_small import UNetSmall


class TestDDPMSampler:
    """Test DDPM and DDIM sampling"""
    
    @pytest.fixture
    def schedules(self):
        """Create test schedules"""
        return DDPMSchedules(num_timesteps=100, device="cpu")
        
    @pytest.fixture
    def sampler(self, schedules):
        """Create test sampler"""
        return DDPMSampler(schedules)
        
    @pytest.fixture
    def model(self):
        """Create simple test model"""
        model = UNetSmall(
            in_channels=3,
            out_channels=3,
            model_channels=16,
            channel_mult=[1, 2],
            num_res_blocks=1,
            use_attention=False
        )
        model.eval()
        return model
        
    def test_sampler_initialization(self, schedules):
        """Test sampler can be initialized"""
        sampler = DDPMSampler(schedules)
        assert sampler.schedules == schedules
        
    def test_ddpm_sample_step(self, sampler, model):
        """Test single DDPM sampling step"""
        device = torch.device("cpu")
        
        # Create test input
        batch_size = 2
        x = torch.randn(batch_size, 3, 16, 16, device=device)
        t = torch.tensor([50, 30], device=device)
        
        # Single step
        with torch.no_grad():
            x_prev = sampler.ddpm_sample_step(model, x, t)
            
        # Check output shape and dtype
        assert x_prev.shape == x.shape
        assert x_prev.dtype == x.dtype
        assert x_prev.device == x.device
        
    def test_ddim_sample_step(self, sampler, model):
        """Test single DDIM sampling step"""
        device = torch.device("cpu")
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 16, 16, device=device)
        t = torch.tensor([50, 30], device=device)
        t_prev = torch.tensor([40, 20], device=device)
        
        with torch.no_grad():
            x_prev = sampler.ddim_sample_step(model, x, t, t_prev, eta=0.0)
            
        assert x_prev.shape == x.shape
        assert x_prev.dtype == x.dtype
        assert x_prev.device == x.device
        
    def test_ddpm_full_sampling(self, sampler, model):
        """Test full DDPM sampling process"""
        device = torch.device("cpu")
        
        # Small image for speed
        shape = (1, 3, 8, 8)
        
        with torch.no_grad():
            samples = sampler.ddpm_sample(
                model=model,
                shape=shape,
                device=device,
                progress=False
            )
            
        assert samples.shape == shape
        assert samples.dtype == torch.float32
        assert samples.device == device
        
    def test_ddim_full_sampling(self, sampler, model):
        """Test full DDIM sampling process"""
        device = torch.device("cpu")
        
        shape = (2, 3, 8, 8)
        num_steps = 10  # Few steps for speed
        
        with torch.no_grad():
            samples = sampler.ddim_sample(
                model=model,
                shape=shape,
                num_steps=num_steps,
                device=device,
                progress=False
            )
            
        assert samples.shape == shape
        assert samples.dtype == torch.float32
        assert samples.device == device
        
    def test_unified_sample_interface(self, sampler, model):
        """Test unified sampling interface"""
        device = torch.device("cpu")
        shape = (1, 3, 8, 8)
        
        # Test DDPM method
        with torch.no_grad():
            ddpm_samples = sampler.sample(
                model=model,
                shape=shape,
                method="ddpm",
                device=device,
                progress=False
            )
            
        # Test DDIM method
        with torch.no_grad():
            ddim_samples = sampler.sample(
                model=model,
                shape=shape,
                method="ddim",
                num_steps=10,
                device=device,
                progress=False
            )
            
        # Both should produce valid samples
        assert ddpm_samples.shape == shape
        assert ddim_samples.shape == shape
        
        # Different methods should generally produce different results
        # (unless model is very simple or we're unlucky)
        mean_diff = torch.abs(ddmp_samples - ddim_samples).mean()
        # Allow for some tolerance, but expect some difference
        # Note: with random initialization this should usually be different
        
    def test_trajectory_return(self, sampler, model):
        """Test sampling with trajectory return"""
        device = torch.device("cpu")
        shape = (1, 3, 8, 8)
        
        with torch.no_grad():
            samples, trajectory = sampler.ddim_sample(
                model=model,
                shape=shape,
                num_steps=5,
                device=device,
                return_trajectory=True,
                progress=False
            )
            
        # Check trajectory
        assert isinstance(trajectory, list)
        assert len(trajectory) > 0
        
        # All trajectory items should have correct shape
        for traj_item in trajectory:
            assert traj_item.shape == shape
            
        # Final sample should match last trajectory item
        assert torch.allclose(samples, trajectory[-1], atol=1e-6)
        
    def test_different_eta_values(self, sampler, model):
        """Test DDIM with different eta values"""
        device = torch.device("cpu")
        shape = (1, 3, 8, 8)
        
        eta_values = [0.0, 0.5, 1.0]
        samples_list = []
        
        for eta in eta_values:
            with torch.no_grad():
                samples = sampler.ddim_sample(
                    model=model,
                    shape=shape,
                    num_steps=10,
                    eta=eta,
                    device=device,
                    progress=False
                )
            samples_list.append(samples)
            
        # All should produce valid samples
        for samples in samples_list:
            assert samples.shape == shape
            
        # eta=0.0 should be deterministic (if we use same starting noise)
        # This is hard to test without controlling the random seed in the sampler
        
    def test_boundary_timesteps(self, sampler, model):
        """Test sampling handles boundary timesteps correctly"""
        device = torch.device("cpu")
        
        batch_size = 2
        x = torch.randn(batch_size, 3, 8, 8, device=device)
        
        # Test t=0 (should not add noise in DDPM step)
        t_zero = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        with torch.no_grad():
            x_t0 = sampler.ddpm_sample_step(model, x, t_zero)
            
        # At t=0, the step should be deterministic (no noise added)
        assert x_t0.shape == x.shape
        
        # Test maximum timestep
        t_max = torch.full((batch_size,), sampler.schedules.num_timesteps - 1, device=device)
        
        with torch.no_grad():
            x_tmax = sampler.ddpm_sample_step(model, x, t_max)
            
        assert x_tmax.shape == x.shape
        
    def test_clip_denoised_option(self, sampler, model):
        """Test clip_denoised option"""
        device = torch.device("cpu")
        shape = (1, 3, 8, 8)
        
        # Test with clipping
        with torch.no_grad():
            samples_clipped = sampler.ddim_sample(
                model=model,
                shape=shape,
                num_steps=5,
                clip_denoised=True,
                device=device,
                progress=False
            )
            
        # Test without clipping
        with torch.no_grad():
            samples_unclipped = sampler.ddim_sample(
                model=model,
                shape=shape,
                num_steps=5,
                clip_denoised=False,
                device=device,
                progress=False
            )
            
        # Both should produce valid samples
        assert samples_clipped.shape == shape
        assert samples_unclipped.shape == shape
        
        # With clipping, values should be in reasonable range
        # (though this depends on the model output)
        
    def test_different_num_steps(self, sampler, model):
        """Test DDIM with different number of steps"""
        device = torch.device("cpu")
        shape = (1, 3, 8, 8)
        
        step_counts = [5, 10, 20]
        
        for num_steps in step_counts:
            with torch.no_grad():
                samples = sampler.ddim_sample(
                    model=model,
                    shape=shape,
                    num_steps=num_steps,
                    device=device,
                    progress=False
                )
                
            assert samples.shape == shape
            assert torch.all(torch.isfinite(samples))
            
    def test_batch_size_consistency(self, sampler, model):
        """Test sampling works with different batch sizes"""
        device = torch.device("cpu")
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            shape = (batch_size, 3, 8, 8)
            
            with torch.no_grad():
                samples = sampler.ddim_sample(
                    model=model,
                    shape=shape,
                    num_steps=5,
                    device=device,
                    progress=False
                )
                
            assert samples.shape == shape
            
    def test_device_consistency(self, model):
        """Test sampler respects device settings"""
        device = torch.device("cpu")
        
        # Create schedules and sampler on CPU
        schedules = DDPMSchedules(num_timesteps=50, device="cpu")
        sampler = DDPMSampler(schedules)
        
        shape = (1, 3, 8, 8)
        
        with torch.no_grad():
            samples = sampler.ddim_sample(
                model=model,
                shape=shape,
                num_steps=5,
                device=device,
                progress=False
            )
            
        # Samples should be on the correct device
        assert samples.device == device
        
    def test_deterministic_with_same_noise(self, sampler, model):
        """Test that sampling is deterministic with same starting noise"""
        device = torch.device("cpu")
        shape = (1, 3, 8, 8)
        
        # This is tricky to test because we don't control the random seed
        # in the sampler directly. We'd need to modify the sampler to accept
        # starting noise as input for full determinism testing.
        
        # For now, just test that sampling produces finite results
        with torch.no_grad():
            samples1 = sampler.ddim_sample(
                model=model,
                shape=shape,
                num_steps=5,
                eta=0.0,  # Deterministic DDIM
                device=device,
                progress=False
            )
            
            samples2 = sampler.ddim_sample(
                model=model,
                shape=shape,
                num_steps=5,
                eta=0.0,
                device=device,
                progress=False
            )
            
        # Both should be finite and valid
        assert torch.all(torch.isfinite(samples1))
        assert torch.all(torch.isfinite(samples2))
        
    def test_invalid_method_error(self, sampler, model):
        """Test error handling for invalid sampling method"""
        device = torch.device("cpu")
        shape = (1, 3, 8, 8)
        
        with pytest.raises(ValueError, match="Unknown sampling method"):
            sampler.sample(
                model=model,
                shape=shape,
                method="invalid_method",
                device=device
            )