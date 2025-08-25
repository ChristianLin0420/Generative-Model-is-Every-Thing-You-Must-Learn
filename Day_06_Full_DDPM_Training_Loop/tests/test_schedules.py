"""
Test suite for DDPM schedules: β, α, ᾱ monotonicity, shapes, and boundaries
"""

import pytest
import torch
import numpy as np
from src.ddpm_schedules import DDPMSchedules, linear_beta_schedule, cosine_beta_schedule


class TestSchedules:
    """Test DDPM noise schedules"""
    
    def test_linear_beta_schedule(self):
        """Test linear beta schedule properties"""
        num_timesteps = 1000
        beta_start = 0.0001
        beta_end = 0.02
        
        betas = linear_beta_schedule(num_timesteps, beta_start, beta_end)
        
        # Check shape
        assert betas.shape == (num_timesteps,)
        
        # Check values are in valid range
        assert torch.all(betas >= 0.0)
        assert torch.all(betas <= 1.0)
        
        # Check monotonicity (non-decreasing)
        assert torch.all(betas[1:] >= betas[:-1])
        
        # Check boundary values
        assert torch.isclose(betas[0], torch.tensor(beta_start), atol=1e-6)
        assert torch.isclose(betas[-1], torch.tensor(beta_end), atol=1e-6)
        
    def test_cosine_beta_schedule(self):
        """Test cosine beta schedule properties"""
        num_timesteps = 1000
        
        betas = cosine_beta_schedule(num_timesteps)
        
        # Check shape
        assert betas.shape == (num_timesteps,)
        
        # Check values are in valid range
        assert torch.all(betas >= 0.0)
        assert torch.all(betas < 1.0)  # Should be clipped below 0.999
        
        # Check no NaN or inf values
        assert torch.all(torch.isfinite(betas))
        
    def test_ddmp_schedules_initialization(self):
        """Test DDPMSchedules initialization"""
        num_timesteps = 100
        schedules = DDPMSchedules(num_timesteps=num_timesteps, device="cpu")
        
        # Check all tensors have correct shape
        assert schedules.betas.shape == (num_timesteps,)
        assert schedules.alphas.shape == (num_timesteps,)
        assert schedules.alphas_cumprod.shape == (num_timesteps,)
        assert schedules.alphas_cumprod_prev.shape == (num_timesteps,)
        
        # Check device
        assert schedules.betas.device.type == "cpu"
        
    def test_alpha_relationships(self):
        """Test mathematical relationships between α, β, ᾱ"""
        schedules = DDPMSchedules(num_timesteps=100, device="cpu")
        
        # α = 1 - β
        expected_alphas = 1.0 - schedules.betas
        assert torch.allclose(schedules.alphas, expected_alphas, atol=1e-6)
        
        # ᾱ should be cumulative product of α
        expected_alphas_cumprod = torch.cumprod(schedules.alphas, dim=0)
        assert torch.allclose(schedules.alphas_cumprod, expected_alphas_cumprod, atol=1e-6)
        
        # ᾱ should be monotonically decreasing
        assert torch.all(schedules.alphas_cumprod[1:] <= schedules.alphas_cumprod[:-1])
        
        # ᾱ_0 should be close to α_0 
        assert torch.isclose(schedules.alphas_cumprod[0], schedules.alphas[0], atol=1e-6)
        
    def test_posterior_variance(self):
        """Test posterior variance computation"""
        schedules = DDPMSchedules(num_timesteps=100, device="cpu")
        
        # Posterior variance should be positive
        assert torch.all(schedules.posterior_variance >= 0)
        
        # Check shape
        assert schedules.posterior_variance.shape == (100,)
        
        # First timestep variance should be small (close to 0)
        assert schedules.posterior_variance[0] < 0.1
        
    def test_extract_function(self):
        """Test extract function for indexing"""
        schedules = DDPMSchedules(num_timesteps=100, device="cpu")
        
        # Test single timestep extraction
        t = torch.tensor([50])
        shape = (1, 3, 32, 32)
        
        extracted = schedules.extract(schedules.betas, t, shape)
        
        # Check shape is broadcastable
        assert extracted.shape == (1, 1, 1, 1)
        assert torch.isclose(extracted.item(), schedules.betas[50].item(), atol=1e-6)
        
        # Test batch extraction
        batch_size = 4
        t_batch = torch.randint(0, 100, (batch_size,))
        shape = (batch_size, 3, 32, 32)
        
        extracted_batch = schedules.extract(schedules.alphas_cumprod, t_batch, shape)
        
        assert extracted_batch.shape == (batch_size, 1, 1, 1)
        
        # Verify correctness
        for i in range(batch_size):
            expected = schedules.alphas_cumprod[t_batch[i]]
            assert torch.isclose(extracted_batch[i, 0, 0, 0], expected, atol=1e-6)
            
    def test_q_sample(self):
        """Test forward diffusion sampling"""
        schedules = DDPMSchedules(num_timesteps=100, device="cpu")
        
        # Create test data
        batch_size = 2
        x_start = torch.randn(batch_size, 3, 16, 16)
        t = torch.randint(0, 100, (batch_size,))
        noise = torch.randn_like(x_start)
        
        # Sample noisy images
        x_t = schedules.q_sample(x_start, t, noise)
        
        # Check shape preservation
        assert x_t.shape == x_start.shape
        
        # Check that x_t is linear combination of x_start and noise
        # At t=0, should be close to x_start
        t_zero = torch.zeros(batch_size, dtype=torch.long)
        x_0 = schedules.q_sample(x_start, t_zero, noise)
        
        # Should be close to original (but not exact due to small noise)
        assert torch.allclose(x_0, x_start, atol=0.1)
        
    def test_snr_computation(self):
        """Test signal-to-noise ratio computation"""
        schedules = DDPMSchedules(num_timesteps=100, device="cpu")
        
        snr = schedules.get_snr()
        
        # Check shape
        assert snr.shape == (100,)
        
        # SNR should be positive
        assert torch.all(snr > 0)
        
        # SNR should be decreasing (more noise over time)
        assert torch.all(snr[1:] <= snr[:-1])
        
        # SNR at t=0 should be high (little noise)
        assert snr[0] > snr[-1]
        
    def test_device_transfer(self):
        """Test moving schedules to different devices"""
        schedules = DDPMSchedules(num_timesteps=50, device="cpu")
        
        # Test to method
        schedules_moved = schedules.to("cpu")
        assert schedules_moved.device == "cpu"
        
        # All tensors should be on the right device
        assert schedules_moved.betas.device.type == "cpu"
        assert schedules_moved.alphas_cumprod.device.type == "cpu"
        
    def test_boundary_conditions(self):
        """Test boundary conditions and edge cases"""
        schedules = DDPMSchedules(num_timesteps=10, device="cpu")
        
        # Test t=0 extraction
        t_zero = torch.tensor([0])
        shape = (1, 1, 4, 4)
        
        alpha_bar_0 = schedules.extract(schedules.alphas_cumprod, t_zero, shape)
        # Should be close to 1 at t=0
        assert alpha_bar_0.item() > 0.9
        
        # Test t=max extraction  
        t_max = torch.tensor([schedules.num_timesteps - 1])
        alpha_bar_max = schedules.extract(schedules.alphas_cumprod, t_max, shape)
        # Should be much smaller at max timestep
        assert alpha_bar_max.item() < alpha_bar_0.item()
        
    def test_different_schedule_types(self):
        """Test different schedule types produce valid results"""
        num_timesteps = 100
        
        # Linear schedule
        linear_schedules = DDPMSchedules(
            num_timesteps=num_timesteps,
            schedule_type="linear",
            device="cpu"
        )
        
        # Cosine schedule
        cosine_schedules = DDPMSchedules(
            num_timesteps=num_timesteps,
            schedule_type="cosine", 
            device="cpu"
        )
        
        # Both should have valid properties
        for schedules in [linear_schedules, cosine_schedules]:
            assert torch.all(schedules.betas >= 0)
            assert torch.all(schedules.betas <= 1)
            assert torch.all(schedules.alphas_cumprod >= 0)
            assert torch.all(schedules.alphas_cumprod <= 1)
            
        # Different schedules should produce different results
        assert not torch.allclose(linear_schedules.betas, cosine_schedules.betas)
        
    def test_posterior_mean_variance(self):
        """Test posterior mean and variance computation"""
        schedules = DDPMSchedules(num_timesteps=100, device="cpu")
        
        # Create test data
        batch_size = 2
        x_start = torch.randn(batch_size, 3, 8, 8)
        x_t = torch.randn(batch_size, 3, 8, 8)
        t = torch.randint(1, 100, (batch_size,))  # Avoid t=0
        
        posterior_mean, posterior_variance, posterior_log_variance = \
            schedules.q_posterior_mean_variance(x_start, x_t, t)
            
        # Check shapes
        assert posterior_mean.shape == x_start.shape
        assert posterior_variance.shape == x_start.shape
        assert posterior_log_variance.shape == x_start.shape
        
        # Variance should be positive
        assert torch.all(posterior_variance > 0)
        
        # Log variance should be finite
        assert torch.all(torch.isfinite(posterior_log_variance))
        
    def test_numerical_stability(self):
        """Test numerical stability with extreme values"""
        # Very small number of timesteps
        schedules_small = DDPMSchedules(num_timesteps=2, device="cpu")
        assert torch.all(torch.isfinite(schedules_small.alphas_cumprod))
        
        # Large number of timesteps
        schedules_large = DDPMSchedules(num_timesteps=2000, device="cpu")
        assert torch.all(torch.isfinite(schedules_large.alphas_cumprod))
        assert schedules_large.alphas_cumprod[-1] > 0  # Should not underflow to 0