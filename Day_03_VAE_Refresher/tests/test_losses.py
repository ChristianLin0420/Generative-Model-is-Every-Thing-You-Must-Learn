"""
Test suite for VAE loss functions.
Tests reconstruction losses, KL divergence, ELBO, and beta scheduling.
"""

import pytest
import torch
import numpy as np

from src.losses import (
    reconstruction_loss, kl_divergence_gaussian, elbo_loss,
    BetaScheduler, compute_iwae_bound, free_bits_kl
)


class TestLossFunctions:
    """Test loss function implementations."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    @pytest.fixture
    def sample_images(self, device):
        """Sample images in [0, 1] range."""
        return torch.rand(4, 1, 8, 8, device=device)
    
    @pytest.fixture
    def sample_logits(self, device):
        """Sample logits for BCE loss."""
        return torch.randn(4, 1, 8, 8, device=device)
    
    @pytest.fixture
    def sample_latent_params(self, device):
        """Sample latent parameters."""
        mu = torch.randn(4, 16, device=device)
        logvar = torch.randn(4, 16, device=device)
        return mu, logvar
    
    def test_bce_reconstruction_loss(self, sample_images, sample_logits, device):
        """Test BCE reconstruction loss."""
        loss = reconstruction_loss(sample_logits, sample_images, "bce", "mean")
        
        # Should be positive
        assert loss.item() > 0
        
        # Should be scalar
        assert loss.shape == ()
        
        # Test with perfect reconstruction (logits that lead to targets)
        perfect_logits = torch.logit(torch.clamp(sample_images, 1e-7, 1-1e-7))
        perfect_loss = reconstruction_loss(perfect_logits, sample_images, "bce", "mean")
        
        # Perfect reconstruction should have lower loss
        assert perfect_loss.item() < loss.item()
    
    def test_l2_reconstruction_loss(self, sample_images, device):
        """Test L2 reconstruction loss."""
        reconstructions = sample_images + 0.1 * torch.randn_like(sample_images)
        
        loss = reconstruction_loss(reconstructions, sample_images, "l2", "mean")
        
        # Should be positive
        assert loss.item() > 0
        
        # Test with perfect reconstruction
        perfect_loss = reconstruction_loss(sample_images, sample_images, "l2", "mean")
        assert perfect_loss.item() == pytest.approx(0.0, abs=1e-6)
    
    def test_charbonnier_reconstruction_loss(self, sample_images, device):
        """Test Charbonnier reconstruction loss."""
        reconstructions = sample_images + 0.1 * torch.randn_like(sample_images)
        
        loss = reconstruction_loss(reconstructions, sample_images, "charbonnier", "mean")
        
        # Should be positive
        assert loss.item() > 0
        
        # Should be different from L2
        l2_loss = reconstruction_loss(reconstructions, sample_images, "l2", "mean")
        assert not torch.allclose(loss, l2_loss)
    
    def test_reconstruction_loss_reductions(self, sample_images, sample_logits):
        """Test different reduction modes for reconstruction loss."""
        loss_mean = reconstruction_loss(sample_logits, sample_images, "bce", "mean")
        loss_sum = reconstruction_loss(sample_logits, sample_images, "bce", "sum")
        loss_none = reconstruction_loss(sample_logits, sample_images, "bce", "none")
        
        # Check shapes
        assert loss_mean.shape == ()
        assert loss_sum.shape == ()
        assert loss_none.shape == (4,)  # Batch size
        
        # Relationships
        assert torch.allclose(loss_mean, loss_none.mean())
        assert torch.allclose(loss_sum, loss_none.sum())
    
    def test_kl_divergence_gaussian(self, sample_latent_params):
        """Test KL divergence computation."""
        mu, logvar = sample_latent_params
        
        kl_loss = kl_divergence_gaussian(mu, logvar, "mean")
        
        # Should be non-negative (KL divergence property)
        assert kl_loss.item() >= 0
        
        # Test with standard normal (should be close to 0)
        mu_zero = torch.zeros_like(mu)
        logvar_zero = torch.zeros_like(logvar)
        kl_zero = kl_divergence_gaussian(mu_zero, logvar_zero, "mean")
        
        assert kl_zero.item() == pytest.approx(0.0, abs=1e-6)
        
        # Test analytical formula
        kl_analytical = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
        kl_function = kl_divergence_gaussian(mu, logvar, "mean")
        
        assert torch.allclose(kl_analytical, kl_function, atol=1e-6)
    
    def test_kl_divergence_reductions(self, sample_latent_params):
        """Test different reduction modes for KL divergence."""
        mu, logvar = sample_latent_params
        
        kl_mean = kl_divergence_gaussian(mu, logvar, "mean")
        kl_sum = kl_divergence_gaussian(mu, logvar, "sum")
        kl_none = kl_divergence_gaussian(mu, logvar, "none")
        
        # Check shapes
        assert kl_mean.shape == ()
        assert kl_sum.shape == ()
        assert kl_none.shape == (4,)  # Batch size
        
        # Relationships
        assert torch.allclose(kl_mean, kl_none.mean())
        assert torch.allclose(kl_sum, kl_none.sum())
    
    def test_elbo_loss(self, sample_images, sample_logits, sample_latent_params):
        """Test ELBO loss computation."""
        mu, logvar = sample_latent_params
        
        total_loss, loss_dict = elbo_loss(sample_logits, sample_images, mu, logvar, beta=1.0)
        
        # Check components
        assert "total_loss" in loss_dict
        assert "recon_loss" in loss_dict
        assert "kl_loss" in loss_dict
        assert "beta" in loss_dict
        assert "weighted_kl" in loss_dict
        
        # Total loss should equal recon + beta * kl
        expected_total = loss_dict["recon_loss"] + loss_dict["weighted_kl"]
        assert torch.allclose(total_loss, expected_total)
        
        # Beta should be correct
        assert loss_dict["beta"].item() == 1.0
    
    def test_elbo_loss_with_beta(self, sample_images, sample_logits, sample_latent_params):
        """Test ELBO loss with different beta values."""
        mu, logvar = sample_latent_params
        
        _, loss_dict_1 = elbo_loss(sample_logits, sample_images, mu, logvar, beta=1.0)
        _, loss_dict_2 = elbo_loss(sample_logits, sample_images, mu, logvar, beta=0.5)
        _, loss_dict_10 = elbo_loss(sample_logits, sample_images, mu, logvar, beta=10.0)
        
        # Reconstruction loss should be the same
        assert torch.allclose(loss_dict_1["recon_loss"], loss_dict_2["recon_loss"])
        assert torch.allclose(loss_dict_1["recon_loss"], loss_dict_10["recon_loss"])
        
        # KL loss should be the same
        assert torch.allclose(loss_dict_1["kl_loss"], loss_dict_2["kl_loss"])
        assert torch.allclose(loss_dict_1["kl_loss"], loss_dict_10["kl_loss"])
        
        # Weighted KL should be different
        expected_weighted_kl_2 = 0.5 * loss_dict_1["kl_loss"]
        expected_weighted_kl_10 = 10.0 * loss_dict_1["kl_loss"]
        
        assert torch.allclose(loss_dict_2["weighted_kl"], expected_weighted_kl_2)
        assert torch.allclose(loss_dict_10["weighted_kl"], expected_weighted_kl_10)
    
    def test_beta_scheduler_none(self):
        """Test beta scheduler with no scheduling."""
        scheduler = BetaScheduler(schedule_type="none", max_beta=1.0)
        
        for epoch in range(20):
            beta = scheduler.get_beta(epoch)
            assert beta == 1.0
    
    def test_beta_scheduler_linear(self):
        """Test linear beta scheduling."""
        scheduler = BetaScheduler(
            schedule_type="linear", max_beta=1.0, warmup_epochs=10, min_beta=0.0
        )
        
        # Test warmup phase
        assert scheduler.get_beta(0) == pytest.approx(0.0)
        assert scheduler.get_beta(5) == pytest.approx(0.5, abs=0.01)
        assert scheduler.get_beta(10) == pytest.approx(1.0, abs=0.01)
        
        # After warmup, should stay at max
        assert scheduler.get_beta(15) == 1.0
        assert scheduler.get_beta(20) == 1.0
    
    def test_beta_scheduler_cyclical(self):
        """Test cyclical beta scheduling."""
        scheduler = BetaScheduler(
            schedule_type="cyclical", max_beta=1.0, min_beta=0.0, 
            warmup_epochs=10, cycle_length=20
        )
        
        # Test one complete cycle
        betas = [scheduler.get_beta(epoch) for epoch in range(20)]
        
        # Should start at min, go to max, then back to min
        assert betas[0] == pytest.approx(0.0, abs=0.01)
        assert betas[10] == pytest.approx(1.0, abs=0.01)
        assert betas[19] == pytest.approx(0.0, abs=0.1)  # Close to min at end of cycle
    
    def test_free_bits_kl(self, sample_latent_params):
        """Test free bits KL implementation."""
        mu, logvar = sample_latent_params
        
        # Test with different free bits values
        free_bits_0 = free_bits_kl(mu, logvar, free_bits=0.0)
        free_bits_1 = free_bits_kl(mu, logvar, free_bits=1.0)
        regular_kl = kl_divergence_gaussian(mu, logvar)
        
        # Free bits = 0 should equal regular KL
        assert torch.allclose(free_bits_0, regular_kl, atol=1e-6)
        
        # Free bits > 0 should be >= regular KL
        assert free_bits_1.item() >= regular_kl.item() - 1e-6
    
    def test_loss_function_gradients(self, sample_images, sample_latent_params):
        """Test that loss functions produce proper gradients."""
        mu, logvar = sample_latent_params
        mu.requires_grad_(True)
        logvar.requires_grad_(True)
        
        # Create some dummy reconstructions
        reconstructions = torch.randn_like(sample_images, requires_grad=True)
        
        # Compute ELBO loss
        total_loss, _ = elbo_loss(reconstructions, sample_images, mu, logvar)
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist
        assert mu.grad is not None
        assert logvar.grad is not None
        assert reconstructions.grad is not None
        
        # Gradients should not be zero (unless by extreme coincidence)
        assert mu.grad.abs().sum().item() > 0
        assert logvar.grad.abs().sum().item() > 0
        assert reconstructions.grad.abs().sum().item() > 0
    
    def test_numerical_stability(self):
        """Test numerical stability of loss functions."""
        device = torch.device("cpu")
        
        # Test with extreme values
        mu_extreme = torch.tensor([[100.0, -100.0, 0.0]], device=device)
        logvar_extreme = torch.tensor([[-20.0, 20.0, 0.0]], device=device)
        
        # Should not produce NaN or inf
        kl_loss = kl_divergence_gaussian(mu_extreme, logvar_extreme)
        assert torch.isfinite(kl_loss).all()
        
        # Test reconstruction loss with extreme values
        targets = torch.rand(1, 1, 4, 4, device=device)
        logits_extreme = torch.tensor([[[[100.0, -100.0, 0.0, 50.0]]]], device=device).expand_as(targets)
        
        recon_loss = reconstruction_loss(logits_extreme, targets, "bce")
        assert torch.isfinite(recon_loss).all()


if __name__ == "__main__":
    pytest.main([__file__])