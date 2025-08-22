"""
Test suite for training functionality.
Tests that training step decreases ELBO and other training utilities.
"""

import pytest
import torch
import torch.optim as optim

from src.models.vae_conv import VAEConv
from src.losses import elbo_loss, BetaScheduler
from src.utils import EMA, MetricsTracker


class TestTrainingStep:
    """Test training step and utilities."""
    
    @pytest.fixture
    def device(self):
        return torch.device("cpu")
    
    @pytest.fixture
    def model(self, device):
        """Small VAE model for testing."""
        model = VAEConv(in_channels=1, latent_dim=8, base_channels=8).to(device)
        return model
    
    @pytest.fixture
    def sample_batch(self, device):
        """Sample batch for training."""
        return torch.randn(2, 1, 28, 28, device=device)
    
    @pytest.fixture
    def optimizer(self, model):
        """Adam optimizer for testing."""
        return optim.Adam(model.parameters(), lr=1e-3)
    
    def test_single_training_step_decreases_loss(self, model, sample_batch, optimizer):
        """Test that a single training step decreases the loss."""
        model.train()
        
        # Initial forward pass
        x_hat_initial, mu_initial, logvar_initial = model(sample_batch)
        initial_loss, _ = elbo_loss(x_hat_initial, sample_batch, mu_initial, logvar_initial)
        initial_loss_value = initial_loss.item()
        
        # Training step
        optimizer.zero_grad()
        
        x_hat, mu, logvar = model(sample_batch)
        loss, loss_dict = elbo_loss(x_hat, sample_batch, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        # Second forward pass (after update)
        with torch.no_grad():
            x_hat_after, mu_after, logvar_after = model(sample_batch)
            loss_after, _ = elbo_loss(x_hat_after, sample_batch, mu_after, logvar_after)
            loss_after_value = loss_after.item()
        
        # Loss should decrease (or at least not increase significantly)
        # Note: With such a small model and batch, there might be some variance
        improvement_ratio = (initial_loss_value - loss_after_value) / abs(initial_loss_value)
        
        # At minimum, loss shouldn't increase by more than 10%
        assert improvement_ratio > -0.1, f"Loss increased too much: {initial_loss_value} -> {loss_after_value}"
    
    def test_multiple_training_steps(self, model, sample_batch, optimizer):
        """Test multiple training steps show improvement."""
        model.train()
        
        losses = []
        
        for step in range(10):
            optimizer.zero_grad()
            
            x_hat, mu, logvar = model(sample_batch)
            loss, _ = elbo_loss(x_hat, sample_batch, mu, logvar)
            
            losses.append(loss.item())
            
            loss.backward()
            optimizer.step()
        
        # Loss should generally trend downward over multiple steps
        # Check that the average of the last 3 losses is lower than the first 3
        early_loss_avg = sum(losses[:3]) / 3
        late_loss_avg = sum(losses[-3:]) / 3
        
        assert late_loss_avg < early_loss_avg * 1.1, f"Training not converging: {early_loss_avg} -> {late_loss_avg}"
    
    def test_training_with_beta_scheduling(self, model, sample_batch, optimizer):
        """Test training with beta scheduling."""
        model.train()
        scheduler = BetaScheduler("linear", max_beta=1.0, warmup_epochs=5)
        
        losses = []
        kl_losses = []
        
        for epoch in range(10):
            beta = scheduler.get_beta(epoch)
            
            optimizer.zero_grad()
            
            x_hat, mu, logvar = model(sample_batch)
            loss, loss_dict = elbo_loss(x_hat, sample_batch, mu, logvar, beta=beta)
            
            losses.append(loss.item())
            kl_losses.append(loss_dict["kl_loss"].item())
            
            loss.backward()
            optimizer.step()
        
        # Beta should increase during warmup
        beta_0 = scheduler.get_beta(0)
        beta_5 = scheduler.get_beta(5)
        beta_9 = scheduler.get_beta(9)
        
        assert beta_0 <= beta_5 <= beta_9
        
        # Training should still converge
        assert losses[-1] < losses[0] * 2  # Allow some tolerance
    
    def test_ema_functionality(self, model, sample_batch, optimizer):
        """Test Exponential Moving Average functionality."""
        model.train()
        ema = EMA(model, decay=0.9)
        
        # Initial parameters
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Training step
        optimizer.zero_grad()
        x_hat, mu, logvar = model(sample_batch)
        loss, _ = elbo_loss(x_hat, sample_batch, mu, logvar)
        loss.backward()
        optimizer.step()
        
        # Update EMA
        ema.update(model)
        
        # EMA should be different from current model
        current_params = {name: param for name, param in model.named_parameters()}
        
        # Apply EMA
        ema.apply_shadow(model)
        ema_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Restore original
        ema.restore(model)
        restored_params = {name: param for name, param in model.named_parameters()}
        
        # Check that EMA is different from current but restoration works
        for name in current_params:
            # Current should be different from initial (due to training)
            assert not torch.allclose(current_params[name], initial_params[name], atol=1e-6)
            
            # EMA should be different from current (but closer to initial)
            assert not torch.allclose(ema_params[name], current_params[name], atol=1e-6)
            
            # Restored should equal current
            assert torch.allclose(restored_params[name], current_params[name], atol=1e-8)
    
    def test_metrics_tracker(self):
        """Test MetricsTracker functionality."""
        tracker = MetricsTracker()
        
        # Update with some metrics
        tracker.update({"loss": 1.0, "accuracy": 0.8}, batch_size=32)
        tracker.update({"loss": 0.8, "accuracy": 0.9}, batch_size=16)
        
        # Get averages
        averages = tracker.get_averages()
        
        # Should compute weighted averages correctly
        expected_loss = (1.0 * 32 + 0.8 * 16) / (32 + 16)
        expected_accuracy = (0.8 * 32 + 0.9 * 16) / (32 + 16)
        
        assert averages["loss"] == pytest.approx(expected_loss)
        assert averages["accuracy"] == pytest.approx(expected_accuracy)
        
        # Test reset
        tracker.reset()
        averages_after_reset = tracker.get_averages()
        assert len(averages_after_reset) == 0
    
    def test_gradient_clipping(self, model, sample_batch, optimizer):
        """Test gradient clipping functionality."""
        model.train()
        
        # Create a loss that might produce large gradients
        x_hat, mu, logvar = model(sample_batch)
        
        # Scale up the loss artificially to create large gradients
        loss, _ = elbo_loss(x_hat, sample_batch, mu, logvar)
        scaled_loss = loss * 1000  # Large scaling
        
        optimizer.zero_grad()
        scaled_loss.backward()
        
        # Check gradients before clipping
        grad_norms_before = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms_before.append(param.grad.norm().item())
        
        max_grad_before = max(grad_norms_before) if grad_norms_before else 0
        
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Check gradients after clipping
        grad_norms_after = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms_after.append(param.grad.norm().item())
        
        max_grad_after = max(grad_norms_after) if grad_norms_after else 0
        
        # If there were large gradients, they should be clipped
        if max_grad_before > 1.0:
            assert max_grad_after <= 1.01  # Allow small numerical tolerance
    
    def test_model_eval_mode_deterministic(self, model, sample_batch):
        """Test that model outputs are deterministic in eval mode."""
        model.eval()
        
        with torch.no_grad():
            # Multiple forward passes should give same reconstruction
            recon1 = model.reconstruct(sample_batch)
            recon2 = model.reconstruct(sample_batch)
            
            assert torch.allclose(recon1, recon2, atol=1e-6)
    
    def test_model_train_mode_stochastic(self, model, sample_batch):
        """Test that model outputs are stochastic in train mode."""
        model.train()
        
        # Multiple forward passes should give different results due to reparameterization
        output1 = model(sample_batch)
        output2 = model(sample_batch)
        
        # Reconstructions should be different (due to sampling)
        assert not torch.allclose(output1[0], output2[0], atol=1e-4)
    
    def test_loss_components_reasonable_magnitudes(self, model, sample_batch):
        """Test that loss components have reasonable magnitudes."""
        model.eval()
        
        with torch.no_grad():
            x_hat, mu, logvar = model(sample_batch)
            loss, loss_dict = elbo_loss(x_hat, sample_batch, mu, logvar)
        
        # All losses should be finite and positive
        assert torch.isfinite(loss_dict["total_loss"]).all()
        assert torch.isfinite(loss_dict["recon_loss"]).all()
        assert torch.isfinite(loss_dict["kl_loss"]).all()
        
        assert loss_dict["total_loss"].item() > 0
        assert loss_dict["recon_loss"].item() > 0
        assert loss_dict["kl_loss"].item() >= 0  # KL can be 0
        
        # Reconstruction loss should dominate early in training
        # (This is a rough heuristic)
        assert loss_dict["recon_loss"].item() > loss_dict["kl_loss"].item() * 0.01


if __name__ == "__main__":
    pytest.main([__file__])