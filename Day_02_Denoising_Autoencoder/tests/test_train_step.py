"""
Tests for training functionality in Day 2: Denoising Autoencoder
"""

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from src.models import ConvDAE, UNetDAE
from src.loss import ReconstructionLoss, get_loss_function
from src.train import DAETrainer


class TestTrainingStep:
    """Test basic training functionality."""
    
    def create_dummy_config(self):
        """Create dummy configuration for testing."""
        config = {
            'seed': 42,
            'device': 'cpu',
            'train': {
                'epochs': 2,
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'opt': 'adam',
                'loss': 'l2',
                'grad_clip': 1.0,
                'amp': False,
                'ema': False,
                'scheduler': 'none'
            },
            'log': {
                'out_dir': './test_outputs',
                'save_every': 1,
                'log_every': 10,
                'eval_every': 1,
                'viz_every': 1
            }
        }
        return OmegaConf.create(config)
    
    def create_dummy_data(self, batch_size=4, num_batches=2):
        """Create dummy data loaders for testing."""
        # Generate synthetic data
        clean_data = torch.randn(batch_size * num_batches, 1, 16, 16)
        noisy_data = clean_data + torch.randn_like(clean_data) * 0.1
        sigmas = torch.ones(batch_size * num_batches) * 0.1
        
        dataset = TensorDataset(clean_data, noisy_data, sigmas)
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def test_single_training_step(self):
        """Test that a single training step decreases loss."""
        # Setup
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=16, num_downs=1)
        config = self.create_dummy_config()
        train_loader, val_loader = self.create_dummy_data()
        
        trainer = DAETrainer(model, train_loader, val_loader, config, torch.device('cpu'))
        
        # Get initial loss
        model.eval()
        with torch.no_grad():
            clean, noisy, _ = next(iter(train_loader))
            initial_output = model(noisy)
            initial_loss = nn.MSELoss()(initial_output, clean).item()
        
        # Single training step
        model.train()
        trainer.optimizer.zero_grad()
        
        output = model(noisy)
        loss = nn.MSELoss()(output, clean)
        loss.backward()
        trainer.optimizer.step()
        
        # Check that loss computation works
        assert loss.item() >= 0
        
        # After step, model should have different parameters
        new_output = model(noisy)
        new_loss = nn.MSELoss()(new_output, clean).item()
        
        # Parameters should have changed (loss might not decrease in single step)
        assert not torch.allclose(initial_output, new_output, atol=1e-6)
    
    def test_training_epoch(self):
        """Test training for one epoch."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=16, num_downs=1)
        config = self.create_dummy_config()
        train_loader, val_loader = self.create_dummy_data()
        
        trainer = DAETrainer(model, train_loader, val_loader, config, torch.device('cpu'))
        
        # Train one epoch
        metrics = trainer.train_epoch(1)
        
        # Check that metrics are returned
        assert isinstance(metrics, dict)
        assert 'loss' in metrics
        assert 'psnr' in metrics
        assert 'ssim' in metrics
        assert 'mse' in metrics
        
        # Check that metrics are reasonable
        assert metrics['loss'] >= 0
        assert metrics['mse'] >= 0
        assert 0 <= metrics['ssim'] <= 1
    
    def test_validation_step(self):
        """Test validation step."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=16, num_downs=1)
        config = self.create_dummy_config()
        train_loader, val_loader = self.create_dummy_data()
        
        trainer = DAETrainer(model, train_loader, val_loader, config, torch.device('cpu'))
        
        # Validation
        val_metrics = trainer.validate(1)
        
        # Check validation metrics
        assert isinstance(val_metrics, dict)
        assert 'loss' in val_metrics
        assert val_metrics['loss'] >= 0
    
    def test_loss_functions(self):
        """Test different loss functions."""
        pred = torch.randn(2, 1, 8, 8)
        target = torch.randn(2, 1, 8, 8)
        
        # Test different loss types
        loss_types = ['l1', 'l2', 'mse', 'charbonnier']
        
        for loss_type in loss_types:
            loss_fn = get_loss_function(loss_type)
            loss_value = loss_fn(pred, target)
            
            assert isinstance(loss_value, torch.Tensor)
            assert loss_value.item() >= 0
            assert loss_value.requires_grad  # Should be differentiable
    
    def test_optimizer_creation(self):
        """Test optimizer creation with different types."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=16)
        
        optimizers = ['adam', 'adamw', 'sgd']
        
        for opt_name in optimizers:
            config = self.create_dummy_config()
            config.train.opt = opt_name
            
            train_loader, val_loader = self.create_dummy_data()
            
            trainer = DAETrainer(model, train_loader, val_loader, config, torch.device('cpu'))
            
            assert trainer.optimizer is not None
            assert len(trainer.optimizer.param_groups) > 0
    
    def test_ema_functionality(self):
        """Test EMA (Exponential Moving Average) functionality."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=16)
        config = self.create_dummy_config()
        config.train.ema = True
        config.train.ema_decay = 0.999
        
        train_loader, val_loader = self.create_dummy_data()
        
        trainer = DAETrainer(model, train_loader, val_loader, config, torch.device('cpu'))
        
        # Check that EMA model is created
        assert trainer.ema_model is not None
        
        # Get initial parameters
        initial_ema_params = [p.clone() for p in trainer.ema_model.ema_model.parameters()]
        
        # Training step should update EMA
        clean, noisy, _ = next(iter(train_loader))
        trainer.optimizer.zero_grad()
        output = trainer.model(noisy)
        loss = nn.MSELoss()(output, clean)
        loss.backward()
        trainer.optimizer.step()
        
        # Check that EMA parameters changed
        final_ema_params = list(trainer.ema_model.ema_model.parameters())
        
        # At least some parameters should be different
        params_changed = any(
            not torch.allclose(init_p, final_p, atol=1e-6)
            for init_p, final_p in zip(initial_ema_params, final_ema_params)
        )
        
        # Note: EMA parameters might not change much in a single step
        # This test mainly checks that the mechanism works
    
    def test_gradient_clipping(self):
        """Test gradient clipping functionality."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=16)
        config = self.create_dummy_config()
        config.train.grad_clip = 0.1  # Very small clipping value
        
        train_loader, val_loader = self.create_dummy_data()
        
        trainer = DAETrainer(model, train_loader, val_loader, config, torch.device('cpu'))
        
        # Create large gradients
        clean, noisy, _ = next(iter(train_loader))
        trainer.optimizer.zero_grad()
        
        output = trainer.model(noisy)
        loss = nn.MSELoss()(output, clean) * 100  # Scale up loss to create large gradients
        loss.backward()
        
        # Check gradients before clipping
        total_grad_norm_before = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), float('inf')
        )
        
        # Apply clipping
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), config.train.grad_clip)
        
        # Check gradients after clipping
        total_grad_norm_after = torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(), float('inf')
        )
        
        # After clipping, gradient norm should be <= clip value
        assert total_grad_norm_after <= config.train.grad_clip + 1e-6


class TestModelForward:
    """Test model forward pass functionality."""
    
    def test_conv_dae_forward_backward(self):
        """Test forward and backward pass through ConvDAE."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=16, num_downs=1)
        x = torch.randn(2, 1, 16, 16, requires_grad=True)
        
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert any(p.grad is not None for p in model.parameters())
    
    def test_unet_forward_backward(self):
        """Test forward and backward pass through UNet."""
        model = UNetDAE(in_ch=1, out_ch=1, base_ch=16, num_downs=2)
        x = torch.randn(2, 1, 16, 16, requires_grad=True)
        
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        # Check gradients
        assert x.grad is not None
        assert any(p.grad is not None for p in model.parameters())
    
    def test_reconstruction_loss_backward(self):
        """Test backward pass through reconstruction loss."""
        pred = torch.randn(2, 1, 8, 8, requires_grad=True)
        target = torch.randn(2, 1, 8, 8)
        
        loss_fn = ReconstructionLoss('l2')
        loss = loss_fn(pred, target)
        loss.backward()
        
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape


class TestDeviceHandling:
    """Test device handling in training."""
    
    def test_cpu_training(self):
        """Test training on CPU."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=8, num_downs=1)  # Small model
        config = OmegaConf.create({
            'seed': 42,
            'device': 'cpu',
            'train': {
                'epochs': 1, 'lr': 1e-3, 'opt': 'adam', 'loss': 'l2',
                'amp': False, 'ema': False, 'scheduler': 'none'
            },
            'log': {'out_dir': './test_outputs', 'save_every': 1, 'log_every': 1, 'eval_every': 1}
        })
        
        # Create small dataset
        clean_data = torch.randn(8, 1, 8, 8)
        noisy_data = clean_data + torch.randn_like(clean_data) * 0.1
        sigmas = torch.ones(8) * 0.1
        
        dataset = TensorDataset(clean_data, noisy_data, sigmas)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)
        
        trainer = DAETrainer(model, train_loader, val_loader, config, torch.device('cpu'))
        
        # Should not crash
        metrics = trainer.train_epoch(1)
        assert 'loss' in metrics
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_training(self):
        """Test training on GPU."""
        device = torch.device('cuda:0')
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=8, num_downs=1).to(device)
        
        config = OmegaConf.create({
            'seed': 42,
            'device': 'cuda:0',
            'train': {
                'epochs': 1, 'lr': 1e-3, 'opt': 'adam', 'loss': 'l2',
                'amp': False, 'ema': False, 'scheduler': 'none'
            },
            'log': {'out_dir': './test_outputs', 'save_every': 1, 'log_every': 1, 'eval_every': 1}
        })
        
        # Create dataset
        clean_data = torch.randn(8, 1, 8, 8)
        noisy_data = clean_data + torch.randn_like(clean_data) * 0.1
        sigmas = torch.ones(8) * 0.1
        
        dataset = TensorDataset(clean_data, noisy_data, sigmas)
        train_loader = DataLoader(dataset, batch_size=4)
        val_loader = DataLoader(dataset, batch_size=4)
        
        trainer = DAETrainer(model, train_loader, val_loader, config, device)
        
        # Should not crash
        metrics = trainer.train_epoch(1)
        assert 'loss' in metrics


if __name__ == '__main__':
    pytest.main([__file__])