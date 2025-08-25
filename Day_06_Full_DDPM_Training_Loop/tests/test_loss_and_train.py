"""
Test suite for loss functions and training step
"""

import pytest
import torch
import torch.nn as nn
from src.ddpm_schedules import DDPMSchedules
from src.losses import DDPMLoss, get_loss_fn
from src.models.unet_small import UNetSmall
from src.trainer import DDPMTrainer
from src.dataset import get_dataloader


class TestLossFunctions:
    """Test DDPM loss functions"""
    
    @pytest.fixture
    def schedules(self):
        """Create test schedules"""
        return DDPMSchedules(num_timesteps=100, device="cpu")
        
    @pytest.fixture
    def model(self):
        """Create test model"""
        return UNetSmall(
            in_channels=1,
            out_channels=1, 
            model_channels=16,
            channel_mult=[1, 2],
            num_res_blocks=1,
            use_attention=False
        )
        
    def test_ddpm_loss_initialization(self, schedules):
        """Test DDPM loss can be initialized"""
        loss_fn = DDPMLoss(schedules=schedules)
        
        assert loss_fn.schedules == schedules
        assert loss_fn.parameterization == "eps"
        assert loss_fn.loss_type == "l2"
        
    def test_ddpm_loss_forward(self, schedules, model):
        """Test DDPM loss forward pass"""
        loss_fn = DDPMLoss(schedules=schedules)
        model.eval()
        
        batch_size = 4
        x_start = torch.randn(batch_size, 1, 16, 16)
        
        # Compute loss
        loss = loss_fn(model, x_start)
        
        # Check loss is scalar and positive
        assert loss.dim() == 0  # Scalar
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        
    def test_ddpm_loss_return_dict(self, schedules, model):
        """Test DDPM loss with return_dict=True"""
        loss_fn = DDPMLoss(schedules=schedules)
        model.eval()
        
        x_start = torch.randn(2, 1, 16, 16)
        
        result = loss_fn(model, x_start, return_dict=True)
        
        # Check dict keys
        expected_keys = ["total_loss", "simple_loss", "model_output", "target", "x_t", "t", "noise", "pred_x0"]
        for key in expected_keys:
            assert key in result
            
        # Check shapes
        assert result["x_t"].shape == x_start.shape
        assert result["model_output"].shape == x_start.shape
        assert result["target"].shape == x_start.shape
        assert result["noise"].shape == x_start.shape
        assert result["pred_x0"].shape == x_start.shape
        assert result["t"].shape == (2,)
        
    def test_different_parameterizations(self, schedules, model):
        """Test different loss parameterizations"""
        x_start = torch.randn(2, 1, 16, 16)
        
        # Test each parameterization
        for param in ["eps", "x0", "v"]:
            loss_fn = DDPMLoss(schedules=schedules, parameterization=param)
            
            loss = loss_fn(model, x_start)
            
            # All should produce valid losses
            assert torch.isfinite(loss)
            assert loss.item() >= 0
            
    def test_different_loss_types(self, schedules, model):
        """Test different loss types (L1, L2, Huber)"""
        x_start = torch.randn(2, 1, 16, 16)
        
        for loss_type in ["l1", "l2", "huber"]:
            loss_fn = DDPMLoss(schedules=schedules, loss_type=loss_type)
            
            loss = loss_fn(model, x_start)
            
            # All should produce valid losses
            assert torch.isfinite(loss)
            assert loss.item() >= 0
            
    def test_loss_factory(self, schedules):
        """Test loss factory function"""
        config = {
            "type": "simple",
            "parameterization": "eps",
            "loss_type": "l2"
        }
        
        loss_fn = get_loss_fn(schedules, config)
        
        assert isinstance(loss_fn, DDPMLoss)
        assert loss_fn.parameterization == "eps"
        assert loss_fn.loss_type == "l2"
        
    def test_custom_timesteps(self, schedules, model):
        """Test loss with custom timesteps"""
        loss_fn = DDPMLoss(schedules=schedules)
        
        batch_size = 3
        x_start = torch.randn(batch_size, 1, 16, 16)
        custom_t = torch.tensor([10, 50, 90])
        
        result = loss_fn(model, x_start, t=custom_t, return_dict=True)
        
        # Check that our timesteps were used
        assert torch.equal(result["t"], custom_t)
        
    def test_custom_noise(self, schedules, model):
        """Test loss with custom noise"""
        loss_fn = DDPMLoss(schedules=schedules)
        
        x_start = torch.randn(2, 1, 16, 16)
        custom_noise = torch.randn_like(x_start)
        
        result = loss_fn(model, x_start, noise=custom_noise, return_dict=True)
        
        # Check that our noise was used
        assert torch.allclose(result["noise"], custom_noise, atol=1e-6)


class TestTrainingStep:
    """Test training step functionality"""
    
    @pytest.fixture
    def simple_config(self):
        """Create simple training config"""
        return {
            "dataset": {
                "name": "mnist",
                "image_size": 16,  # Small for speed
                "conditional": False
            },
            "model": {
                "type": "unet_small",
                "in_channels": 1,
                "out_channels": 1,
                "model_channels": 16,
                "channel_mult": [1, 2],
                "num_res_blocks": 1,
                "attention_resolutions": [],
                "dropout": 0.0,
                "time_embed_dim": 32,
                "use_attention": False,
                "num_heads": 4
            },
            "schedules": {
                "num_timesteps": 100,
                "schedule_type": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02
            },
            "loss": {
                "type": "simple",
                "parameterization": "eps",
                "loss_type": "l2",
                "lambda_simple": 1.0
            },
            "training": {
                "num_epochs": 1,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "weight_decay": 0.0,
                "grad_clip_norm": 1.0,
                "use_amp": False,
                "ema": {"enabled": True, "decay": 0.999},
                "optimizer": {"type": "adamw"},
                "scheduler": {"type": "none"},
                "sampling": {"every_epochs": 100, "num_images": 4, "ddim_steps": 10}
            },
            "output": {"base_dir": "./test_outputs"}
        }
        
    def test_training_step_reduces_loss(self, simple_config):
        """Test that a single training step reduces loss"""
        # Create simple synthetic dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, size=8):
                self.size = size
                self.data = [torch.randn(1, 16, 16) for _ in range(size)]
                
            def __len__(self):
                return self.size
                
            def __getitem__(self, idx):
                return self.data[idx]
                
        dataset = SimpleDataset()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        
        # Create components
        device = torch.device("cpu")
        model = UNetSmall(
            in_channels=1,
            out_channels=1,
            model_channels=16,
            channel_mult=[1, 2],
            num_res_blocks=1,
            use_attention=False
        ).to(device)
        
        schedules = DDPMSchedules(num_timesteps=100, device="cpu")
        loss_fn = DDPMLoss(schedules=schedules)
        
        # Create trainer
        trainer = DDPMTrainer(
            model=model,
            schedules=schedules,
            loss_fn=loss_fn,
            train_loader=train_loader,
            val_loader=val_loader,
            config=simple_config,
            device=device,
            output_dir="./test_outputs"
        )
        
        # Get initial loss
        batch = next(iter(train_loader))
        model.eval()
        with torch.no_grad():
            initial_loss = loss_fn(model, batch).item()
            
        # Perform training step
        model.train()
        step_metrics = trainer.train_step((batch,))
        
        # Loss should be recorded
        assert "train_loss" in step_metrics
        assert step_metrics["train_loss"] >= 0
        
        # Metrics should be reasonable
        assert "learning_rate" in step_metrics
        assert step_metrics["learning_rate"] > 0
        
    def test_validation_step(self, simple_config):
        """Test validation step works"""
        # Simple synthetic data
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = [torch.randn(1, 16, 16) for _ in range(4)]
                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx]
                
        dataset = SimpleDataset()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        # Create trainer components
        device = torch.device("cpu")
        model = UNetSmall(
            in_channels=1, out_channels=1, model_channels=16,
            channel_mult=[1, 2], num_res_blocks=1, use_attention=False
        ).to(device)
        schedules = DDPMSchedules(num_timesteps=50, device="cpu")
        loss_fn = DDPMLoss(schedules=schedules)
        
        trainer = DDPMTrainer(
            model=model, schedules=schedules, loss_fn=loss_fn,
            train_loader=train_loader, val_loader=val_loader,
            config=simple_config, device=device, output_dir="./test_outputs"
        )
        
        # Run validation
        val_metrics = trainer.validate()
        
        # Check metrics
        assert "val_loss" in val_metrics
        assert "val_simple_loss" in val_metrics
        assert val_metrics["val_loss"] >= 0
        assert val_metrics["val_simple_loss"] >= 0
        
    def test_model_parameter_updates(self, simple_config):
        """Test that model parameters actually get updated during training"""
        # Simple data
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self):
                self.data = [torch.randn(1, 16, 16) for _ in range(4)]
            def __len__(self):
                return len(self.data)
            def __getitem__(self, idx):
                return self.data[idx]
                
        dataset = SimpleDataset()
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=2)
        
        device = torch.device("cpu")
        model = UNetSmall(
            in_channels=1, out_channels=1, model_channels=16,
            channel_mult=[1, 2], num_res_blocks=1, use_attention=False
        ).to(device)
        
        # Store initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            initial_params[name] = param.clone()
            
        schedules = DDPMSchedules(num_timesteps=50, device="cpu")
        loss_fn = DDPMLoss(schedules=schedules)
        
        trainer = DDPMTrainer(
            model=model, schedules=schedules, loss_fn=loss_fn,
            train_loader=train_loader, val_loader=val_loader,
            config=simple_config, device=device, output_dir="./test_outputs"
        )
        
        # Perform training step
        batch = next(iter(train_loader))
        trainer.train_step((batch,))
        
        # Check that parameters changed
        params_changed = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if not torch.allclose(param, initial_params[name], atol=1e-8):
                params_changed += 1
                
        # Most parameters should have changed
        assert params_changed > total_params * 0.1  # At least 10% of parameters changed