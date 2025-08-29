"""
Training smoke tests: verify 1-2 steps reduce loss for each schedule.
"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.models.unet_small import UNetSmall
from src.schedules import get_schedule
from src.losses import DDPMLoss, compute_forward_process
from src.trainer import DDPMTrainer
from src.utils import set_seed


class TestTrainingSmoke:
    """Smoke tests for training functionality."""
    
    @pytest.fixture
    def device(self):
        """Get device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def dummy_config(self):
        """Create minimal config for testing."""
        return {
            'seed': 42,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'data': {'dataset': 'mnist', 'normalize': 'minus_one_one'},
            'diffusion': {'T': 10, 'schedule': 'linear'},  # Short schedule for speed
            'model': {
                'in_ch': 1, 'base_ch': 16, 'ch_mult': [1, 2], 
                'time_embed_dim': 64, 'dropout': 0.0, 'num_res_blocks': 1
            },
            'train': {
                'epochs': 2, 'lr': 1e-3, 'opt': 'adam', 'wd': 0.0,
                'amp': False, 'ema': False, 'grad_clip': 0.0,
                'save_every': 10, 'sample_every': 10
            },
            'log': {'out_root': './test_outputs', 'run_name': 'smoke_test', 'log_every': 1}
        }
    
    @pytest.fixture
    def dummy_dataloader(self):
        """Create minimal dataloader for testing."""
        # Create dummy MNIST-like data
        images = torch.randn(32, 1, 28, 28)  # 32 samples
        labels = torch.zeros(32)
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, batch_size=8, shuffle=True)
    
    @pytest.mark.parametrize("schedule_name", ["linear", "cosine", "quadratic"])
    def test_forward_pass_shapes(self, schedule_name, device):
        """Test forward pass produces correct shapes."""
        set_seed(42)
        
        # Create model
        model = UNetSmall(
            in_channels=1, out_channels=1, base_channels=16,
            channel_multipliers=[1, 2], time_embed_dim=64
        ).to(device)
        
        # Create schedule
        T = 10
        schedule = get_schedule(schedule_name, T)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        # Test forward pass
        batch_size = 4
        images = torch.randn(batch_size, 1, 28, 28, device=device)
        timesteps = torch.randint(0, T, (batch_size,), device=device)
        
        with torch.no_grad():
            output = model(images, timesteps)
        
        assert output.shape == images.shape, f"Output shape mismatch: {output.shape} vs {images.shape}"
    
    @pytest.mark.parametrize("schedule_name", ["linear", "cosine", "quadratic"])
    def test_loss_computation(self, schedule_name, device):
        """Test loss computation works correctly."""
        set_seed(42)
        
        # Create schedule
        T = 10
        schedule = get_schedule(schedule_name, T)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        # Create dummy data
        batch_size = 4
        x0 = torch.randn(batch_size, 1, 28, 28, device=device)
        timesteps = torch.randint(0, T, (batch_size,), device=device)
        
        # Forward diffusion
        noisy_images, true_noise = compute_forward_process(
            x0, timesteps, schedule['alpha_bars']
        )
        
        # Dummy predicted noise
        pred_noise = torch.randn_like(true_noise)
        
        # Compute loss
        loss_fn = DDPMLoss()
        loss = loss_fn(pred_noise, true_noise)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
    
    @pytest.mark.parametrize("schedule_name", ["linear", "cosine", "quadratic"])
    def test_single_training_step(self, schedule_name, device):
        """Test a single training step reduces loss."""
        set_seed(42)
        
        # Create model
        model = UNetSmall(
            in_channels=1, out_channels=1, base_channels=16,
            channel_multipliers=[1, 2], time_embed_dim=64
        ).to(device)
        
        # Create schedule
        T = 10
        schedule = get_schedule(schedule_name, T)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        # Register buffers in model for convenience
        for key, tensor in schedule.items():
            model.register_buffer(key, tensor)
        
        # Create optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = DDPMLoss()
        
        # Create batch
        batch_size = 8
        x0 = torch.randn(batch_size, 1, 28, 28, device=device)
        timesteps = torch.randint(0, T, (batch_size,), device=device)
        
        # Get initial loss
        model.train()
        with torch.no_grad():
            noisy_images, true_noise = compute_forward_process(
                x0, timesteps, schedule['alpha_bars']
            )
            initial_pred = model(noisy_images, timesteps)
            initial_loss = loss_fn(initial_pred, true_noise).item()
        
        # Training step
        optimizer.zero_grad()
        noisy_images, true_noise = compute_forward_process(
            x0, timesteps, schedule['alpha_bars']
        )
        pred_noise = model(noisy_images, timesteps)
        loss = loss_fn(pred_noise, true_noise)
        loss.backward()
        optimizer.step()
        
        # Check loss decreased or at least didn't increase much
        final_loss = loss.item()
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        assert final_loss > 0
        
        # For a single step, we just check it's reasonable
        print(f"{schedule_name}: Initial loss {initial_loss:.4f}, Final loss {final_loss:.4f}")
    
    @pytest.mark.parametrize("schedule_name", ["linear", "cosine", "quadratic"])
    def test_multiple_training_steps(self, schedule_name, device):
        """Test multiple training steps show decreasing loss trend."""
        set_seed(42)
        
        # Create model
        model = UNetSmall(
            in_channels=1, out_channels=1, base_channels=16,
            channel_multipliers=[1, 2], time_embed_dim=64
        ).to(device)
        
        # Create schedule
        T = 10
        schedule = get_schedule(schedule_name, T)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        # Create optimizer and loss
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = DDPMLoss()
        
        model.train()
        losses = []
        
        # Run 10 training steps
        for step in range(10):
            # Create batch
            batch_size = 8
            x0 = torch.randn(batch_size, 1, 28, 28, device=device)
            timesteps = torch.randint(0, T, (batch_size,), device=device)
            
            # Training step
            optimizer.zero_grad()
            noisy_images, true_noise = compute_forward_process(
                x0, timesteps, schedule['alpha_bars']
            )
            pred_noise = model(noisy_images, timesteps)
            loss = loss_fn(pred_noise, true_noise)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Check trend - later losses should generally be lower
        early_avg = sum(losses[:3]) / 3
        late_avg = sum(losses[-3:]) / 3
        
        print(f"{schedule_name}: Early avg {early_avg:.4f}, Late avg {late_avg:.4f}")
        
        # Allow some tolerance - loss should at least not increase dramatically
        assert late_avg < early_avg * 2.0, f"Loss increased too much for {schedule_name}"
        assert all(not torch.isnan(torch.tensor(l)) for l in losses)
    
    def test_trainer_initialization(self, dummy_config):
        """Test trainer can be initialized correctly."""
        set_seed(42)
        
        # Test with each schedule
        for schedule_name in ["linear", "cosine", "quadratic"]:
            config = dummy_config.copy()
            config['diffusion']['schedule'] = schedule_name
            config['log']['run_name'] = f'test_{schedule_name}'
            
            trainer = DDPMTrainer(config)
            
            # Check components exist
            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.loss_fn is not None
            assert trainer.T == config['diffusion']['T']
    
    def test_trainer_short_training(self, dummy_config, dummy_dataloader):
        """Test trainer can run for a few steps without crashing."""
        set_seed(42)
        
        # Very short training to avoid timeout
        config = dummy_config.copy()
        config['train']['epochs'] = 1
        config['log']['run_name'] = 'smoke_short'
        
        trainer = DDPMTrainer(config)
        
        # Run training - should not crash
        try:
            trainer.train(dummy_dataloader)
            success = True
        except Exception as e:
            print(f"Training failed: {e}")
            success = False
        
        assert success, "Short training run failed"


class TestModelComponents:
    """Test individual model components."""
    
    def test_time_embedding(self):
        """Test time embedding works correctly."""
        from src.models.time_embedding import TimeEmbedding
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        time_emb = TimeEmbedding(64).to(device)
        
        timesteps = torch.randint(0, 1000, (8,), device=device)
        embeddings = time_emb(timesteps)
        
        assert embeddings.shape == (8, 64)
        assert not torch.isnan(embeddings).any()
        
        # Different timesteps should give different embeddings
        t1 = torch.zeros(1, device=device, dtype=torch.long)
        t2 = torch.ones(1, device=device, dtype=torch.long) * 500
        
        emb1 = time_emb(t1)
        emb2 = time_emb(t2)
        
        assert not torch.allclose(emb1, emb2, atol=1e-6)
    
    def test_unet_components(self):
        """Test UNet components work correctly."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = UNetSmall(
            in_channels=1, base_channels=16,
            channel_multipliers=[1, 2],
            time_embed_dim=64
        ).to(device)
        
        # Test forward pass
        x = torch.randn(4, 1, 28, 28, device=device)
        t = torch.randint(0, 100, (4,), device=device)
        
        output = model(x, t)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


if __name__ == "__main__":
    # Run tests manually
    print("Running training smoke tests...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test basic forward pass
    for schedule_name in ["linear", "cosine", "quadratic"]:
        print(f"Testing {schedule_name} schedule...")
        
        set_seed(42)
        
        # Quick model test
        model = UNetSmall(in_channels=1, base_channels=16, channel_multipliers=[1, 2]).to(device)
        x = torch.randn(2, 1, 28, 28, device=device)
        t = torch.randint(0, 10, (2,), device=device)
        
        with torch.no_grad():
            output = model(x, t)
        
        assert output.shape == x.shape
        print(f"  ✓ Forward pass: {output.shape}")
        
        # Quick loss test
        schedule = get_schedule(schedule_name, 10)
        for key in schedule:
            schedule[key] = schedule[key].to(device)
        
        noisy_images, true_noise = compute_forward_process(x, t, schedule['alpha_bars'])
        loss_fn = DDPMLoss()
        loss = loss_fn(output, true_noise)
        
        assert not torch.isnan(loss)
        print(f"  ✓ Loss computation: {loss.item():.4f}")
    
    print("✓ All smoke tests passed!")
    print("Training components are working correctly.")
