"""
Test compatibility with Day 6 checkpoints.
Verifies that models can load Day 6 weights and perform forward passes.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models.unet_small import UNetSmall, create_unet_small
from src.models.time_embedding import TimestepEmbedder
from src.checkpoints import CheckpointManager, load_model_checkpoint
from src.ddpm_schedules import DDPMSchedules


def create_mock_day6_checkpoint(model_config, device='cpu'):
    """Create a mock Day 6 checkpoint for testing."""
    # Create model
    model = create_unet_small({'model': model_config})
    
    # Create mock checkpoint with various possible formats
    checkpoint_variants = {
        'model': model.state_dict(),  # Standard format
        'model_ema': model.state_dict(),  # EMA format
        'epoch': 42,
        'step': 10000,
        'loss': 0.123,
        'model_config': model_config
    }
    
    return checkpoint_variants


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory with mock checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # MNIST model config
        mnist_config = {
            'in_ch': 1,
            'base_ch': 32,
            'ch_mult': [1, 2],
            'time_embed_dim': 128
        }
        
        # CIFAR-10 model config
        cifar_config = {
            'in_ch': 3,
            'base_ch': 64,
            'ch_mult': [1, 2, 2],
            'time_embed_dim': 256
        }
        
        # Create mock checkpoints
        checkpoints_to_create = [
            ('epoch_10.pt', mnist_config),
            ('epoch_30.pt', mnist_config),
            ('ema.pt', mnist_config),
            ('cifar_epoch_20.pt', cifar_config)
        ]
        
        for filename, config in checkpoints_to_create:
            ckpt_data = create_mock_day6_checkpoint(config)
            torch.save(ckpt_data, tmpdir / filename)
        
        yield tmpdir


def test_load_standard_checkpoint(temp_checkpoint_dir):
    """Test loading standard model checkpoint format."""
    model_config = {
        'in_ch': 1,
        'base_ch': 32,
        'ch_mult': [1, 2],
        'time_embed_dim': 128
    }
    
    model = create_unet_small({'model': model_config})
    ckpt_path = temp_checkpoint_dir / "epoch_10.pt"
    
    # Load checkpoint
    metadata = load_model_checkpoint(ckpt_path, model, use_ema=False)
    
    # Check metadata
    assert metadata['epoch'] == 42
    assert metadata['ema_used'] == False
    assert 'checkpoint_path' in metadata


def test_load_ema_checkpoint(temp_checkpoint_dir):
    """Test loading EMA checkpoint format."""
    model_config = {
        'in_ch': 1,
        'base_ch': 32,
        'ch_mult': [1, 2],
        'time_embed_dim': 128
    }
    
    model = create_unet_small({'model': model_config})
    ckpt_path = temp_checkpoint_dir / "ema.pt"
    
    # Load EMA checkpoint
    metadata = load_model_checkpoint(ckpt_path, model, use_ema=True)
    
    # Check that EMA was used
    assert metadata['ema_used'] == True


def test_checkpoint_manager_initialization(temp_checkpoint_dir):
    """Test CheckpointManager with mock checkpoints."""
    ckpt_list = ["epoch_10.pt", "epoch_30.pt", "ema.pt"]
    
    manager = CheckpointManager(
        ckpt_dir=temp_checkpoint_dir,
        ckpt_list=ckpt_list,
        use_ema=True
    )
    
    # Check that all checkpoints were found
    assert len(manager.ckpt_paths) == 3
    
    # Check checkpoint info
    info_list = manager.get_checkpoint_info()
    assert len(info_list) == 3
    
    for info in info_list:
        assert info['exists'] == True
        assert info['size_mb'] > 0


def test_model_forward_after_checkpoint_load(temp_checkpoint_dir):
    """Test that model works correctly after loading checkpoint."""
    model_config = {
        'in_ch': 1,
        'base_ch': 32,
        'ch_mult': [1, 2],
        'time_embed_dim': 128
    }
    
    # Create model
    model = create_unet_small({'model': model_config})
    model.eval()
    
    # Load checkpoint
    ckpt_path = temp_checkpoint_dir / "epoch_10.pt"
    load_model_checkpoint(ckpt_path, model)
    
    # Test forward pass
    batch_size = 2
    x = torch.randn(batch_size, 1, 28, 28)
    t = torch.randint(0, 1000, (batch_size,))
    
    with torch.no_grad():
        output = model(x, t)
    
    # Check output
    assert output.shape == x.shape
    assert not torch.isnan(output).any()
    assert not torch.isinf(output).any()


def test_time_embedding_compatibility():
    """Test that time embedding produces consistent outputs."""
    embed_dim = 256
    embedder = TimestepEmbedder(embed_dim)
    
    # Test with various timesteps
    timesteps = torch.tensor([0, 100, 500, 999])
    embeddings = embedder(timesteps)
    
    assert embeddings.shape == (4, embed_dim)
    assert not torch.isnan(embeddings).any()
    
    # Test consistency
    emb1 = embedder(torch.tensor([42]))
    emb2 = embedder(torch.tensor([42]))
    torch.testing.assert_close(emb1, emb2)


def test_schedules_compatibility():
    """Test that DDPM schedules match Day 6 format."""
    # Test different schedule types
    for schedule in ['linear', 'cosine']:
        schedules = DDPMSchedules(timesteps=1000, schedule=schedule)
        
        # Check that all required attributes exist
        required_attrs = [
            'betas', 'alphas', 'alphas_cumprod', 'alphas_cumprod_prev',
            'sqrt_recip_alphas', 'sqrt_one_minus_alphas_cumprod',
            'posterior_variance_beta', 'posterior_variance_posterior'
        ]
        
        for attr in required_attrs:
            assert hasattr(schedules, attr), f"Missing attribute: {attr}"
            assert getattr(schedules, attr).shape == (1000,), f"Wrong shape for {attr}"


def test_config_compatibility():
    """Test loading different model configurations."""
    configs = [
        # MNIST config
        {
            'model': {
                'in_ch': 1,
                'base_ch': 64,
                'ch_mult': [1, 2, 2],
                'time_embed_dim': 256
            }
        },
        # CIFAR-10 config
        {
            'model': {
                'in_ch': 3,
                'base_ch': 64,
                'ch_mult': [1, 2, 2, 2],
                'time_embed_dim': 256
            }
        }
    ]
    
    for config in configs:
        model = create_unet_small(config)
        
        # Test that model can be created
        assert isinstance(model, UNetSmall)
        
        # Test forward pass
        in_ch = config['model']['in_ch']
        x = torch.randn(1, in_ch, 32, 32)
        t = torch.tensor([500])
        
        with torch.no_grad():
            output = model(x, t)
        
        assert output.shape == x.shape


def test_strict_vs_non_strict_loading(temp_checkpoint_dir):
    """Test strict vs non-strict checkpoint loading."""
    model_config = {
        'in_ch': 1,
        'base_ch': 32,
        'ch_mult': [1, 2],
        'time_embed_dim': 128
    }
    
    model = create_unet_small({'model': model_config})
    ckpt_path = temp_checkpoint_dir / "epoch_10.pt"
    
    # Test strict loading (should work)
    metadata = load_model_checkpoint(ckpt_path, model, strict=True)
    assert metadata is not None
    
    # Test non-strict loading (should also work)
    metadata = load_model_checkpoint(ckpt_path, model, strict=False)
    assert metadata is not None


def test_checkpoint_iteration(temp_checkpoint_dir):
    """Test iterating through multiple checkpoints."""
    ckpt_list = ["epoch_10.pt", "epoch_30.pt", "ema.pt"]
    
    manager = CheckpointManager(
        ckpt_dir=temp_checkpoint_dir,
        ckpt_list=ckpt_list,
        use_ema=True
    )
    
    model_config = {
        'in_ch': 1,
        'base_ch': 32,
        'ch_mult': [1, 2],
        'time_embed_dim': 128
    }
    
    model = create_unet_small({'model': model_config})
    
    # Iterate through checkpoints
    checkpoint_count = 0
    for metadata, loaded_model in manager.iterate_checkpoints(model):
        checkpoint_count += 1
        
        # Check metadata
        assert 'checkpoint_path' in metadata
        assert 'checkpoint_index' in metadata
        
        # Test that model still works
        x = torch.randn(1, 1, 28, 28)
        t = torch.tensor([100])
        
        with torch.no_grad():
            output = loaded_model(x, t)
        
        assert output.shape == x.shape
    
    assert checkpoint_count == 3


def test_missing_checkpoint_handling(temp_checkpoint_dir):
    """Test handling of missing checkpoints."""
    # Include a non-existent checkpoint
    ckpt_list = ["epoch_10.pt", "nonexistent.pt", "ema.pt"]
    
    # Should handle missing checkpoints gracefully
    try:
        manager = CheckpointManager(
            ckpt_dir=temp_checkpoint_dir,
            ckpt_list=ckpt_list,
            use_ema=True
        )
        # Should only find the existing ones
        assert len(manager.ckpt_paths) == 2
    except FileNotFoundError:
        # This is also acceptable behavior
        pass


def test_device_handling(temp_checkpoint_dir):
    """Test that checkpoints can be loaded to different devices."""
    model_config = {
        'in_ch': 1,
        'base_ch': 32,
        'ch_mult': [1, 2],
        'time_embed_dim': 128
    }
    
    # Test CPU loading
    model_cpu = create_unet_small({'model': model_config})
    ckpt_path = temp_checkpoint_dir / "epoch_10.pt"
    
    metadata = load_model_checkpoint(ckpt_path, model_cpu, device=torch.device('cpu'))
    
    # Test that model is on CPU
    assert next(model_cpu.parameters()).device == torch.device('cpu')
    
    # Test forward pass
    x = torch.randn(1, 1, 28, 28)
    t = torch.tensor([100])
    
    with torch.no_grad():
        output = model_cpu(x, t)
    
    assert output.device == torch.device('cpu')


if __name__ == "__main__":
    pytest.main([__file__])
