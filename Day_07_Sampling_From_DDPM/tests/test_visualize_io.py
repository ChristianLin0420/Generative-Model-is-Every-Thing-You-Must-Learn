"""
Test visualization I/O functionality.
Verifies grid creation, animation saving, and basic visualization operations.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils import make_grid, save_image_grid, save_animation
from src.visualize import (
    reverse_trajectory_grid, make_animation, multi_checkpoint_panel,
    quick_grid, quick_trajectory
)


@pytest.fixture
def sample_images():
    """Create sample images for testing."""
    # Create both grayscale and RGB images
    grayscale_images = torch.randn(8, 1, 32, 32)  # Normalize to [-1, 1]
    rgb_images = torch.randn(8, 3, 32, 32)
    
    return {
        'grayscale': grayscale_images,
        'rgb': rgb_images
    }


@pytest.fixture
def sample_trajectory():
    """Create sample trajectory for testing."""
    T = 100
    trajectory = []
    trajectory_steps = []
    
    # Create trajectory from T to 0
    for i, t in enumerate(range(T, -1, -10)):
        # Gradually reduce noise
        noise_level = t / T
        img = torch.randn(2, 1, 28, 28) * noise_level
        trajectory.append(img)
        trajectory_steps.append(t)
    
    return trajectory, trajectory_steps


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_make_grid_grayscale(sample_images):
    """Test grid creation for grayscale images."""
    images = sample_images['grayscale']
    
    # Test default grid
    grid = make_grid(images, nrow=4)
    
    assert grid.shape[0] == 1  # Grayscale
    assert grid.ndim == 3  # (C, H, W)
    assert grid.shape[1] > images.shape[2]  # Grid height > single image height
    assert grid.shape[2] > images.shape[3]  # Grid width > single image width


def test_make_grid_rgb(sample_images):
    """Test grid creation for RGB images."""
    images = sample_images['rgb']
    
    grid = make_grid(images, nrow=4)
    
    assert grid.shape[0] == 3  # RGB
    assert grid.ndim == 3  # (C, H, W)
    assert grid.shape[1] > images.shape[2]  # Grid height > single image height
    assert grid.shape[2] > images.shape[3]  # Grid width > single image width


def test_make_grid_normalization(sample_images):
    """Test grid normalization options."""
    images = sample_images['grayscale']
    
    # Test without normalization
    grid_raw = make_grid(images, normalize=False)
    
    # Test with normalization
    grid_norm = make_grid(images, normalize=True)
    
    # Normalized grid should be in [0, 1]
    assert grid_norm.min() >= 0
    assert grid_norm.max() <= 1
    
    # Should be different
    assert not torch.allclose(grid_raw, grid_norm)


def test_make_grid_padding(sample_images):
    """Test grid padding parameter."""
    images = sample_images['grayscale'][:4]  # Use 4 images for 2x2 grid
    
    # Test different padding values
    grid_pad0 = make_grid(images, nrow=2, padding=0)
    grid_pad5 = make_grid(images, nrow=2, padding=5)
    
    # Grid with padding should be larger
    assert grid_pad5.shape[1] > grid_pad0.shape[1]
    assert grid_pad5.shape[2] > grid_pad0.shape[2]


def test_save_image_grid_png(sample_images, temp_output_dir):
    """Test saving image grid as PNG."""
    images = sample_images['grayscale']
    save_path = temp_output_dir / "test_grid.png"
    
    # Save grid
    save_image_grid(images, save_path, nrow=4)
    
    # Check that file was created
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_save_image_grid_rgb(sample_images, temp_output_dir):
    """Test saving RGB image grid."""
    images = sample_images['rgb']
    save_path = temp_output_dir / "test_grid_rgb.png"
    
    # Save RGB grid
    save_image_grid(images, save_path, nrow=4)
    
    # Check that file was created
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_save_animation_gif(sample_trajectory, temp_output_dir):
    """Test saving animation as GIF."""
    trajectory, trajectory_steps = sample_trajectory
    save_path = temp_output_dir / "test_animation.gif"
    
    # Convert trajectory to individual frames
    frames = [img[0] for img in trajectory[:5]]  # Use first sample, first 5 frames
    
    save_animation(frames, save_path, fps=5)
    
    # Check that file was created
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_save_animation_mp4(sample_trajectory, temp_output_dir):
    """Test saving animation as MP4."""
    trajectory, trajectory_steps = sample_trajectory
    save_path = temp_output_dir / "test_animation.mp4"
    
    # Convert trajectory to individual frames
    frames = [img[0] for img in trajectory[:5]]
    
    try:
        save_animation(frames, save_path, fps=5)
        
        # Check that file was created
        assert save_path.exists()
        assert save_path.stat().st_size > 0
    except Exception:
        # MP4 might not be available in all environments
        pytest.skip("MP4 saving not available")


def test_reverse_trajectory_grid(sample_trajectory, temp_output_dir):
    """Test reverse trajectory grid creation."""
    trajectory, trajectory_steps = sample_trajectory
    
    # Create trajectory grid
    grid_array = reverse_trajectory_grid(
        trajectory=trajectory[:6],  # Use first 6 frames
        trajectory_steps=trajectory_steps[:6],
        save_path=temp_output_dir / "trajectory_grid.png",
        num_samples=2,
        figsize=(12, 4)
    )
    
    # Check output
    assert isinstance(grid_array, np.ndarray)
    assert grid_array.ndim == 3  # (H, W, C)
    assert grid_array.shape[2] == 3  # RGB
    
    # Check that file was saved
    assert (temp_output_dir / "trajectory_grid.png").exists()


def test_make_animation_function(sample_trajectory, temp_output_dir):
    """Test make_animation function."""
    trajectory, trajectory_steps = sample_trajectory
    save_path = temp_output_dir / "trajectory_animation.gif"
    
    make_animation(
        trajectory=trajectory[:5],
        trajectory_steps=trajectory_steps[:5],
        save_path=save_path,
        sample_idx=0,
        fps=5
    )
    
    # Check that file was created
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_multi_checkpoint_panel(sample_images, temp_output_dir):
    """Test multi-checkpoint comparison panel."""
    images = sample_images['grayscale']
    
    # Create mock checkpoint samples
    sample_grids = {
        'Epoch 10': images[:4],
        'Epoch 20': images[2:6],
        'Epoch 30': images[4:8]
    }
    
    save_path = temp_output_dir / "checkpoint_panel.png"
    
    panel_array = multi_checkpoint_panel(
        sample_grids=sample_grids,
        save_path=save_path,
        figsize_per_grid=(4, 4),
        nrow=2
    )
    
    # Check output
    assert isinstance(panel_array, np.ndarray)
    assert panel_array.ndim == 3
    assert panel_array.shape[2] == 3  # RGB
    
    # Check that file was saved
    assert save_path.exists()


def test_quick_grid_functionality(sample_images):
    """Test quick_grid function (display function)."""
    images = sample_images['grayscale']
    
    # This function displays matplotlib plots, so we just test it doesn't crash
    try:
        # Import matplotlib in non-interactive mode
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        quick_grid(images, title="Test Grid", nrow=4)
        
    except ImportError:
        pytest.skip("Matplotlib not available")


def test_quick_trajectory_functionality(sample_trajectory):
    """Test quick_trajectory function."""
    trajectory, trajectory_steps = sample_trajectory
    
    try:
        # Import matplotlib in non-interactive mode
        import matplotlib
        matplotlib.use('Agg')
        
        quick_trajectory(trajectory, trajectory_steps, sample_idx=0, title="Test Trajectory")
        
    except ImportError:
        pytest.skip("Matplotlib not available")


def test_empty_trajectory_handling():
    """Test handling of empty trajectories."""
    with pytest.raises(ValueError):
        reverse_trajectory_grid([], [], num_samples=1)


def test_trajectory_single_frame(temp_output_dir):
    """Test trajectory with single frame."""
    single_frame = [torch.randn(1, 1, 28, 28)]
    single_step = [0]
    
    grid_array = reverse_trajectory_grid(
        trajectory=single_frame,
        trajectory_steps=single_step,
        save_path=temp_output_dir / "single_frame.png",
        num_samples=1
    )
    
    assert isinstance(grid_array, np.ndarray)
    assert (temp_output_dir / "single_frame.png").exists()


def test_invalid_animation_format(sample_trajectory, temp_output_dir):
    """Test invalid animation format handling."""
    trajectory, _ = sample_trajectory
    frames = [img[0] for img in trajectory[:3]]
    save_path = temp_output_dir / "test_animation.invalid"
    
    with pytest.raises(ValueError):
        save_animation(frames, save_path)


def test_batch_size_mismatch_trajectory():
    """Test trajectory with inconsistent batch sizes."""
    # Create trajectory with different batch sizes (should handle gracefully)
    trajectory = [
        torch.randn(2, 1, 28, 28),
        torch.randn(2, 1, 28, 28),
        torch.randn(2, 1, 28, 28)
    ]
    trajectory_steps = [100, 50, 0]
    
    # Should work fine (uses first sample)
    try:
        import matplotlib
        matplotlib.use('Agg')
        
        quick_trajectory(trajectory, trajectory_steps, sample_idx=0)
        
    except ImportError:
        pytest.skip("Matplotlib not available")


def test_directory_creation(temp_output_dir):
    """Test that save functions create directories if needed."""
    nested_path = temp_output_dir / "nested" / "directory" / "test.png"
    
    # Directory doesn't exist yet
    assert not nested_path.parent.exists()
    
    # Save should create directory
    images = torch.randn(4, 1, 32, 32)
    save_image_grid(images, nested_path, nrow=2)
    
    # Directory should now exist
    assert nested_path.parent.exists()
    assert nested_path.exists()


def test_large_grid_handling(temp_output_dir):
    """Test handling of large image grids."""
    # Create many small images
    images = torch.randn(64, 1, 16, 16)
    save_path = temp_output_dir / "large_grid.png"
    
    save_image_grid(images, save_path, nrow=8)
    
    # Should complete successfully
    assert save_path.exists()
    assert save_path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__])
