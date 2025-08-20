"""
Tests for visualization functionality
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.visualize import make_progressive_grid, make_animation, plot_metrics
from src.utils import save_image_grid


class TestVisualization:
    """Test visualization functions."""
    
    def test_save_image_grid(self):
        """Test basic image grid saving."""
        # Create test tensor
        batch = torch.rand(4, 1, 8, 8)  # Small images for testing
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_grid.png"
            
            save_image_grid(batch, output_path, nrow=2)
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0  # File has content
    
    def test_make_progressive_grid(self):
        """Test progressive noise grid creation."""
        batch = torch.rand(8, 1, 16, 16)
        sigmas = [0.0, 0.1, 0.2]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "progressive_grid.png"
            
            # Should not raise exception
            make_progressive_grid(
                batch=batch,
                sigmas=sigmas,
                path=output_path,
                nrow=4,
                normalize_range=(0, 1)
            )
            
            assert output_path.exists()
    
    def test_make_progressive_grid_different_ranges(self):
        """Test progressive grid with different normalization ranges."""
        batch_01 = torch.rand(4, 1, 8, 8)  # [0, 1] range
        batch_11 = torch.rand(4, 1, 8, 8) * 2 - 1  # [-1, 1] range
        sigmas = [0.0, 0.1]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test [0, 1] range
            path1 = Path(tmpdir) / "grid_01.png"
            make_progressive_grid(batch_01, sigmas, path1, normalize_range=(0, 1))
            assert path1.exists()
            
            # Test [-1, 1] range
            path2 = Path(tmpdir) / "grid_11.png"
            make_progressive_grid(batch_11, sigmas, path2, normalize_range=(-1, 1))
            assert path2.exists()
    
    def test_make_animation_gif(self):
        """Test GIF animation creation."""
        batch = torch.rand(4, 1, 8, 8)
        sigmas = [0.0, 0.1, 0.2]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_animation.gif"
            
            make_animation(
                batch=batch,
                sigmas=sigmas,
                path=output_path,
                fps=1,
                normalize_range=(0, 1)
            )
            
            assert output_path.exists()
            assert output_path.stat().st_size > 0
    
    @pytest.mark.skipif(True, reason="MP4 creation may require additional codecs")
    def test_make_animation_mp4(self):
        """Test MP4 animation creation (may fail without proper codecs)."""
        batch = torch.rand(4, 1, 8, 8)
        sigmas = [0.0, 0.1]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_animation.mp4"
            
            try:
                make_animation(
                    batch=batch,
                    sigmas=sigmas,
                    path=output_path,
                    fps=1,
                    normalize_range=(0, 1)
                )
                
                assert output_path.exists()
            except Exception as e:
                # MP4 creation might fail depending on system setup
                pytest.skip(f"MP4 creation failed: {e}")
    
    def test_unsupported_animation_format(self):
        """Test that unsupported format raises error."""
        batch = torch.rand(4, 1, 8, 8)
        sigmas = [0.0, 0.1]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_animation.xyz"  # Invalid format
            
            with pytest.raises(ValueError, match="Unsupported format"):
                make_animation(batch, sigmas, output_path)
    
    def test_plot_metrics(self):
        """Test metrics plotting functionality."""
        # Create test CSV data
        import csv
        
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_metrics.csv"
            plot_path = Path(tmpdir) / "test_plot.png"
            
            # Create test data
            data = [
                {'sigma': 0.0, 'snr_db': 50.0, 'mse': 0.0, 'psnr': 50.0, 'ssim': 1.0},
                {'sigma': 0.1, 'snr_db': 20.0, 'mse': 0.01, 'psnr': 20.0, 'ssim': 0.9},
                {'sigma': 0.2, 'snr_db': 14.0, 'mse': 0.04, 'psnr': 14.0, 'ssim': 0.8},
            ]
            
            with open(csv_path, 'w', newline='') as csvfile:
                fieldnames = ['sigma', 'snr_db', 'mse', 'psnr', 'ssim']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            # Test plotting
            plot_metrics(csv_path, plot_path, figsize=(8, 6))
            
            assert plot_path.exists()
            assert plot_path.stat().st_size > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_batch(self):
        """Test behavior with empty batch."""
        batch = torch.empty(0, 1, 8, 8)
        sigmas = [0.0, 0.1]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "empty_grid.png"
            
            # Should handle gracefully or raise informative error
            try:
                make_progressive_grid(batch, sigmas, output_path)
            except (ValueError, RuntimeError):
                # Expected for empty batch
                pass
    
    def test_single_image(self):
        """Test with single image."""
        batch = torch.rand(1, 1, 16, 16)
        sigmas = [0.0, 0.1, 0.2]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "single_image_grid.png"
            
            make_progressive_grid(batch, sigmas, output_path, nrow=1)
            
            assert output_path.exists()
    
    def test_large_batch_truncation(self):
        """Test that large batches are truncated appropriately."""
        batch = torch.rand(100, 1, 8, 8)  # Large batch
        sigmas = [0.0, 0.1]
        nrow = 4
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "large_batch_grid.png"
            
            # Should handle large batch by selecting subset
            make_progressive_grid(batch, sigmas, output_path, nrow=nrow)
            
            assert output_path.exists()


class TestOutputDirectoryCreation:
    """Test that output directories are created automatically."""
    
    def test_nested_directory_creation(self):
        """Test that nested directories are created."""
        batch = torch.rand(4, 1, 8, 8)
        sigmas = [0.0, 0.1]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested path that doesn't exist
            output_path = Path(tmpdir) / "nested" / "dirs" / "test_grid.png"
            
            make_progressive_grid(batch, sigmas, output_path)
            
            assert output_path.exists()
            assert output_path.parent.exists()


if __name__ == '__main__':
    pytest.main([__file__])