"""
Tests for visualization functionality in Day 2: Denoising Autoencoder
"""

import pytest
import torch
import tempfile
from pathlib import Path

from src.visualize import (
    create_reconstruction_grid,
    create_sigma_panel,
    plot_metrics_curves,
    plot_training_curves,
    create_failure_cases_grid
)
from src.models import ConvDAE


class TestVisualization:
    """Test visualization functions."""
    
    def test_create_reconstruction_grid(self):
        """Test reconstruction grid creation."""
        clean = torch.rand(4, 1, 16, 16)
        noisy = torch.rand(4, 1, 16, 16) 
        recon = torch.rand(4, 1, 16, 16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "recon_grid.png"
            
            create_reconstruction_grid(
                clean, noisy, recon, save_path, num_samples=4
            )
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_create_sigma_panel(self):
        """Test sigma panel creation."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=8, num_downs=1)
        clean_image = torch.rand(1, 16, 16)
        sigmas = [0.1, 0.2, 0.3]
        device = torch.device('cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "sigma_panel.png"
            
            create_sigma_panel(
                model, clean_image, sigmas, save_path, device
            )
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_plot_metrics_curves(self):
        """Test metrics curve plotting."""
        # Create dummy metrics data
        metrics_data = {
            0.1: {'psnr': 25.0, 'ssim': 0.8, 'mse': 0.001, 'mae': 0.02},
            0.2: {'psnr': 22.0, 'ssim': 0.7, 'mse': 0.004, 'mae': 0.04},
            0.3: {'psnr': 20.0, 'ssim': 0.6, 'mse': 0.009, 'mae': 0.06}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "metrics_curves.png"
            
            plot_metrics_curves(metrics_data, save_path)
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_plot_training_curves(self):
        """Test training curves plotting."""
        import csv
        
        with tempfile.TemporaryDirectory() as tmpdir:
            train_csv = Path(tmpdir) / "train_metrics.csv"
            val_csv = Path(tmpdir) / "val_metrics.csv"
            plot_path = Path(tmpdir) / "training_curves.png"
            
            # Create dummy CSV files
            train_data = [
                {'epoch': 1, 'loss': 0.1, 'psnr': 20.0, 'ssim': 0.7, 'mse': 0.01},
                {'epoch': 2, 'loss': 0.08, 'psnr': 22.0, 'ssim': 0.75, 'mse': 0.008},
                {'epoch': 3, 'loss': 0.06, 'psnr': 24.0, 'ssim': 0.8, 'mse': 0.006}
            ]
            
            val_data = [
                {'epoch': 1, 'loss': 0.12, 'psnr': 19.0, 'ssim': 0.65, 'mse': 0.012},
                {'epoch': 2, 'loss': 0.09, 'psnr': 21.0, 'ssim': 0.7, 'mse': 0.009},
                {'epoch': 3, 'loss': 0.07, 'psnr': 23.0, 'ssim': 0.75, 'mse': 0.007}
            ]
            
            # Write CSV files
            with open(train_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'psnr', 'ssim', 'mse'])
                writer.writeheader()
                writer.writerows(train_data)
            
            with open(val_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'psnr', 'ssim', 'mse'])
                writer.writeheader()
                writer.writerows(val_data)
            
            # Test plotting
            plot_training_curves(train_csv, val_csv, plot_path)
            
            assert plot_path.exists()
            assert plot_path.stat().st_size > 0
    
    def test_create_failure_cases_grid(self):
        """Test failure cases grid creation."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dummy data
        clean_data = torch.rand(20, 1, 8, 8)
        noisy_data = torch.rand(20, 1, 8, 8)
        sigmas = torch.ones(20) * 0.1
        
        dataset = TensorDataset(clean_data, noisy_data, sigmas)
        test_loader = DataLoader(dataset, batch_size=4)
        
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=8, num_downs=1)
        device = torch.device('cpu')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "failure_cases.png"
            
            create_failure_cases_grid(
                model, test_loader, device, save_path,
                num_cases=4, worst=True
            )
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_different_image_formats(self):
        """Test visualization with different image formats."""
        # Test grayscale
        clean_gray = torch.rand(2, 1, 16, 16)
        noisy_gray = torch.rand(2, 1, 16, 16)
        recon_gray = torch.rand(2, 1, 16, 16)
        
        # Test RGB
        clean_rgb = torch.rand(2, 3, 16, 16)
        noisy_rgb = torch.rand(2, 3, 16, 16)
        recon_rgb = torch.rand(2, 3, 16, 16)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test grayscale
            gray_path = Path(tmpdir) / "gray_grid.png"
            create_reconstruction_grid(
                clean_gray, noisy_gray, recon_gray, gray_path
            )
            assert gray_path.exists()
            
            # Test RGB
            rgb_path = Path(tmpdir) / "rgb_grid.png"
            create_reconstruction_grid(
                clean_rgb, noisy_rgb, recon_rgb, rgb_path
            )
            assert rgb_path.exists()
    
    def test_large_batch_visualization(self):
        """Test visualization with large batch (should handle gracefully)."""
        clean = torch.rand(100, 1, 8, 8)  # Large batch
        noisy = torch.rand(100, 1, 8, 8)
        recon = torch.rand(100, 1, 8, 8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "large_batch_grid.png"
            
            # Should handle large batch by selecting subset
            create_reconstruction_grid(
                clean, noisy, recon, save_path, num_samples=8
            )
            
            assert save_path.exists()
            assert save_path.stat().st_size > 0
    
    def test_empty_metrics_data(self):
        """Test metrics plotting with empty or minimal data."""
        # Empty metrics
        empty_metrics = {}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "empty_metrics.png"
            
            # Should handle gracefully (might create empty plot)
            try:
                plot_metrics_curves(empty_metrics, save_path)
            except Exception:
                # If it fails, that's also acceptable for empty data
                pass
        
        # Single point metrics
        single_metrics = {
            0.1: {'psnr': 25.0, 'ssim': 0.8, 'mse': 0.001}
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "single_metrics.png"
            
            plot_metrics_curves(single_metrics, save_path)
            assert save_path.exists()


class TestVisualizationEdgeCases:
    """Test edge cases in visualization."""
    
    def test_single_image_visualization(self):
        """Test visualization with single image."""
        clean = torch.rand(1, 1, 8, 8)
        noisy = torch.rand(1, 1, 8, 8)
        recon = torch.rand(1, 1, 8, 8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "single_image.png"
            
            create_reconstruction_grid(
                clean, noisy, recon, save_path, num_samples=1
            )
            
            assert save_path.exists()
    
    def test_very_small_images(self):
        """Test visualization with very small images."""
        clean = torch.rand(4, 1, 2, 2)  # Very small images
        noisy = torch.rand(4, 1, 2, 2)
        recon = torch.rand(4, 1, 2, 2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "tiny_images.png"
            
            create_reconstruction_grid(
                clean, noisy, recon, save_path
            )
            
            assert save_path.exists()
    
    def test_extreme_values(self):
        """Test visualization with extreme pixel values."""
        # All zeros
        clean_zeros = torch.zeros(2, 1, 8, 8)
        noisy_zeros = torch.zeros(2, 1, 8, 8)
        recon_zeros = torch.zeros(2, 1, 8, 8)
        
        # All ones
        clean_ones = torch.ones(2, 1, 8, 8)
        noisy_ones = torch.ones(2, 1, 8, 8)
        recon_ones = torch.ones(2, 1, 8, 8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Test zeros
            zeros_path = Path(tmpdir) / "zeros_grid.png"
            create_reconstruction_grid(
                clean_zeros, noisy_zeros, recon_zeros, zeros_path
            )
            assert zeros_path.exists()
            
            # Test ones
            ones_path = Path(tmpdir) / "ones_grid.png"
            create_reconstruction_grid(
                clean_ones, noisy_ones, recon_ones, ones_path
            )
            assert ones_path.exists()
    
    def test_sigma_panel_edge_cases(self):
        """Test sigma panel with edge case inputs."""
        model = ConvDAE(in_ch=1, out_ch=1, base_ch=8, num_downs=1)
        device = torch.device('cpu')
        
        # Single sigma
        single_sigma = [0.1]
        clean_image = torch.rand(1, 8, 8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "single_sigma_panel.png"
            
            create_sigma_panel(
                model, clean_image, single_sigma, save_path, device
            )
            
            assert save_path.exists()
        
        # Many sigmas
        many_sigmas = [i * 0.05 for i in range(1, 11)]  # 10 different sigmas
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "many_sigma_panel.png"
            
            create_sigma_panel(
                model, clean_image, many_sigmas, save_path, device
            )
            
            assert save_path.exists()
    
    def test_directory_creation(self):
        """Test that directories are created automatically."""
        clean = torch.rand(2, 1, 8, 8)
        noisy = torch.rand(2, 1, 8, 8)
        recon = torch.rand(2, 1, 8, 8)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Deep nested path that doesn't exist
            nested_path = Path(tmpdir) / "deep" / "nested" / "path" / "grid.png"
            
            create_reconstruction_grid(
                clean, noisy, recon, nested_path
            )
            
            assert nested_path.exists()
            assert nested_path.parent.exists()


if __name__ == '__main__':
    pytest.main([__file__])