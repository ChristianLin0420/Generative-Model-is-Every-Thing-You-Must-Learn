"""
Test suite for visualization utilities: grid and GIF smoke tests
"""

import pytest
import torch
from pathlib import Path
import tempfile
import os
from src.visualize import (
    make_sample_grid, 
    VisualizationManager,
    setup_matplotlib,
    tensor_to_pil
)


class TestVisualization:
    """Test visualization functions"""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample image tensors for testing"""
        # Create 4 sample images, 3 channels, 32x32
        return torch.randn(4, 3, 32, 32)
        
    @pytest.fixture
    def sample_images_grayscale(self):
        """Create sample grayscale images"""
        return torch.randn(4, 1, 28, 28)
        
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
            
    def test_make_sample_grid_basic(self, sample_images, temp_dir):
        """Test basic sample grid creation"""
        save_path = os.path.join(temp_dir, "test_grid.png")
        
        result_path = make_sample_grid(
            samples=sample_images,
            save_path=save_path,
            nrow=2
        )
        
        # Check file was created
        assert os.path.exists(result_path)
        assert result_path == save_path
        
        # Check file size is reasonable
        file_size = os.path.getsize(result_path)
        assert file_size > 1000  # Should be at least 1KB for a real image
        
    def test_make_sample_grid_normalization(self, temp_dir):
        """Test grid creation with different normalization settings"""
        # Create images in different value ranges
        images_negative = torch.randn(2, 3, 16, 16) * 2 - 1  # Roughly [-3, 1]
        images_positive = torch.rand(2, 3, 16, 16)  # [0, 1]
        
        for i, (images, value_range) in enumerate([
            (images_negative, (-1, 1)),
            (images_positive, (0, 1))
        ]):
            save_path = os.path.join(temp_dir, f"test_norm_{i}.png")
            
            result_path = make_sample_grid(
                samples=images,
                save_path=save_path,
                normalize=True,
                value_range=value_range
            )
            
            assert os.path.exists(result_path)
            
    def test_make_sample_grid_grayscale(self, sample_images_grayscale, temp_dir):
        """Test grid creation with grayscale images"""
        save_path = os.path.join(temp_dir, "test_grayscale.png")
        
        result_path = make_sample_grid(
            samples=sample_images_grayscale,
            save_path=save_path,
            nrow=2
        )
        
        assert os.path.exists(result_path)
        
    def test_tensor_to_pil_rgb(self):
        """Test tensor to PIL conversion for RGB images"""
        # RGB image in [-1, 1] range
        tensor = torch.randn(3, 32, 32)
        
        pil_image = tensor_to_pil(tensor)
        
        # Check PIL image properties
        assert pil_image.mode == "RGB"
        assert pil_image.size == (32, 32)
        
    def test_tensor_to_pil_grayscale(self):
        """Test tensor to PIL conversion for grayscale images"""
        # Grayscale image
        tensor = torch.randn(1, 28, 28)
        
        pil_image = tensor_to_pil(tensor)
        
        # Check PIL image properties
        assert pil_image.mode == "L"  # Grayscale mode
        assert pil_image.size == (28, 28)
        
    def test_tensor_to_pil_batch(self):
        """Test tensor to PIL with batch dimension"""
        # Batch of images (should take first one)
        tensor = torch.randn(4, 3, 16, 16)
        
        pil_image = tensor_to_pil(tensor)
        
        assert pil_image.mode == "RGB"
        assert pil_image.size == (16, 16)
        
    def test_visualization_manager_init(self, temp_dir):
        """Test VisualizationManager initialization"""
        viz_manager = VisualizationManager(output_dir=temp_dir)
        
        # Check directories are created
        assert viz_manager.output_dir.exists()
        assert viz_manager.grids_dir.exists()
        assert viz_manager.curves_dir.exists()
        assert viz_manager.animations_dir.exists()
        
    def test_visualization_manager_save_grid(self, sample_images, temp_dir):
        """Test VisualizationManager sample grid saving"""
        viz_manager = VisualizationManager(output_dir=temp_dir)
        
        filename = "test_samples.png"
        result_path = viz_manager.save_sample_grid(
            samples=sample_images,
            filename=filename
        )
        
        # Check file was saved in correct directory
        expected_path = viz_manager.grids_dir / filename
        assert os.path.exists(expected_path)
        assert result_path == str(expected_path)
        
    def test_setup_matplotlib(self):
        """Test matplotlib setup doesn't crash"""
        # This should run without error
        setup_matplotlib()
        
        # Import matplotlib to check it's available
        try:
            import matplotlib.pyplot as plt
            assert True  # matplotlib is available
        except ImportError:
            pytest.skip("matplotlib not available")
            
    def test_grid_with_title(self, sample_images, temp_dir):
        """Test grid creation with title"""
        save_path = os.path.join(temp_dir, "test_title.png")
        
        result_path = make_sample_grid(
            samples=sample_images,
            save_path=save_path,
            title="Test Sample Grid"
        )
        
        assert os.path.exists(result_path)
        
        # File should be larger due to title space
        file_size = os.path.getsize(result_path)
        assert file_size > 1000
        
    def test_grid_with_captions(self, sample_images, temp_dir):
        """Test grid creation with captions"""
        save_path = os.path.join(temp_dir, "test_captions.png")
        captions = [f"Sample {i}" for i in range(len(sample_images))]
        
        result_path = make_sample_grid(
            samples=sample_images,
            save_path=save_path,
            captions=captions
        )
        
        assert os.path.exists(result_path)
        
    def test_different_nrow_values(self, temp_dir):
        """Test grid creation with different nrow values"""
        images = torch.randn(6, 3, 16, 16)
        
        for nrow in [1, 2, 3, 6]:
            save_path = os.path.join(temp_dir, f"test_nrow_{nrow}.png")
            
            result_path = make_sample_grid(
                samples=images,
                save_path=save_path,
                nrow=nrow
            )
            
            assert os.path.exists(result_path)
            
    def test_directory_creation(self, temp_dir):
        """Test automatic directory creation"""
        # Use nested path that doesn't exist
        nested_path = os.path.join(temp_dir, "nested", "path", "test.png")
        images = torch.randn(2, 3, 8, 8)
        
        result_path = make_sample_grid(
            samples=images,
            save_path=nested_path
        )
        
        # Directory should be created and file should exist
        assert os.path.exists(result_path)
        assert os.path.dirname(result_path) == os.path.dirname(nested_path)
        
    def test_empty_input_handling(self, temp_dir):
        """Test handling of empty or minimal input"""
        # Single image
        single_image = torch.randn(1, 3, 16, 16)
        save_path = os.path.join(temp_dir, "single_image.png")
        
        result_path = make_sample_grid(
            samples=single_image,
            save_path=save_path
        )
        
        assert os.path.exists(result_path)
        
    def test_extreme_image_sizes(self, temp_dir):
        """Test with very small and reasonably large images"""
        # Very small images
        small_images = torch.randn(2, 3, 4, 4)
        small_path = os.path.join(temp_dir, "small.png")
        
        make_sample_grid(small_images, small_path)
        assert os.path.exists(small_path)
        
        # Larger images (but not too large for CI)
        large_images = torch.randn(2, 3, 128, 128)
        large_path = os.path.join(temp_dir, "large.png")
        
        make_sample_grid(large_images, large_path)
        assert os.path.exists(large_path)
        
    def test_visualization_manager_complete_workflow(self, temp_dir):
        """Test complete workflow with VisualizationManager"""
        viz_manager = VisualizationManager(output_dir=temp_dir)
        
        # Generate some sample images
        samples = torch.randn(8, 3, 32, 32)
        
        # Save grid
        grid_path = viz_manager.save_sample_grid(
            samples=samples,
            filename="workflow_test.png",
            nrow=4,
            title="Workflow Test"
        )
        
        # Verify all outputs
        assert os.path.exists(grid_path)
        
        # Check file structure
        assert (Path(temp_dir) / "grids" / "workflow_test.png").exists()
        
    def test_error_handling(self, temp_dir):
        """Test error handling for invalid inputs"""
        # Invalid tensor shape (missing channel dimension)
        with pytest.raises(Exception):  # Should raise some error
            invalid_tensor = torch.randn(4, 32, 32)  # Missing channel dim
            tensor_to_pil(invalid_tensor)
            
        # Very small tensor should still work
        tiny_tensor = torch.randn(1, 1, 1)
        pil_image = tensor_to_pil(tiny_tensor)
        assert pil_image.size == (1, 1)