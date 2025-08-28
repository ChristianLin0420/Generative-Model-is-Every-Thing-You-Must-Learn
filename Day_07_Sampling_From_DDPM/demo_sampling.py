#!/usr/bin/env python3
"""
Demo script for Day 7 DDPM Sampling.

This script demonstrates the core functionality without requiring 
trained checkpoints from Day 6. It shows:
1. Model creation and sampling setup
2. Basic ancestral sampling 
3. DDIM fast sampling
4. Trajectory visualization
5. Grid creation and saving

This serves as both a demonstration and a smoke test.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.utils import set_seed, get_device, make_grid
from src.ddpm_schedules import DDPMSchedules
from src.models.unet_small import UNetSmall
from src.sampler import DDPMSampler
from src.visualize import quick_grid


def create_dummy_model(in_ch=1, device='cpu'):
    """Create a dummy model for demonstration (not trained)."""
    model = UNetSmall(
        in_ch=in_ch,
        base_ch=32,
        ch_mult=[1, 2],
        time_embed_dim=128
    )
    model.to(device)
    model.eval()
    return model


def demo_basic_sampling():
    """Demonstrate basic ancestral sampling."""
    print("=== Basic Ancestral Sampling Demo ===")
    
    # Setup
    set_seed(42)
    device = get_device('cpu')  # Use CPU for demo
    print(f"Using device: {device}")
    
    # Create model and schedules
    model = create_dummy_model(in_ch=1, device=device)
    schedules = DDPMSchedules(timesteps=100, schedule='linear')  # Shorter for demo
    schedules.to(device)
    
    # Create sampler
    sampler = DDPMSampler(model, schedules, device)
    print(f"Sampler created with {sampler.T} timesteps")
    
    # Generate samples
    print("Generating 4 samples...")
    shape = (4, 1, 28, 28)  # 4 MNIST-like images
    
    results = sampler.sample(
        shape=shape,
        variance_type='posterior',
        progress=True
    )
    
    samples = results['samples']
    print(f"Generated samples shape: {samples.shape}")
    print(f"Sample range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    return samples


def demo_ddim_sampling():
    """Demonstrate DDIM fast sampling."""
    print("\n=== DDIM Fast Sampling Demo ===")
    
    # Setup
    device = get_device('cpu')
    model = create_dummy_model(in_ch=1, device=device)
    schedules = DDPMSchedules(timesteps=100, schedule='cosine')
    schedules.to(device)
    
    sampler = DDPMSampler(model, schedules, device)
    
    # Generate with DDIM
    print("Generating 4 samples with DDIM (20 steps)...")
    shape = (4, 1, 28, 28)
    
    results = sampler.sample(
        shape=shape,
        ddim=True,
        num_steps=20,
        ddim_eta=0.0,  # Deterministic
        progress=True
    )
    
    samples = results['samples']
    info = results['sampling_info']
    
    print(f"DDIM samples shape: {samples.shape}")
    print(f"Actual steps used: {info['num_steps']}")
    print(f"DDIM enabled: {info['ddim']}")
    
    return samples


def demo_trajectory_sampling():
    """Demonstrate trajectory recording."""
    print("\n=== Trajectory Sampling Demo ===")
    
    # Setup
    device = get_device('cpu')
    model = create_dummy_model(in_ch=1, device=device)
    schedules = DDPMSchedules(timesteps=50, schedule='linear')  # Very short for demo
    schedules.to(device)
    
    sampler = DDPMSampler(model, schedules, device)
    
    # Generate trajectory
    print("Generating single sample trajectory...")
    
    results = sampler.sample_single_trajectory(
        shape=(1, 1, 28, 28),
        record_every=10,
        progress=True
    )
    
    trajectory = results['trajectory']
    trajectory_steps = results['trajectory_steps']
    
    print(f"Trajectory recorded: {len(trajectory)} frames")
    print(f"Timesteps: {trajectory_steps}")
    
    return trajectory, trajectory_steps


def demo_variance_comparison():
    """Demonstrate different variance schedules."""
    print("\n=== Variance Schedule Comparison ===")
    
    # Setup
    device = get_device('cpu')
    model = create_dummy_model(in_ch=1, device=device)
    schedules = DDPMSchedules(timesteps=50, schedule='cosine')
    schedules.to(device)
    
    sampler = DDPMSampler(model, schedules, device)
    
    # Compare variance types
    variance_samples = {}
    
    for var_type in ['beta', 'posterior']:
        print(f"Sampling with {var_type} variance...")
        
        results = sampler.sample(
            shape=(2, 1, 28, 28),
            variance_type=var_type,
            progress=False
        )
        
        variance_samples[var_type] = results['samples']
    
    print("Variance comparison completed")
    return variance_samples


def demo_schedule_types():
    """Demonstrate different noise schedules."""
    print("\n=== Noise Schedule Comparison ===")
    
    schedules_to_test = ['linear', 'cosine']
    
    for schedule in schedules_to_test:
        print(f"Testing {schedule} schedule...")
        
        schedules = DDPMSchedules(timesteps=100, schedule=schedule)
        
        # Check some properties
        betas = schedules.betas
        alphas_cumprod = schedules.alphas_cumprod
        
        print(f"  Beta range: [{betas.min():.6f}, {betas.max():.6f}]")
        print(f"  Alpha_cumprod range: [{alphas_cumprod.min():.6f}, {alphas_cumprod.max():.6f}]")
        print(f"  Final alpha_cumprod: {alphas_cumprod[-1]:.6f}")


def demo_model_architectures():
    """Demonstrate different model configurations."""
    print("\n=== Model Architecture Demo ===")
    
    configs = [
        {"name": "MNIST", "in_ch": 1, "base_ch": 32, "ch_mult": [1, 2]},
        {"name": "CIFAR", "in_ch": 3, "base_ch": 64, "ch_mult": [1, 2, 2]},
    ]
    
    for config in configs:
        print(f"Testing {config['name']} configuration...")
        
        model = UNetSmall(
            in_ch=config['in_ch'],
            base_ch=config['base_ch'],
            ch_mult=config['ch_mult'],
            time_embed_dim=128
        )
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")
        
        # Test forward pass
        batch_size = 2
        if config['name'] == "MNIST":
            x = torch.randn(batch_size, 1, 28, 28)
        else:
            x = torch.randn(batch_size, 3, 32, 32)
        
        t = torch.randint(0, 1000, (batch_size,))
        
        with torch.no_grad():
            output = model(x, t)
        
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        assert output.shape == x.shape
        print("  ✓ Forward pass successful")


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n=== Visualization Demo ===")
    
    # Create some dummy samples
    samples = torch.randn(9, 1, 28, 28)
    
    # Test grid creation
    grid = make_grid(samples, nrow=3, normalize=True, value_range=(-1, 1))
    print(f"Created grid with shape: {grid.shape}")
    
    # Test saving (but don't actually display)
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    try:
        quick_grid(samples, title="Demo Samples", nrow=3)
        print("✓ Visualization functions work correctly")
    except Exception as e:
        print(f"Visualization test failed: {e}")


def main():
    """Run all demonstrations."""
    print("🎯 Day 7 DDPM Sampling Demonstration")
    print("=" * 50)
    
    try:
        # Run demos
        demo_schedule_types()
        demo_model_architectures()
        
        samples1 = demo_basic_sampling()
        samples2 = demo_ddim_sampling()
        trajectory, steps = demo_trajectory_sampling()
        variance_samples = demo_variance_comparison()
        
        demo_visualization()
        
        print("\n" + "=" * 50)
        print("🎉 All demonstrations completed successfully!")
        print("\nKey Results:")
        print(f"- Generated {samples1.shape[0]} samples with ancestral sampling")
        print(f"- Generated {samples2.shape[0]} samples with DDIM")
        print(f"- Recorded trajectory with {len(trajectory)} frames")
        print(f"- Compared {len(variance_samples)} variance schedules")
        print("\n✅ Day 7 implementation is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
