# Day 7: Sampling from DDPM

## 🎯 Objective
Implement the complete DDPM sampling procedure and evaluate sample quality using trained models from Day 6.

## ✅ Implementation Status
This project is **COMPLETE** with all core functionality implemented:

- ✅ **Ancestral DDPM Sampling**: Full step-by-step implementation
- ✅ **DDIM Fast Sampling**: Deterministic sampling with fewer steps  
- ✅ **Multiple Variance Schedules**: β_t vs posterior variance options
- ✅ **Trajectory Visualization**: T→0 denoising process visualization
- ✅ **Quality Evaluation**: FID-proxy and LPIPS metrics
- ✅ **CLI Tools**: Complete command-line interface
- ✅ **Interactive Jupyter Notebook**: Hands-on exploration
- ✅ **Comprehensive Tests**: Unit tests for all components
- ✅ **Day 6 Compatibility**: Seamless checkpoint loading

## 📋 Features

### Core Sampling Methods
- **Ancestral DDPM**: Traditional stochastic sampling with proper variance handling
- **DDIM**: Deterministic Denoising Implicit Models for fast sampling (10-50x speedup)
- **Variance Options**: Choose between β_t or posterior variance schedules

### Visualization & Analysis  
- **Trajectory Recording**: Capture intermediate denoising steps T→0
- **Grid Generation**: Create sample galleries for checkpoint comparison
- **Animation Export**: Generate GIF/MP4 animations of sampling process
- **Quality Metrics**: FID-proxy scores and perceptual distances

### Checkpoint Management
- **Multi-Checkpoint Support**: Load and compare different training epochs
- **EMA Weight Loading**: Exponential moving average checkpoint support
- **Compatibility Verification**: Automatic architecture matching with Day 6 models

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate your environment
conda activate generative

# Install dependencies
pip install -r requirements.txt

# Create output directories
make create-dirs
```

### 2. Basic Usage
```bash
# Generate sample grids (works without Day 6 checkpoints)
python demo_sampling.py

# Run interactive notebook
jupyter notebook notebooks/07_sampling_playground.ipynb

# CLI sampling (requires Day 6 checkpoints)
python -m src.cli sample.grid --config configs/mnist.yaml
```

### 3. With Day 6 Checkpoints
```bash
# Compare checkpoints and generate quality report
./scripts/compare_ckpts.sh mnist

# Generate trajectory animation
./scripts/animate_single.sh mnist ema.pt 0

# Quick MNIST samples
./scripts/sample_mnist.sh
```

## 🧮 DDPM Sampling Algorithm

### Ancestral Sampling (Implemented)
```
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    ε̂ = ε_θ(x_t, t)                    # Predict noise
    x̂_0 = (x_t - √(1-ᾱ_t) * ε̂) / √ᾱ_t   # Predict clean image
    
    if t > 0:
        z ~ N(0, I)                    # Sample noise
    else:
        z = 0                          # No noise at final step
    
    # Ancestral step with proper variance
    x_{t-1} = (1/√α_t) * (x_t - (1-α_t)/√(1-ᾱ_t) * ε̂) + σ_t * z
```

### DDIM Sampling (Implemented) 
```
x_T ~ N(0, I)
for t in [T, T-Δt, T-2Δt, ..., 0]:   # Fewer steps
    ε̂ = ε_θ(x_t, t)
    x̂_0 = (x_t - √(1-ᾱ_t) * ε̂) / √ᾱ_t
    
    # Deterministic step (η=0)
    x_{t-Δt} = √ᾱ_{t-Δt} * x̂_0 + √(1-ᾱ_{t-Δt}) * ε̂
```

### Variance Schedules
- **β_t Schedule**: σ_t = β_t (simpler, more variance)
- **Posterior Schedule**: σ_t = √((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t (DDPM paper default)

## 📁 Project Structure

```
Day_07_Sampling_From_DDPM/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration  
├── Makefile                          # Build commands
├── demo_sampling.py                  # Standalone demo script
├── configs/
│   ├── mnist.yaml                    # MNIST sampling config
│   └── cifar10.yaml                  # CIFAR-10 sampling config
├── src/
│   ├── utils.py                      # Utilities (device, I/O, visualization)
│   ├── ddpm_schedules.py            # Noise schedules (β_t, α_t, ᾱ_t)
│   ├── checkpoints.py               # Checkpoint loading and management
│   ├── models/
│   │   ├── time_embedding.py        # Timestep embeddings
│   │   └── unet_small.py           # UNet architecture (Day 6 compatible)
│   ├── sampler.py                   # 🔥 CORE: Ancestral & DDIM samplers
│   ├── visualize.py                 # Trajectory grids, animations, plots
│   ├── quality.py                   # FID-proxy, LPIPS evaluation  
│   └── cli.py                       # Command-line interface
├── scripts/
│   ├── sample_mnist.sh              # Quick MNIST sampling
│   ├── animate_single.sh            # Generate trajectory animation
│   └── compare_ckpts.sh             # Compare multiple checkpoints
├── notebooks/
│   └── 07_sampling_playground.ipynb # 📓 Interactive exploration
├── tests/
│   ├── test_sampler_step.py         # Sampling correctness tests
│   ├── test_compat_day6_ckpt.py     # Day 6 compatibility tests
│   └── test_visualize_io.py         # Visualization I/O tests
└── outputs/                         # Generated content
    ├── grids/                       # Sample image grids
    ├── animations/                  # T→0 trajectory videos
    ├── curves/                      # Quality vs checkpoint plots
    ├── logs/                        # CSV metrics and timing
    └── reports/                     # Markdown analysis reports
```

## 🛠️ CLI Commands

### Sample Generation
```bash
# Generate 64-sample grid with latest checkpoint
python -m src.cli sample.grid --config configs/mnist.yaml

# Generate grid with specific checkpoint  
python -m src.cli sample.grid --config configs/mnist.yaml --ckpt epoch_30.pt

# Generate DDIM samples (fast)
python -m src.cli sample.grid --config configs/mnist.yaml --ddim
```

### Trajectory Visualization
```bash
# Create T→0 animation for sample 0
python -m src.cli sample.traj --config configs/mnist.yaml --idx 0

# High-resolution trajectory (record every 10 steps)
python -m src.cli sample.traj --config configs/mnist.yaml --record-every 10 --fps 12
```

### Quality Evaluation  
```bash
# Compute FID-proxy scores across all checkpoints
python -m src.cli eval.quality --config configs/mnist.yaml

# Quick evaluation with fewer samples
python -m src.cli eval.quality --config configs/mnist.yaml --num-samples 100
```

### Multi-Checkpoint Comparison
```bash
# Generate comparison panel + quality analysis
python -m src.cli compare.ckpts --config configs/mnist.yaml
```

## 📊 Outputs & Evaluation

### Generated Content
- **Sample Grids**: 8×8 image galleries showing sample quality per checkpoint
- **Trajectory Animations**: Step-by-step denoising visualization (GIF/MP4)
- **Quality Curves**: FID-proxy scores vs training progress
- **Comparison Panels**: Side-by-side checkpoint evaluation

### Quality Metrics
- **FID-Proxy**: Fréchet Inception Distance using CNN features
- **LPIPS**: Learned Perceptual Image Patch Similarity (optional)
- **Basic Statistics**: Pixel value distributions, sample diversity

### Analysis Reports
- **Checkpoint Evolution**: How sample quality improves with training
- **Variance Schedule Impact**: β_t vs posterior variance comparison  
- **DDIM vs DDPM**: Speed/quality trade-off analysis

## 🧪 Testing & Verification

### Run Tests
```bash
# Run all tests
PYTHONPATH=. python -m pytest tests/ -v

# Test specific components
PYTHONPATH=. python -m pytest tests/test_sampler_step.py -v
PYTHONPATH=. python -m pytest tests/test_compat_day6_ckpt.py -v
```

### Demo Script (No Day 6 Required)
```bash
# Comprehensive demonstration with random weights
python demo_sampling.py

# Should output: "✅ Day 7 implementation is working correctly!"
```

### Make Commands
```bash
make setup          # Install dependencies
make create-dirs     # Create output structure
make test           # Run pytest suite
make test-quick     # Quick test run
make sample         # Generate samples (requires checkpoints)
make compare        # Compare checkpoints (requires checkpoints)
make clean          # Clean outputs
```

## 🎓 Learning Outcomes

By completing this implementation, you will understand:

1. **Ancestral Sampling Mathematics**: The step-by-step reverse diffusion process
2. **Variance Schedule Impact**: How different σ_t choices affect sample quality
3. **DDIM Acceleration**: Deterministic sampling for 10-50x speedup
4. **Quality Evaluation**: Objective metrics for generative model assessment
5. **Production Workflows**: CLI tools, batch processing, and evaluation pipelines

## 🔗 Integration with Day 6

This sampling implementation is **fully compatible** with Day 6 training:

- **Checkpoint Loading**: Automatically detects EMA vs raw weights
- **Architecture Matching**: Verifies model configuration consistency  
- **Schedule Compatibility**: Uses identical β_t, α_t, ᾱ_t computations
- **Seamless Workflow**: Train on Day 6 → Sample on Day 7

## 📖 References & Resources

- **DDPM Paper**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- **DDIM Paper**: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song et al., 2021)
- **Implementation Guide**: This comprehensive codebase with tests and demos

---

## 🎉 Milestone: Complete DDPM Pipeline!

**Congratulations!** You now have:
- ✅ A complete DDPM training pipeline (Day 6)
- ✅ A full-featured sampling and evaluation system (Day 7)  
- ✅ Deep understanding of diffusion model fundamentals
- ✅ Production-ready tools for generative modeling

**Next Steps**: Advanced techniques (classifier guidance, conditional generation, latent diffusion)

---
**Implementation Time**: ~6-8 hours  
**Difficulty**: ⭐⭐⭐⭐☆
**Status**: ✅ **COMPLETE & TESTED**