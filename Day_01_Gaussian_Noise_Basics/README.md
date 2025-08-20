# Day 1: Gaussian Noise Basics

> **30-Day Diffusion & Flow Matching Challenge - Day 01**  
> Complete implementation of Gaussian noise fundamentals for deep learning

## 🎯 Overview

This project implements a comprehensive toolkit for understanding and visualizing Gaussian noise effects on images. It serves as the foundation for diffusion model concepts by providing hands-on experience with noise processes, image degradation metrics, and progressive noise injection.

## ✨ Features

- **🔧 Core Noise Engine**: Deterministic Gaussian noise addition with multiple schedules
- **📊 Quality Metrics**: MSE, PSNR, SSIM, and SNR computation
- **🎨 Rich Visualizations**: Progressive grids, animations, and metric plots
- **⚡ CLI Interface**: Three powerful commands for noise analysis
- **📓 Interactive Notebook**: Jupyter-based exploration and experimentation
- **🧪 Full Test Suite**: Comprehensive testing with pytest
- **🛠️ Professional Setup**: Makefile, configuration management, and CI-ready

## 🚀 Quick Start

### Installation

```bash
# Clone and navigate to project
cd Day_01_Gaussian_Noise_Basics

# Install dependencies
make install
# or: pip install -r requirements.txt

# Run complete demonstration
make demo
```

### Basic Usage

```bash
# Generate noise progression grids
python -m src.cli add-noise --config configs/default.yaml --num-batches 2

# Create noise animation
python -m src.cli animate --config configs/default.yaml --num-images 16

# Compute theoretical statistics
python -m src.cli stats --config configs/default.yaml

# Run one-click demo
./scripts/run_noise_demo.sh
```

### Interactive Exploration

```bash
# Launch Jupyter notebook for hands-on experimentation
jupyter notebook notebooks/01_gaussian_noise_exploration.ipynb
```

## 📁 Project Structure

```
Day_01_Gaussian_Noise_Basics/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Python project configuration
├── Makefile                    # Development commands
├── configs/
│   └── default.yaml            # Experiment configuration
├── data/                       # Auto-downloaded MNIST data
├── src/                        # Core implementation
│   ├── __init__.py
│   ├── utils.py                # Utilities (seeding, device, etc.)
│   ├── dataset.py              # MNIST data loading
│   ├── noise.py                # Core noise functionality
│   ├── visualize.py            # Visualization tools
│   ├── metrics.py              # Quality metrics (MSE/PSNR/SSIM)
│   └── cli.py                  # Command line interface
├── scripts/
│   ├── run_noise_demo.sh       # Complete demonstration
│   └── export_animation.sh     # Animation export utilities
├── notebooks/
│   └── 01_gaussian_noise_exploration.ipynb
├── outputs/
│   ├── grids/                  # Generated noise progression grids
│   ├── animations/             # GIF/MP4 animations
│   └── logs/                   # Metrics CSV and plots
└── tests/
    ├── test_noise.py           # Core functionality tests
    └── test_visualize.py       # Visualization tests
```

## 🔧 Configuration

Edit `configs/default.yaml` to customize experiments:

```yaml
seed: 42
device: "cpu"                   # or "cuda"

dataset:
  root: "./data"
  batch_size: 64
  normalize_range: [0, 1]       # or [-1, 1]

noise:
  sigmas: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
  schedule: "linear"            # linear | cosine | custom

visualization:
  grid_nrow: 8
  animation_fps: 2
```

## 📊 Generated Outputs

### Visual Outputs
- **Progressive Grids**: `outputs/grids/batch_XXX_noise_grid.png`
- **Animations**: `outputs/animations/mnist_noise.gif`
- **Metrics Plots**: `outputs/logs/metrics_plot.png`

### Data Outputs
- **Metrics CSV**: `outputs/logs/noise_metrics.csv`
- **Statistics**: `outputs/logs/theoretical_stats.csv`

### Example Results

| Noise Level (σ) | SNR (dB) | PSNR (dB) | SSIM  | MSE    |
|------------------|----------|-----------|-------|--------|
| 0.0              | ∞        | ∞         | 1.000 | 0.000  |
| 0.1              | 20.0     | ~25.0     | 0.95+ | ~0.001 |
| 0.3              | 10.5     | ~15.0     | 0.80+ | ~0.009 |
| 0.5              | 6.0      | ~10.0     | 0.60+ | ~0.025 |
| 1.0              | 0.0      | ~5.0      | 0.30+ | ~0.100 |

## 🎯 Key Concepts Demonstrated

### 1. Forward Noise Process
```python
from src.noise import add_gaussian_noise

# Add Gaussian noise with standard deviation σ
noisy_image = add_gaussian_noise(clean_image, sigma=0.3, clip_range=(0, 1))
```

### 2. Noise Schedules
```python
from src.noise import sigma_schedule, NoiseScheduler

# Generate different noise progressions
linear_schedule = sigma_schedule('linear', 10, 0.0, 1.0)
cosine_schedule = sigma_schedule('cosine', 10, 0.0, 1.0)
scheduler = NoiseScheduler('linear', num_levels=10)
```

### 3. Quality Metrics
```python
from src.metrics import noise_degradation_metrics

# Comprehensive quality analysis
metrics = noise_degradation_metrics(original, sigma=0.3, noisy_image, max_val=1.0)
# Returns: {'mse': ..., 'psnr': ..., 'ssim': ..., 'snr_db': ...}
```

## 🧪 Testing

```bash
# Run all tests
make test
# or: pytest tests/ -v

# Run specific test file
pytest tests/test_noise.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📈 Development Commands

```bash
make help          # Show available commands
make install       # Install dependencies
make run           # Run noise generation
make demo          # Complete demonstration
make test          # Run tests
make lint          # Format code
make clean         # Clean outputs
```

## 🎓 Learning Outcomes

By completing this project, you will understand:

- ✅ **Gaussian Noise Properties**: Mathematical foundations and statistical behavior
- ✅ **Image Degradation**: How noise affects visual quality and structure
- ✅ **Quality Metrics**: PSNR, SSIM, MSE relationships and interpretations
- ✅ **Signal-to-Noise Ratio**: Theoretical and empirical SNR computation
- ✅ **Reproducible Research**: Seeded experiments and deterministic workflows
- ✅ **Software Engineering**: Testing, documentation, and professional code practices

## 🔗 Next Steps

1. **Experiment**: Try different noise schedules and parameters
2. **Extend**: Add new datasets (CIFAR-10, custom images)
3. **Optimize**: GPU acceleration for large-scale experiments
4. **Advance**: Move to **Day 02: Denoising Autoencoder**

## 📚 References

- **Gaussian Distribution Theory**: Statistical foundations of noise processes
- **Image Quality Assessment**: PSNR, SSIM, and perceptual metrics
- **Diffusion Models**: Foundation concepts for generative modeling

---

**⭐ Difficulty**: ⭐⭐☆☆☆ | **⏱️ Time Estimate**: 2-4 hours | **🎯 Status**: Complete Implementation