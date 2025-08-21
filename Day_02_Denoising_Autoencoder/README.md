# Day 2: Denoising Autoencoder (DAE)

> **30-Day Diffusion & Flow Matching Challenge - Day 02**  
> Complete implementation of denoising autoencoders with limitations analysis

## 🎯 Overview

This project implements a comprehensive denoising autoencoder framework with multiple architectures (ConvDAE, UNetDAE), advanced training features, and thorough analysis of limitations compared to generative models. It serves as a crucial stepping stone to understand why diffusion models and other generative approaches are necessary.

## ✨ Features

- **🏗️ Multiple Architectures**: ConvDAE and UNet-based DAE implementations
- **⚡ Advanced Training**: Mixed precision, EMA weights, adaptive schedules, curriculum learning
- **📊 Comprehensive Metrics**: PSNR, SSIM, MSE, LPIPS, and custom quality measures
- **🎨 Rich Visualizations**: Reconstruction grids, sigma sweep panels, failure case analysis
- **🔬 Limitations Analysis**: Over-smoothing detection, diversity collapse analysis, frequency content comparison
- **🧪 Robust Testing**: Comprehensive test suite with multiple architectures and edge cases
- **⚙️ Professional Setup**: CLI interface, configuration management, automated reporting

## 🚀 Quick Start

### Installation
```bash
cd Day_02_Denoising_Autoencoder

# Install dependencies
make install
# or: pip install -r requirements.txt

# Quick MNIST training
make train
# or: ./scripts/train_mnist.sh
```

### Basic Usage
```bash
# Train on MNIST
python -m src.cli train --config configs/mnist.yaml

# Evaluate trained model
python -m src.cli eval --config configs/mnist.yaml

# Generate visualizations
python -m src.cli viz.recon_grid --config configs/mnist.yaml
python -m src.cli viz.sigma_panel --config configs/mnist.yaml

# Analyze limitations
python -m src.cli compare.limitations --config configs/mnist.yaml
```

### One-Click Demo
```bash
# Complete MNIST pipeline
make demo-mnist

# Generate all figures
make figures
```

## 📁 Project Structure

```
Day_02_Denoising_Autoencoder/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project configuration
├── Makefile                          # Development commands
├── configs/
│   └── mnist.yaml                    # MNIST training configuration
├── data/                             # Auto-downloaded datasets
├── src/                              # Core implementation
│   ├── models/
│   │   ├── dae_conv.py               # Convolutional DAE architectures
│   │   └── dae_unet.py               # U-Net DAE with skip connections
│   ├── utils.py                      # Utilities, EMA, checkpointing
│   ├── dataset.py                    # MNIST loaders with on-the-fly noising
│   ├── noise.py                      # Advanced noise generation and schedules
│   ├── loss.py                       # Multiple loss functions (L1/L2/Charbonnier/Perceptual)
│   ├── metrics.py                    # Comprehensive evaluation metrics
│   ├── train.py                      # Full training pipeline with modern features
│   ├── eval.py                       # Comprehensive evaluation framework
│   ├── visualize.py                  # Advanced visualization and reporting
│   ├── compare.py                    # Limitations analysis vs generative models
│   └── cli.py                        # Unified command-line interface
├── scripts/
│   ├── train_mnist.sh                # One-click MNIST training
│   ├── eval_mismatch_noise.sh        # Robustness testing
│   └── make_figures.sh               # Complete visualization pipeline
├── notebooks/
│   └── 02_dae_experiments.ipynb      # Interactive experiments and analysis
├── outputs/
│   ├── ckpts/                        # Model checkpoints (regular + EMA)
│   ├── logs/                         # Training metrics, evaluation results
│   ├── grids/                        # Reconstruction visualizations
│   ├── panels/                       # Sigma sweep panels
│   └── reports/                      # Comprehensive analysis reports
└── tests/
    ├── test_noise.py                 # Noise functionality tests
    ├── test_models.py                # Model architecture tests
    ├── test_train_step.py            # Training pipeline tests
    └── test_visualize.py             # Visualization tests
```

## 🏗️ Architectures Available

### ConvDAE (Lightweight Baseline)
- Simple encoder-decoder with conv/transpose-conv layers
- Fewer parameters, faster training
- Good baseline for comparison

### UNetDAE (Advanced Architecture)
- U-Net with skip connections for detail preservation
- Better reconstruction quality, especially for complex images
- Configurable depth and normalization options

```python
from src.models import create_model

# Create models
conv_model = create_model('conv', in_ch=1, out_ch=1, base_ch=64)
unet_model = create_model('unet', in_ch=1, out_ch=1, base_ch=64, num_downs=4)
```

## 🔧 Configuration

### MNIST Configuration (`configs/mnist.yaml`)
```yaml
model:
  name: "unet"              # unet | conv
  in_ch: 1                  # Grayscale
  base_ch: 64
  num_downs: 3

train:
  epochs: 20
  lr: 2e-4
  loss: "l2"               # l1 | l2 | charbonnier
  amp: true                # Mixed precision
  ema: true                # Exponential moving average

noise:
  train_sigmas: [0.1, 0.2, 0.3, 0.5]
  test_sigmas: [0.1, 0.3, 0.7, 1.0]    # Includes unseen levels
```



## 📊 Generated Outputs

### Training Artifacts
- **Model Checkpoints**: `outputs/ckpts/best_model.pth`
- **Training Logs**: `outputs/logs/train_metrics.csv`, `outputs/logs/val_metrics.csv`
- **Configuration**: Saved with each checkpoint for reproducibility

### Visualizations
- **Reconstruction Grids**: `outputs/grids/recon_grid_*.png` (clean | noisy | reconstructed)
- **Sigma Sweep Panels**: `outputs/panels/sigma_panel_*.png` (same image, different noise levels)
- **Training Curves**: `outputs/reports/training_curves.png`
- **Failure Cases**: `outputs/grids/worst_cases.png`, `outputs/grids/best_cases.png`

### Analysis Reports
- **Limitations Analysis**: `outputs/reports/limitations_analysis.html`
- **Evaluation Results**: `outputs/logs/evaluation_results.json`
- **Frequency Analysis**: `outputs/reports/frequency_analysis.png`

## 🎯 Key Features Demonstrated

### 1. Advanced Training Pipeline
```python
# Modern training with all features
python -m src.cli train --config configs/mnist.yaml
# Includes: Mixed precision, EMA weights, gradient clipping, cosine LR schedule
```

### 2. Comprehensive Evaluation
```python
# Multi-faceted evaluation
python -m src.cli eval --config configs/mnist.yaml
# Tests: Overall metrics, noise robustness, out-of-distribution generalization
```

### 3. Limitations Analysis
```python
# Quantitative analysis of DAE limitations
python -m src.cli compare.limitations --config configs/mnist.yaml
# Analyzes: Over-smoothing, diversity collapse, frequency content preservation
```

## 📈 Expected Results

### MNIST Performance (typical results after 20 epochs)
| Noise Level (σ) | PSNR (dB) | SSIM | MSE | Interpretation |
|------------------|-----------|------|-----|----------------|
| 0.1 | 28-32 | 0.85+ | 0.001 | Excellent reconstruction |
| 0.2 | 25-28 | 0.80+ | 0.003 | Good quality |
| 0.3 | 22-25 | 0.75+ | 0.006 | Acceptable quality |
| 0.5 | 18-22 | 0.65+ | 0.015 | Degraded but recognizable |
| 0.7 | 15-18 | 0.55+ | 0.030 | Poor quality |
| 1.0 | 12-15 | 0.45+ | 0.050 | Severe degradation |

### Limitations Analysis
- **Over-smoothing Score**: 0.2-0.4 (moderate detail loss)
- **Diversity Collapse**: 0.3-0.6 (similar outputs from different inputs)
- **High-Frequency Preservation**: 0.6-0.8 (some detail loss)

## 🧪 Testing

```bash
# Run all tests
make test
# or: pytest tests/ -v

# Test specific components
pytest tests/test_models.py -v          # Architecture tests
pytest tests/test_train_step.py -v      # Training tests
pytest tests/test_noise.py -v           # Noise functionality
pytest tests/test_visualize.py -v       # Visualization tests

# Quick smoke test
pytest tests/ -x                        # Stop on first failure
```

## 📈 Development Commands

```bash
make help              # Show all available commands
make install           # Install dependencies
make train             # Train on MNIST
make eval              # Evaluate trained model
make viz               # Generate visualizations
make compare           # Analyze limitations
make test              # Run tests
make clean             # Clean outputs
make full-pipeline     # Complete train→eval→viz→compare pipeline
```

## 🔬 Advanced Features

### Noise Robustness Testing
```bash
# Test model on unseen noise levels
./scripts/eval_mismatch_noise.sh
```

### Architecture Comparison
```python
# Compare ConvDAE vs UNetDAE
models = {
    'conv': create_model('conv', in_ch=1, out_ch=1),
    'unet': create_model('unet', in_ch=1, out_ch=1)
}
# UNet typically shows better detail preservation due to skip connections
```

### Loss Function Ablation
```yaml
# In config file, try different losses:
train:
  loss: "l2"              # Standard MSE
  loss: "l1"              # More robust to outliers
  loss: "charbonnier"     # Combines L1/L2 benefits
  loss: "perceptual"      # VGG-based perceptual loss
```

## 🎓 Learning Outcomes

By completing this project, you will understand:

- ✅ **Autoencoder Principles**: Encoder-decoder architectures and bottleneck representations
- ✅ **Reconstruction vs Generation**: Fundamental differences in model objectives
- ✅ **Training Techniques**: Modern deep learning practices (mixed precision, EMA, etc.)
- ✅ **Quality Assessment**: Multiple metrics for evaluating image reconstruction
- ✅ **Limitation Analysis**: Why reconstruction models fall short for generation tasks
- ✅ **Professional ML Engineering**: Testing, configuration management, reproducible experiments

## 🔍 Key Limitations Identified

### 1. **Over-Smoothing Problem**
```
DAEs minimize reconstruction error → tend to average/blur uncertain details
High-frequency content ratio: ~0.7 (30% detail loss)
```

### 2. **Diversity Collapse** 
```
Different noisy inputs → similar reconstructed outputs
Pairwise output distance: ~0.001 MSE (very similar)
```

### 3. **No Generative Capability**
```
Cannot generate new samples from noise
Only maps noisy → clean, doesn't model P(x)
```

### 4. **Mode Averaging**
```
When uncertain between modes, averages rather than choosing
Results in unrealistic blended outputs
```

## 🔗 Connection to Diffusion Models

This project provides essential foundation for understanding:

- **Why generative models are needed**: DAE limitations motivate probabilistic approaches
- **Quality metrics**: PSNR/SSIM evaluation frameworks carry forward
- **Noise processes**: Experience with noise schedules and robustness
- **Architecture patterns**: UNet backbone knowledge for diffusion models

## 🚀 Next Steps

1. **Experiment**: Try different architectures, loss functions, and datasets
2. **Extend**: Add attention mechanisms, residual connections, or other improvements
3. **Compare**: Benchmark against other denoising methods
4. **Advance**: Move to **Day 03: VAE Refresher** to explore probabilistic generative modeling

## 📚 References

- **Denoising Autoencoders** (Vincent et al., 2008): Original denoising autoencoder paper
- **U-Net Architecture** (Ronneberger et al., 2015): Skip connections for detail preservation  
- **Image Quality Assessment**: PSNR, SSIM, and perceptual metrics
- **Reconstruction vs Generation**: Fundamental differences in model capabilities

---

**⭐ Difficulty**: ⭐⭐⭐☆☆ | **⏱️ Time Estimate**: 3-5 hours | **🎯 Status**: Complete Implementation