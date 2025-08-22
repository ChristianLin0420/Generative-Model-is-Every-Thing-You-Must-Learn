# Day 3: Variational Autoencoder (VAE) Refresher

> **🚀 Quick Start**: `make install` → `make train-mnist` → `make eval` → Done! ✨
> 
> **All operations available through simple make commands** - no complex CLI needed.

## 🎯 Objective
Comprehensive VAE implementation from scratch with **simple make commands** for training, evaluation, visualization, and comparison. This implementation covers everything from basic VAE theory to advanced features like β-VAE, IWAE bounds, and interactive latent space exploration.

**✨ Key Feature**: Everything is accessible through simple `make` commands - no complex Python CLI needed!

## ✅ Completed Features

### 🏗️ Core Implementation
- **Multiple VAE Architectures**
  - `VAEConv`: CNN-based encoder/decoder (default)
  - `VAEResNet`: ResNet-based architecture for deeper models
  - `VAEMLP`: Fully-connected baseline for MNIST
- **Advanced Loss Functions**
  - Multiple reconstruction losses: BCE, L2, Charbonnier
  - Proper KL divergence computation with numerical stability
  - β-VAE support with flexible scheduling (linear, cyclical)
  - IWAE bounds for tighter log-likelihood estimates
  - Free bits implementation to prevent posterior collapse

### 🚀 Training Pipeline
- **Full Training Loop** with comprehensive logging
- **Automatic Mixed Precision (AMP)** support for faster training
- **KL Annealing** with multiple schedules (linear, cyclical)
- **Exponential Moving Average (EMA)** for stable model weights
- **TensorBoard Integration** for training visualization
- **Flexible Configuration** via YAML files

### 📊 Evaluation & Analysis
- **Reconstruction Metrics**: PSNR, SSIM, LPIPS
- **Generative Quality**: IWAE bounds, FID-proxy features
- **Latent Space Analysis**: Distribution plots, KL per dimension
- **Comprehensive Logging** with CSV export

### 🎨 Visualization Tools
- **Reconstruction Grids**: Original vs reconstructed comparisons
- **Latent Traversals**: Explore individual latent dimensions
- **2D Latent Scatter**: Visualize 2D latent spaces with class colors
- **Interpolation Sequences**: Smooth transitions between images
- **Prior Samples**: Generate new images from learned distribution

### 🔧 Sampling & Generation
- **Prior Sampling** with temperature control
- **Latent Interpolation** (linear and spherical)
- **Conditional Sampling** from posterior distributions
- **Latent Arithmetic** for concept manipulation

### 💻 User Interface
- **🎯 Make Commands**: Simple `make train-mnist`, `make eval`, `make sample` etc.
- **Command-Line Interface** with comprehensive subcommands (advanced users)
- **Shell Scripts** for easy training and evaluation
- **Interactive Notebook** for latent space exploration
- **Rich Progress Bars** and beautiful console output

## 📁 Project Structure

```
Day_03_VAE_Refresher/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
├── Makefile                    # Common operations
├── configs/
│   ├── mnist.yaml              # MNIST training config
│   └── cifar10.yaml            # CIFAR-10 training config
├── src/
│   ├── utils.py                # Core utilities
│   ├── dataset.py              # Data loading and preprocessing
│   ├── losses.py               # Loss functions and schedulers
│   ├── models/
│   │   ├── vae_conv.py         # CNN VAE (default)
│   │   ├── vae_resnet.py       # ResNet VAE
│   │   └── vae_mlp.py          # MLP VAE
│   ├── train.py                # Training pipeline
│   ├── eval.py                 # Evaluation metrics
│   ├── sample.py               # Sampling utilities
│   ├── visualize.py            # Visualization tools
│   └── cli.py                  # Command-line interface
├── scripts/
│   ├── train_mnist.sh          # Quick MNIST training
│   ├── train_cifar10.sh        # Quick CIFAR-10 training
│   ├── make_samples.sh         # Generate all visualizations
│   └── compare_with_dae.sh     # Compare with Day-2 DAE
├── notebooks/
│   └── 03_vae_latent_playground.ipynb  # Interactive exploration
├── tests/
│   ├── test_models.py          # Model architecture tests
│   ├── test_losses.py          # Loss function tests
│   ├── test_train_step.py      # Training step tests
│   └── test_visualize.py       # Visualization tests
└── outputs/
    ├── ckpts/                  # Model checkpoints
    ├── logs/                   # Training logs & TensorBoard
    ├── grids/                  # Visualization grids
    ├── samples/                # Generated samples
    └── reports/                # Analysis reports
```

## 🚀 Quick Start

### 1. Installation
```bash
make install
```

### 2. Train a VAE
```bash
# MNIST (quick start - 5 minutes on GPU)
make train-mnist

# CIFAR-10 (more challenging - 20 minutes on GPU)  
make train-cifar10
```

### 3. Generate Samples & Visualizations
```bash
# Evaluate model performance
make eval

# Generate samples from prior
make sample

# Create reconstruction grids
make viz-recon

# Create latent traversals
make viz-traverse

# Create 2D latent scatter (if 2D latent space)
make viz-scatter

# Generate interpolations
make interpolate
```

### 4. Compare with DAE from Day 2
```bash
make compare-dae
```

### 5. Interactive Exploration
```bash
# Start Jupyter and open the interactive notebook
jupyter notebook notebooks/03_vae_latent_playground.ipynb
```

### 6. Run Tests
```bash
make test
```

### 7. Clean Outputs
```bash
make clean      # Clean generated outputs
make clean-all  # Clean everything including data
```

## ⚡ Complete Workflow (One-Command Training to Analysis)

```bash
# 1. Install dependencies
make install

# 2. Train VAE on MNIST (5 minutes on GPU)
make train-mnist

# 3. Generate all visualizations and analysis
make eval && make sample && make viz-recon && make viz-traverse

# 4. (Optional) Compare with Day-2 DAE
make compare-dae

# 5. (Optional) Interactive exploration
jupyter notebook notebooks/03_vae_latent_playground.ipynb
```

**That's it! 🎉** You'll have a fully trained VAE with comprehensive analysis.

### 📋 See All Available Commands
```bash
make help    # Shows all available make commands
```

or just run `make` to see the help menu.

### 📊 Make Commands Summary

| Command | Description | Time |
|---------|-------------|------|
| `make install` | Install dependencies | 30s |
| `make train-mnist` | Train VAE on MNIST | 5 min (GPU) |
| `make train-cifar10` | Train VAE on CIFAR-10 | 20 min (GPU) |
| `make eval` | Evaluate trained model | 1 min |
| `make sample` | Generate prior samples | 30s |
| `make viz-recon` | Create reconstruction grids | 30s |
| `make viz-traverse` | Create latent traversals | 1 min |
| `make interpolate` | Generate interpolations | 30s |
| `make compare-dae` | Compare with Day-2 DAE | 2 min |
| `make test` | Run test suite | 1 min |
| `make clean` | Clean outputs | 5s |

## 📋 Command Reference

### 🎯 Make Commands (Recommended)

#### Training
```bash
make train-mnist     # Train VAE on MNIST
make train-cifar10   # Train VAE on CIFAR-10
```

#### Evaluation & Visualization  
```bash
make eval           # Evaluate trained model
make sample         # Generate prior samples
make viz-recon      # Create reconstruction grids
make viz-traverse   # Create latent traversals  
make viz-scatter    # Create 2D latent scatter (if 2D latent)
make interpolate    # Generate interpolations
```

#### Analysis & Comparison
```bash
make compare-dae    # Compare with Day-2 DAE
make test          # Run test suite
make clean         # Clean generated outputs
make clean-all     # Clean everything including data
```

#### Development
```bash
make install       # Install dependencies
make install-dev   # Install with dev dependencies
make lint         # Run linting checks
make format       # Format code
```

### 🔧 Direct Python CLI Commands (Advanced)

If you need more control, you can also use the CLI directly:

#### Training
```bash
python -m src.cli train --config configs/mnist.yaml
python -m src.cli train --config configs/cifar10.yaml
```

#### Evaluation
```bash
python -m src.cli eval --config configs/mnist.yaml
```

#### Sampling & Visualization
```bash
python -m src.cli sample.prior --config configs/mnist.yaml
python -m src.cli sample.interpolate --config configs/mnist.yaml
python -m src.cli viz.recon_grid --config configs/mnist.yaml
python -m src.cli viz.traverse --config configs/mnist.yaml --dims 0 1 2
python -m src.cli viz.latent_scatter --config configs/mnist.yaml  # 2D only
```

#### Comparison
```bash
python -m src.cli compare.dae --config configs/mnist.yaml \
    --dae_ckpt ../Day_02_Denoising_Autoencoder/outputs/ckpts/best.pt
```

## 📊 Expected Outputs

After training and evaluation, you'll have:

### Core Visualizations
- `outputs/grids/recon_grid.png` - Reconstruction quality comparison
- `outputs/samples/prior_samples.png` - Generated samples from prior
- `outputs/grids/interpolations.png` - Latent space interpolations
- `outputs/grids/traverse_dim_k.png` - Individual dimension traversals
- `outputs/grids/latent_scatter.png` - 2D latent space (if applicable)

### Quantitative Metrics
- `outputs/logs/metrics.csv` - PSNR, SSIM, LPIPS, IWAE scores
- `outputs/logs/train.log` - Detailed training logs
- TensorBoard logs in `outputs/logs/` for training curves

### Interactive Tools
- Jupyter notebook for real-time latent space exploration
- Rich CLI with progress bars and colored output
- Comprehensive error handling and helpful messages

## 🎓 Learning Outcomes

By completing this implementation, you will have:

- **Deep VAE Understanding**: Theory and practice of variational inference
- **PyTorch Expertise**: Advanced training loops, AMP, model architectures
- **Generative Modeling**: Understanding of latent variable models
- **Evaluation Skills**: Quantitative metrics for generative models
- **Software Engineering**: Clean code, testing, configuration management

## 📚 Key Concepts Covered

### Theoretical
- **Variational Inference**: Approximate posterior with neural networks
- **Evidence Lower Bound (ELBO)**: Reconstruction + KL divergence
- **Reparameterization Trick**: Backpropagation through stochastic layers
- **β-VAE**: Controlling reconstruction-disentanglement trade-off
- **Posterior Collapse**: When latent variables are ignored

### Practical
- **KL Annealing**: Gradual introduction of regularization
- **IWAE Bounds**: Tighter likelihood estimates with importance sampling
- **Architecture Choices**: CNN vs ResNet vs MLP trade-offs
- **Numerical Stability**: Avoiding NaN/inf in loss computations
- **Evaluation Metrics**: PSNR, SSIM, LPIPS, FID for generative models

## 📖 References
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) (Kingma & Welling, 2014)
- [β-VAE: Learning Basic Visual Concepts](https://openreview.net/forum?id=Sy2fzU9gl)
- [Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519) (IWAE)
- [Understanding disentangling in β-VAE](https://arxiv.org/abs/1804.03599)

---
**Status**: ✅ **COMPLETE** - Comprehensive implementation with all features  
**Time Invested**: ~8 hours of development  
**Difficulty**: ⭐⭐⭐⭐⭐ (Advanced implementation with production-quality features)