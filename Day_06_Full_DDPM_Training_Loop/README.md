# Day 6: Full DDPM Training Loop ✅

**Complete implementation of DDPM with training, sampling, evaluation, and visualization pipeline**

## 🎯 Overview

This is a production-ready implementation of Denoising Diffusion Probabilistic Models (DDPM) featuring:
- **Full training pipeline** with AMP, EMA, grad clipping, and LR scheduling
- **DDPM & DDIM sampling** with configurable steps and stochasticity  
- **Comprehensive evaluation** with PSNR/SSIM/LPIPS and FID-proxy metrics
- **Rich visualizations** including sample grids, trajectories, and training curves
- **Interactive notebook** for experimentation and parameter exploration
- **Production CLI** with comprehensive commands for all operations

## 🚀 Quick Start

### Installation
```bash
# Setup environment
make setup

# Or manually:
pip install -r requirements.txt
pip install -e .
```

### Training
```bash
# Train on MNIST (recommended for first run)
bash scripts/train_mnist.sh

# Train on CIFAR-10 (requires more compute)  
bash scripts/train_cifar10.sh

# Or use CLI directly
python -m src.cli train --config configs/mnist.yaml
```

### Sampling
```bash
# Generate 64 samples
bash scripts/sample_64.sh

# Create reverse trajectory animation
bash scripts/animate_traj.sh

# Or use CLI
python -m src.cli sample.grid --config configs/mnist.yaml --num_samples 64
```

### Evaluation & Visualization
```bash
# Comprehensive evaluation
python -m src.cli eval --config configs/mnist.yaml

# Plot training curves  
python -m src.cli viz.curves --log_dir outputs/logs

# Generate complete report
bash scripts/make_report.sh
```

## 📁 Project Structure

```
Day_06_Full_DDPM_Training_Loop/
├── src/                           # Core implementation
│   ├── ddpm_schedules.py          # β, α, ᾱ noise schedule builders
│   ├── models/                    # UNet + time embeddings
│   │   ├── time_embedding.py      # Sinusoidal embeddings + MLP
│   │   └── unet_small.py          # UNet with time-FiLM conditioning
│   ├── losses.py                  # ε-prediction loss + parameterizations
│   ├── trainer.py                 # Full training loop with AMP/EMA
│   ├── sampler.py                 # DDPM ancestral + DDIM sampling
│   ├── eval.py                    # PSNR/SSIM/LPIPS/FID metrics
│   ├── visualize.py               # Grids, trajectories, curves
│   ├── dataset.py                 # MNIST/CIFAR loaders + preprocessing
│   ├── utils.py                   # Utilities, EMA, checkpoints, logging
│   └── cli.py                     # Command-line interface
├── configs/                       # Dataset configurations
│   ├── mnist.yaml                 # MNIST training config
│   └── cifar10.yaml               # CIFAR-10 training config  
├── scripts/                       # Training & evaluation scripts
│   ├── train_mnist.sh             # MNIST training script
│   ├── train_cifar10.sh           # CIFAR-10 training script
│   ├── sample_64.sh               # Generate sample grid
│   ├── animate_traj.sh            # Create trajectory animation
│   └── make_report.sh             # Comprehensive report generator
├── tests/                         # Comprehensive test suite
│   ├── test_schedules.py          # Schedule validation tests
│   ├── test_unet_small.py         # Model architecture tests
│   ├── test_loss_and_train.py     # Training step tests
│   ├── test_sampler.py            # Sampling functionality tests
│   └── test_visualize.py          # Visualization tests
├── notebooks/                     # Interactive experimentation
│   └── 06_ddpm_training_playground.ipynb
└── outputs/                       # Generated results
    ├── ckpts/                     # Model checkpoints (best, latest, EMA)
    ├── logs/                      # Training metrics & evaluation results
    ├── grids/                     # Sample image grids
    ├── animations/                # Reverse trajectory GIFs/MP4s
    ├── curves/                    # Training curves & schedule plots
    └── reports/                   # Generated reports & summaries
```

## 🧮 DDPM Training Algorithm

### Core Training Loop
```python
for each training step:
    1. Sample batch of images x_0
    2. Sample time steps t ~ Uniform(1, T)  
    3. Sample noise ε ~ N(0, I)
    4. Compute noisy images x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
    5. Predict noise: ε_pred = ε_θ(x_t, t)
    6. Compute loss: L = ||ε - ε_pred||²
    7. Backpropagate and update θ
```

### Advanced Features
- **Mixed Precision Training** (AMP) for 2x speedup
- **Exponential Moving Average** (EMA) for stable sampling
- **Gradient Clipping** for training stability
- **Cosine LR Scheduling** for better convergence
- **Periodic Sampling** during training for monitoring

## 🏗️ Architecture Details

### UNet Backbone
- **Multi-scale encoder-decoder** with skip connections
- **Time-FiLM conditioning** in residual blocks
- **Multi-head self-attention** at 16×16 resolution
- **GroupNorm + SiLU** for stable training

### Time Embedding
- **Sinusoidal positional encoding** for time steps
- **MLP transformation** for conditioning vectors
- **Broadcast-compatible shapes** for FiLM layers

### Noise Schedules
- **Linear schedule**: Standard β₁=0.0001 to β₁₀₀₀=0.02
- **Cosine schedule**: Better performance, less noise collapse
- **Configurable time steps**: 100-2000 supported

## 📊 Results & Evaluation

### Training Deliverables
- ✅ **Model checkpoints** in `outputs/ckpts/` (best, latest, EMA)
- ✅ **Training curves** showing loss convergence and metrics
- ✅ **Sample grids** (64+ images) generated during training
- ✅ **Trajectory animations** showing reverse diffusion process

### Evaluation Metrics
- **Image Quality**: PSNR, SSIM for reconstruction fidelity
- **Perceptual Quality**: LPIPS for perceptual similarity  
- **Generation Quality**: FID-proxy using Inception features
- **Diversity**: Pairwise LPIPS distances between samples

### Expected Performance
- **MNIST**: High-quality digit generation, FID < 5
- **CIFAR-10**: Natural image generation, FID < 15
- **Sampling Speed**: DDIM 10-100× faster than DDPM

## 🛠️ Advanced Usage

### Custom Training
```python
# Load configuration
config = load_config("configs/mnist.yaml")

# Create model and schedules
model = UNetSmall(**config["model"])
schedules = DDPMSchedules(**config["schedules"])

# Train with custom parameters
trainer = DDPMTrainer(model, schedules, ...)
trainer.train()
```

### Custom Sampling
```python
# Create sampler
sampler = DDPMSampler(schedules)

# DDPM ancestral sampling (slow, high quality)
samples = sampler.ddpm_sample(model, shape=(16, 3, 32, 32))

# DDIM deterministic sampling (fast)
samples = sampler.ddim_sample(model, shape, num_steps=50, eta=0.0)
```

### Interactive Exploration
Open `notebooks/06_ddpm_training_playground.ipynb` for:
- **Parameter exploration** with interactive widgets
- **Schedule visualization** and comparison
- **Step-by-step sampling** visualization
- **Real-time experimentation** with different configurations

## 🧪 Testing

```bash
# Run full test suite
pytest -q tests/

# Test specific modules
pytest tests/test_schedules.py -v
pytest tests/test_unet_small.py -v
pytest tests/test_sampler.py -v

# Check code coverage
pytest --cov=src tests/
```

## 🎛️ Configuration

Modify `configs/mnist.yaml` or `configs/cifar10.yaml` to experiment with:

- **Model architecture**: channels, attention, layers
- **Training hyper-parameters**: LR, batch size, epochs  
- **Noise schedules**: linear vs cosine, time steps
- **Loss functions**: ε-prediction, x₀-prediction, v-parameterization
- **Sampling methods**: DDPM, DDIM, step counts

## 📈 Monitoring & Visualization

### Training Curves
- **Loss convergence** over epochs
- **Learning rate schedule** visualization
- **EMA decay** and parameter tracking
- **Sample quality** evolution during training

### Noise Schedule Analysis
- **β, α, ᾱ schedules** comparison
- **Signal-to-noise ratio** over time steps
- **Posterior variance** analysis

### Sample Quality
- **Image grids** at various training stages
- **Trajectory animations** showing denoising process
- **Metric evolution** (PSNR/SSIM/LPIPS/FID)

## 💡 Tips for Best Results

### Training
- **Start with MNIST** for quick experimentation (10-30 min training)
- **Use cosine schedule** for better sample quality
- **Monitor EMA weights** - often perform better than raw weights
- **Adjust batch size** based on GPU memory (32-128 typical)

### Sampling  
- **Use EMA weights** for final sampling
- **DDIM 50 steps** good speed/quality tradeoff
- **η=0.0** for deterministic, η=1.0 for stochastic
- **More steps** = higher quality but slower

### Evaluation
- **Generate 1000+ samples** for robust FID estimation
- **Use validation set** for PSNR/SSIM metrics
- **Compare multiple methods** (DDPM vs DDIM) 

## 🔧 Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch size or model channels
- **Training instability**: Lower learning rate, increase grad clipping
- **Poor sample quality**: Train longer, use EMA, check noise schedule
- **Slow sampling**: Use DDIM instead of DDPM, reduce steps

### Performance Optimization
- **Mixed precision** enabled by default (2× speedup)
- **DataLoader workers** set to 4 (adjust for your CPU)
- **Gradient checkpointing** available for memory reduction
- **Model size** adjustable via `model_channels` parameter

## 📚 References

- **DDPM Paper**: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) (Ho et al., 2020)
- **DDIM Paper**: [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) (Song et al., 2020)  
- **Improved DDPM**: [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) (Nichol & Dhariwal, 2021)

## 🎓 Learning Outcomes

After completing this implementation, you will have:
- ✅ **Deep understanding** of DDPM training and sampling
- ✅ **Practical experience** with diffusion model optimization  
- ✅ **Production-ready code** for real-world applications
- ✅ **Evaluation framework** for quality assessment
- ✅ **Visualization tools** for analysis and debugging
- ✅ **Extensible base** for advanced techniques (CFG, v-parameterization)

---
**Status**: ✅ Complete implementation  
**Estimated Training Time**: 10-30min (MNIST), 3-12hrs (CIFAR-10)  
**Difficulty**: ⭐⭐⭐⭐⭐