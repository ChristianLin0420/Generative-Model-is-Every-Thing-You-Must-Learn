# Day 5: Reverse Process Intuition

**Complete DDPM implementation with reverse diffusion, UNet denoising, and comprehensive evaluation tools.**

---

## 🎯 Objective

Build intuition for the reverse diffusion process through a complete DDPM implementation. This project focuses on understanding how neural networks learn to invert the noise addition process, with modern features like mixed precision training, EMA, and multiple sampling algorithms.

## 📋 What You'll Learn

### Core DDPM Concepts
- **Forward vs Reverse Process**: How diffusion adds noise and how we learn to remove it
- **Noise Scheduling**: Linear, cosine, and quadratic beta schedules
- **Neural Parameterization**: Different ways to parameterize the denoising function (ε, x₀, v)
- **Time Conditioning**: How to make neural networks aware of the noise level

### Implementation Skills  
- **UNet Architecture**: Time-conditioned U-Net with attention and FiLM normalization
- **Training Loop**: Modern training with AMP, EMA, gradient clipping, and learning rate scheduling
- **Sampling Algorithms**: Both DDPM ancestral sampling and DDIM deterministic sampling
- **Evaluation Metrics**: PSNR, SSIM, LPIPS for quantitative assessment

## 🏗️ Project Structure

```
Day_05_Reverse_Process_Intuition/
├── README.md                     # This file
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
├── Makefile                     # Common commands
├── configs/
│   ├── mnist.yaml              # MNIST training config
│   └── cifar10.yaml            # CIFAR-10 training config
├── data/                       # Auto-downloaded datasets
├── src/
│   ├── utils.py                # Utilities: seed, device, checkpoints, EMA
│   ├── dataset.py              # Dataset loaders with normalization
│   ├── ddpm_schedules.py       # Beta schedules, alpha computations
│   ├── losses.py               # ε-pred, x₀-pred, v-pred losses
│   ├── trainer.py              # Training loop with AMP, EMA, logging
│   ├── sampler.py              # DDPM and DDIM sampling algorithms
│   ├── eval.py                 # PSNR, SSIM, LPIPS evaluation
│   ├── visualize.py            # Trajectory grids, animations, comparisons
│   ├── compare_forward_reverse.py  # Day 4 vs Day 5 comparison
│   ├── models/
│   │   ├── time_embedding.py   # Sinusoidal time embeddings + FiLM
│   │   └── unet_tiny.py        # Lightweight UNet with time conditioning
│   └── cli.py                  # Command-line interface
├── scripts/
│   ├── train_mnist.sh          # Train on MNIST
│   ├── train_cifar10.sh        # Train on CIFAR-10
│   ├── sample_trajectories.sh  # Generate trajectory visualizations
│   └── make_figures.sh         # Generate all figures
├── notebooks/
│   └── 05_reverse_process_playground.ipynb  # Interactive exploration
├── outputs/                    # Generated content
│   ├── ckpts/                  # Model checkpoints
│   ├── logs/                   # Tensorboard logs
│   ├── grids/                  # Trajectory and comparison grids
│   ├── animations/             # GIF/MP4 reverse process animations
│   ├── samples/                # Generated sample grids
│   └── reports/                # Evaluation reports and analysis
└── tests/
    ├── test_unet_tiny.py       # Model architecture tests
    ├── test_loss_and_train.py  # Training pipeline tests
    ├── test_sampler_shapes.py  # Sampling algorithm tests
    └── test_visualize.py       # Visualization tests
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone and enter the directory
cd Day_05_Reverse_Process_Intuition

# Install dependencies
pip install -r requirements.txt
# or
make install
```

### 2. Train Your First Model
```bash
# Quick MNIST training (5-10 minutes)
make train-mnist-quick

# Full MNIST training
scripts/train_mnist.sh

# CIFAR-10 training (requires GPU, 30+ minutes)
scripts/train_cifar10.sh
```

### 3. Generate Visualizations
```bash
# Generate all figures and analysis
scripts/make_figures.sh

# Or individual commands:
python -m src.cli sample.grid --ckpt outputs/ckpts/model_latest.pth
python -m src.cli sample.traj --ckpt outputs/ckpts/model_latest.pth --animation
python -m src.cli viz.compare --ckpt outputs/ckpts/model_latest.pth --report
```

### 4. Interactive Exploration
```bash
jupyter notebook notebooks/05_reverse_process_playground.ipynb
```

## 📊 Key Equations & Concepts

### Forward Process (Day 4 → Day 5 Connection)
The forward process adds noise according to:
```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
x_t = √ᾱ_t x_0 + √(1-ᾱ_t) ε,  where ε ~ N(0,I)
```

### Reverse Process (What We Learn)
The reverse process removes noise:
```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t I)
μ_θ(x_t, t) = 1/√α_t (x_t - β_t/√(1-ᾱ_t) ε_θ(x_t, t))
```

### Training Objective (Simple DDPM Loss)
```
L = E_{t,x_0,ε} [||ε - ε_θ(x_t, t)||²]
```

Where:
- `ε_θ` is our neural network (UNet) that predicts noise
- `t` is sampled uniformly from {1, 2, ..., T}
- `x_t` is created by adding noise to `x_0`

## 🎨 Generated Artifacts

After training, you'll have:

### Visual Outputs
- **Trajectory Grids**: Step-by-step reverse diffusion visualization
- **Sample Grids**: 64+ clean samples generated from noise  
- **Animations**: GIF/MP4 showing the denoising process
- **Forward vs Reverse**: Side-by-side comparison with Day 4

### Analysis Reports
- **Evaluation Metrics**: PSNR/SSIM reconstruction quality
- **Comparison Reports**: Forward vs reverse process analysis
- **Training Logs**: Loss curves and sample evolution

### Model Checkpoints
- **EMA Weights**: Exponential moving average for stable sampling
- **Training State**: Full checkpoint with optimizer state
- **Configuration**: Saved hyperparameters for reproducibility

## 🔧 Advanced Usage

### Custom Training
```bash
python -m src.cli train \
    --config configs/mnist.yaml \
    --epochs 50 \
    --batch-size 256 \
    --learning-rate 1e-4
```

### Different Sampling Methods
```bash
# DDPM ancestral sampling (stochastic, high quality)
python -m src.cli sample.grid --sampler ddpm --num-samples 64

# DDIM sampling (deterministic, fast)
python -m src.cli sample.grid --sampler ddim --steps 20 --num-samples 64
```

### Comprehensive Evaluation
```bash
python -m src.cli eval \
    --ckpt outputs/ckpts/model_best.pth \
    --num-samples 1000 \
    --sampler ddpm
```

## 📈 Expected Results

### MNIST Results
- **Training Time**: 5-15 minutes on GPU
- **Model Size**: ~500K parameters  
- **Sample Quality**: Clean digit generation
- **PSNR**: 25+ dB reconstruction quality

### CIFAR-10 Results  
- **Training Time**: 30+ minutes on GPU
- **Model Size**: ~5M parameters
- **Sample Quality**: Recognizable objects with some artifacts
- **PSNR**: 20+ dB reconstruction quality

## 🎓 Learning Outcomes

By completing this project, you'll understand:

✅ **How reverse diffusion works**: The mathematical relationship between forward and reverse processes

✅ **Neural network denoising**: How time-conditioned UNets learn to predict and remove noise

✅ **Training dynamics**: Why EMA, mixed precision, and proper scheduling matter for diffusion models  

✅ **Sampling trade-offs**: Speed vs quality differences between DDPM and DDIM

✅ **Evaluation methods**: Quantitative metrics for generative model assessment

## 🚨 Common Issues & Solutions

### Training Issues
- **GPU Memory**: Reduce batch size or model channels in config
- **Slow Training**: Use mixed precision (`mixed_precision: true`)
- **Poor Samples**: Train longer or adjust learning rate

### Sampling Issues  
- **Blurry Samples**: Use EMA weights or increase model capacity
- **Mode Collapse**: Try different beta schedules (cosine vs linear)
- **Slow Sampling**: Use DDIM with fewer steps (20-50)

### Technical Issues
- **Import Errors**: Ensure you're in the project root directory
- **Missing Data**: Datasets auto-download on first use
- **Checkpoint Issues**: Check file paths and device compatibility

## 📚 References & Further Reading

- **DDPM Paper**: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
- **DDIM Paper**: "Denoising Diffusion Implicit Models" (Song et al., 2021)  
- **Score Matching**: "Generative Modeling by Estimating Gradients" (Song & Ermon, 2019)
- **UNet Architecture**: "U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger et al., 2015)

## 🤝 Acknowledgments

This implementation builds upon concepts from Day 4 (Forward Process) and provides a complete educational framework for understanding modern diffusion models.

---

**Time Estimate**: 2-4 hours (exploration + training)  
**Difficulty**: ⭐⭐⭐⭐☆
**Prerequisites**: Day 4 (Forward Diffusion Process)