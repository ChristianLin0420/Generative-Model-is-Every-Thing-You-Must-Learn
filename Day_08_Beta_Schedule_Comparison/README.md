# Day 8: Beta Schedule Comparison for DDPM

A comprehensive comparison framework for analyzing different noise schedules (β-schedules) in Denoising Diffusion Probabilistic Models and their impact on training dynamics and sample quality.

## 🎯 Project Overview

This project implements and compares three key beta schedules used in diffusion models:
- **Linear**: Uniform noise addition progression
- **Cosine**: Preserves signal longer in early timesteps  
- **Quadratic**: Aggressive early noise corruption

## 🚀 Quick Start

```bash
# Setup environment
make setup

# Train all three schedules
make train_all

# Generate samples and visualizations
make sample_all

# Compare results and generate report
make compare

# Run tests
make test
```

## 📁 Project Structure

```
Day_08_Beta_Schedule_Comparison/
├── README.md
├── requirements.txt
├── pyproject.toml
├── Makefile                               # setup | train_all | sample_all | compare | test
├── configs/
│   ├── base.yaml                          # Common settings (dataset/model/T/optimizer)
│   ├── linear.yaml                        # Linear schedule config
│   ├── cosine.yaml                        # Cosine schedule config
│   └── quadratic.yaml                     # Quadratic schedule config
├── data/                                  # Auto-downloaded by torchvision
├── src/
│   ├── __init__.py
│   ├── utils.py                           # Seeding, device, ckpt I/O, timers, grid/gif savers
│   ├── dataset.py                         # MNIST/CIFAR loaders, normalization, labels (opt)
│   ├── schedules.py                       # β schedulers: linear/cosine/quadratic + α, ᾱ, SNR
│   ├── models/
│   │   ├── __init__.py
│   │   ├── time_embedding.py              # Sinusoidal + MLP
│   │   └── unet_small.py                  # UNet architecture for apples-to-apples comparison
│   ├── losses.py                          # DDPM ε-prediction loss
│   ├── trainer.py                         # Training loop w/ AMP, EMA, LR schedule, periodic sampling
│   ├── sampler.py                         # DDPM ancestral + DDIM(η=0) sampler
│   ├── visualize.py                       # Schedule plots; trajectory grids; animations
│   ├── quality.py                         # FID-proxy, PSNR/SSIM/LPIPS, step-time logging
│   └── cli.py                             # train | sample | plot.schedules | compare
├── scripts/
│   ├── train_all.sh                       # Trains 3 schedules (linear/cosine/quadratic)
│   ├── sample_all.sh                      # Samples grids + animations for each schedule
│   └── compare.sh                         # Computes metrics & renders comparison report
├── notebooks/
│   └── 08_beta_schedule_playground.ipynb  # Interactive T & schedule tweaking
├── outputs/                               # Generated during training/evaluation
│   ├── linear/
│   │   ├── ckpts/ logs/ curves/ grids/ animations/
│   ├── cosine/
│   │   ├── ckpts/ logs/ curves/ grids/ animations/
│   ├── quadratic/
│   │   ├── ckpts/ logs/ curves/ grids/ animations/
│   ├── plots/                             # Schedule comparison plots
│   └── comparison/                        # Cross-schedule analysis
└── tests/
    ├── test_schedules_shapes.py           # β∈(0,1), ᾱ monotone↓, shapes OK for all schedules
    ├── test_training_smoke.py             # 1-2 steps reduce loss for each schedule
    ├── test_sampler_smoke.py              # Reverse pass shapes, no NaNs
    └── test_visualize.py                  # Schedule plot + grid + GIF write
```

## 🧮 Beta Schedule Formulations

### Linear Schedule
```python
β_t = β_min + (β_max - β_min) * t/T
```
- **Default**: β_min = 1e-4, β_max = 0.02
- Simple, uniform noise addition

### Cosine Schedule  
```python
α̅_t = cos²(π/2 * (t/T + s)/(1 + s))
β_t = 1 - α̅_t/α̅_{t-1}
```
- **Parameter**: s = 0.008 (small offset)
- Slower early diffusion, preserves fine details

### Quadratic Schedule
```python
β_t = β_min + (β_max - β_min) * (t/T)²
```
- Quadratic progression
- Faster early noise addition

## 🎮 Usage Examples

### Training Individual Schedules
```bash
# Train with specific schedule
python -m src.cli train --config configs/linear.yaml
python -m src.cli train --config configs/cosine.yaml
python -m src.cli train --config configs/quadratic.yaml
```

### Sample Generation
```bash
# Generate DDPM samples
python -m src.cli sample.grid --config configs/cosine.yaml

# Generate DDIM samples (faster)
python -m src.cli sample.grid --config configs/cosine.yaml --ddim

# Generate trajectory visualization
python -m src.cli sample.traj --config configs/cosine.yaml
```

### Visualization & Analysis
```bash
# Plot schedule comparison
python -m src.cli plot.schedules --configs configs/linear.yaml configs/cosine.yaml configs/quadratic.yaml

# Full comparison with metrics
python -m src.cli compare --configs configs/linear.yaml configs/cosine.yaml configs/quadratic.yaml
```

## 📊 Expected Deliverables

After running the complete pipeline, you'll obtain:

### ✅ **Trained Models**
- `outputs/{linear,cosine,quadratic}/ckpts/ema.pt`
- `outputs/{linear,cosine,quadratic}/ckpts/best.pt`

### ✅ **Visualizations**
- `outputs/plots/schedules_overlay.png` - β, ᾱ, and SNR overlays
- `outputs/{run}/grids/samples_ema.png` - Sample grids (≥64 samples)
- `outputs/{run}/animations/reverse_traj.gif` - Reverse diffusion T→0
- `outputs/{run}/grids/trajectory_grid.png` - Fixed sample across time steps

### ✅ **Metrics & Comparison**
- `outputs/{run}/logs/metrics.csv` - Training loss, PSNR/SSIM, time/iter
- `outputs/comparison/comparison.csv` - Cross-schedule metrics (FID-proxy, LPIPS, PSNR, SSIM, steps/sec)
- `outputs/comparison/quality_vs_schedule.png` - Quality comparison plots
- `outputs/comparison/report.md` - Key findings and insights

### ✅ **Tests Pass**
- Schedule validity tests (β∈(0,1), ᾱ monotone decreasing)
- Training smoke tests (loss reduction)
- Sampling functionality tests
- Visualization pipeline tests

## 🔍 Key Insights

### **Linear Schedule**
- ✅ Simple implementation and interpretation
- ✅ Good baseline for comparison
- ⚠️ May not be optimal for fine detail preservation

### **Cosine Schedule**  
- ✅ Slower early SNR decay preserves fine details longer
- ✅ Often produces higher quality samples
- ✅ Better for image generation tasks
- ⚠️ Slightly more complex implementation

### **Quadratic Schedule**
- ✅ Faster training convergence possible
- ✅ Aggressive noise corruption
- ⚠️ May sacrifice fine detail quality
- ⚠️ Steep early SNR decline

## 🎓 Learning Outcomes

- **Schedule Impact**: Understanding how β-schedule choice affects diffusion dynamics
- **Training Analysis**: Experience with comparing training curves and convergence
- **Quality Metrics**: Hands-on experience with FID, PSNR, SSIM evaluation
- **Hyperparameter Sensitivity**: Insights into schedule parameter effects
- **Implementation Skills**: Building modular, configurable diffusion frameworks

## 🧪 Interactive Exploration

Explore schedules interactively in the Jupyter notebook:

```bash
cd notebooks/
jupyter notebook 08_beta_schedule_playground.ipynb
```

Features:
- 🔬 Interactive parameter adjustment (T, β_min, β_max, cosine_s)
- 📊 Real-time schedule visualization (β_t, ᾱ_t, SNR curves)
- 🎯 Forward diffusion process visualization
- 🧪 Custom schedule designer
- 📈 Mathematical property analysis

## 📖 Technical References

- **Improved DDPM** (Nichol & Dhariwal, 2021) - Cosine schedule introduction
- **DDPM** (Ho et al., 2020) - Original linear schedule formulation  
- **DDIM** (Song et al., 2020) - Deterministic sampling with schedule flexibility

## 🛠️ Configuration

All configurations use YAML with inheritance:

**Base Config** (`configs/base.yaml`):
```yaml
seed: 808
device: "cuda:0"
data: { dataset: "mnist", batch_size: 128, normalize: "minus_one_one" }
diffusion: { T: 1000, schedule: "cosine" }
model: { in_ch: 1, base_ch: 64, ch_mult: [1,2,2], time_embed_dim: 256 }
train: { epochs: 40, lr: 2e-4, opt: "adamw", ema: true, amp: true }
sample: { num_images: 64, ddim_steps: 50 }
```

**Schedule-Specific** configs inherit base and override `diffusion.schedule` and `log.run_name`.

---

**Time Estimate**: 4-6 hours (including training)  
**Difficulty**: ⭐⭐⭐⭐☆  
**Prerequisites**: Understanding of DDPM fundamentals