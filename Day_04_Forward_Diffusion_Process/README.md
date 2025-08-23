# Day 4: Forward Diffusion Process 🌊

A complete implementation of the forward diffusion process for DDPMs, featuring interactive visualizations, comprehensive statistical analysis, and multiple noise schedules.

## 🎯 Overview

This implementation provides a deep dive into the mathematical foundation of diffusion models through the forward noising process. You can explore how images gradually transform from clean data to pure noise, analyze different noise schedules, and understand the statistical properties that make DDPMs work.

## ✨ Key Features

- **📈 Multiple Noise Schedules**: Linear, cosine, and sigmoid schedules with comparative analysis
- **🎨 Interactive Visualizations**: Trajectory grids, animations, and real-time parameter tuning
- **📊 Statistical Analysis**: SNR tracking, MSE/KL convergence, pixel distribution evolution  
- **🧪 Comprehensive Testing**: Unit tests for mathematical correctness and edge cases
- **💻 CLI Interface**: Complete command-line tools for batch processing
- **📓 Interactive Notebook**: Jupyter notebook with widgets for exploration

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
make install

# Or manually
pip install -r requirements.txt
pip install -e .
```

### Run Experiments
```bash
# Generate all MNIST results
make run-mnist

# Generate all CIFAR-10 results  
make run-cifar10

# Run both datasets
make run-all
```

### Interactive Exploration
```bash
# Launch Jupyter notebook
make notebook

# Or directly
jupyter lab notebooks/04_forward_diffusion.ipynb
```

## 🗂️ Project Structure

```
Day_04_Forward_Diffusion_Process/
├── README.md                    # This file
├── requirements.txt            # Dependencies
├── pyproject.toml             # Package configuration
├── Makefile                   # Build automation
├── configs/                   # YAML configurations
│   ├── mnist.yaml            # MNIST experiment settings
│   └── cifar10.yaml          # CIFAR-10 experiment settings
├── src/                       # Core implementation
│   ├── utils.py              # Utilities (seed, device, normalization)
│   ├── dataset.py            # MNIST/CIFAR loaders
│   ├── ddpm_schedules.py     # Noise schedule implementations
│   ├── forward.py            # Forward diffusion mathematics
│   ├── stats.py              # Statistical analysis
│   ├── visualize.py          # Plotting and animation
│   └── cli.py                # Command-line interface
├── scripts/                   # Automation scripts
│   ├── run_mnist.sh          # One-click MNIST experiments
│   ├── run_cifar10.sh        # One-click CIFAR-10 experiments
│   └── make_figures.sh       # Custom figure generation
├── notebooks/                 # Interactive exploration
│   └── 04_forward_diffusion.ipynb
├── tests/                     # Test suite
│   ├── test_schedules.py     # Schedule validation
│   ├── test_forward_xt_x0.py # Forward process tests
│   └── test_visualize.py     # Visualization tests
└── outputs/                   # Generated results
    ├── grids/                # Trajectory visualizations
    ├── animations/           # Diffusion animations
    ├── plots/                # Statistical plots
    └── logs/                 # CSV data logs
```

## 🧮 Mathematical Implementation

### Forward Process
- **Sequential**: `q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)`
- **Closed-form**: `q(x_t|x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)`  
- **Cumulative**: `ᾱ_t = ∏_{s=1}^t α_s` where `α_s = 1 - β_s`

### Noise Schedules
- **Linear**: `β_t` increases linearly from `β_1` to `β_T`
- **Cosine**: Smooth S-curve based on cosine function for gradual noise addition
- **Sigmoid**: Steep middle transition with gentle start/end

### Statistical Analysis
- **SNR**: Signal-to-noise ratio `ᾱ_t / (1 - ᾱ_t)` tracked in dB
- **MSE**: Mean squared error between `x_t` and `x_0`
- **KL Divergence**: `KL(q(x_t|x_0) || N(0,I))` convergence measurement

## 💻 Usage Examples

### CLI Commands
```bash
# Individual experiments
python -m src.cli traj.grid --config configs/mnist.yaml
python -m src.cli traj.animate --config configs/cifar10.yaml --idx 5
python -m src.cli plots.schedules --config configs/mnist.yaml
python -m src.cli plots.snr --config configs/cifar10.yaml
python -m src.cli plots.hist --config configs/mnist.yaml

# Batch processing
python -m src.cli run.all --config configs/mnist.yaml
```

### Makefile Targets
```bash
# Development
make test          # Run test suite
make lint          # Check code quality
make clean         # Clean outputs

# Experiments
make trajectory-grid DATASET=mnist
make plot-schedules DATASET=cifar10
make figures DATASET=mnist  # Custom figure bundle

# Convenience
make quick-mnist   # Fast trajectory + schedules
make examples      # Show usage examples
```

### Python API
```python
from src.ddpm_schedules import get_ddpm_schedule
from src.forward import q_xt_given_x0, sample_trajectory
from src.visualize import create_trajectory_grid

# Get noise schedule
betas, alphas, alpha_bars = get_ddpm_schedule(1000, "cosine")

# Sample at specific time step
x_t, noise = q_xt_given_x0(x0, t, alpha_bars)

# Create full trajectory
trajectory = sample_trajectory(x0, 1000, betas, alpha_bars)

# Visualize results
create_trajectory_grid(x0, [0, 100, 500, 999], alpha_bars, "trajectory.png")
```

## 📊 Expected Outputs

After running experiments, you'll find:

### Visualizations
- **`outputs/grids/mnist_traj_grid.png`**: 16 images × selected time steps
- **`outputs/animations/sample_000.gif`**: Single image diffusion animation
- **`outputs/plots/beta_alpha_snr.png`**: Schedule comparison curves
- **`outputs/plots/hist_t_cosine.png`**: Pixel distribution evolution

### Data Logs
- **`outputs/logs/forward_stats.csv`**: Complete statistical time series
  - Columns: `[t, beta, alpha_bar, snr_db, mse_to_x0, kl_to_unit]`

### Analysis Results
- **SNR Thresholds**: When images become more noise than signal
- **Schedule Efficiency**: Comparative time-to-threshold analysis
- **Convergence Verification**: Statistical validation of N(0,1) convergence

## 🎓 Key Insights

### Schedule Comparison
- **Cosine**: Gradual noise addition, preserves structure longer
- **Linear**: Uniform progression, standard DDPM default
- **Sigmoid**: Sharp middle transition, aggressive noise addition

### Critical Time Steps  
- **t ≈ 200-400**: Image structure begins degrading significantly
- **SNR = -5dB**: Typical "more noise than signal" threshold
- **t ≈ 800+**: Convergence to pure Gaussian noise

### Mathematical Verification
- **MSE Convergence**: Approaches `1 - ᾱ_t` for unit-variance data
- **Pixel Statistics**: Mean → 0, Std → 1 as t → T
- **Distribution Shape**: Gradual transformation to standard Gaussian

## 🧪 Testing

Run the comprehensive test suite:
```bash
make test                    # Full test suite
make test-coverage          # With coverage report
pytest tests/ -v           # Verbose output
pytest tests/test_schedules.py  # Specific module
```

Tests verify:
- ✅ Schedule monotonicity and boundary conditions  
- ✅ Forward sampling mathematical correctness
- ✅ Statistical convergence properties
- ✅ Visualization output generation

## 🔧 Development

### Code Quality
```bash
make format     # Auto-format with black/isort
make lint       # Check with flake8
make check      # Format + lint + test
```

### Interactive Development
```bash
make notebook   # Launch Jupyter Lab
make dev-setup  # Install development dependencies
```

## 📚 Mathematical Background

This implementation follows the mathematical framework from:
- **Ho et al. (2020)**: "Denoising Diffusion Probabilistic Models" 
- **Sohl-Dickstein et al. (2015)**: "Deep Unsupervised Learning using Nonequilibrium Thermodynamics"

Key mathematical concepts:
- Markov chain forward process
- Gaussian reparameterization trick
- Signal-to-noise ratio analysis  
- Statistical convergence theory

## 🤝 Contributing

The implementation is modular and extensible:
- Add new noise schedules in `ddpm_schedules.py`
- Extend statistical analysis in `stats.py`
- Create custom visualizations in `visualize.py`
- Add dataset support in `dataset.py`

## 🎯 Next Steps

This implementation prepares you for:
- **Day 5**: Reverse process and denoising
- **Day 6**: Complete DDPM training loop
- **Day 7**: Advanced sampling techniques

---

**⏱️ Time Investment**: 3-4 hours to explore fully  
**🎯 Difficulty**: ⭐⭐⭐☆☆ (Mathematical foundations)  
**🔗 Prerequisite**: Basic understanding of Gaussian processes