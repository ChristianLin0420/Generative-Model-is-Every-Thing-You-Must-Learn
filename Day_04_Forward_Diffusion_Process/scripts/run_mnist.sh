#!/bin/bash

# One-liner script to generate all figures for MNIST dataset

echo "Running all forward diffusion experiments for MNIST..."

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Run all experiments using the CLI
python -m src.cli run.all --config configs/mnist.yaml

echo "MNIST experiments completed!"
echo "Check outputs/ directory for results:"
echo "  - outputs/grids/mnist_traj_grid.png"
echo "  - outputs/animations/sample_000.gif" 
echo "  - outputs/plots/beta_alpha_snr.png"
echo "  - outputs/plots/snr_analysis.png"
echo "  - outputs/plots/hist_t_cosine.png"
echo "  - outputs/plots/mse_kl_curves.png"
echo "  - outputs/logs/forward_stats.csv"