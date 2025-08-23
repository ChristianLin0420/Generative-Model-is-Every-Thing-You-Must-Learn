#!/bin/bash

# Script to create schedule + trajectory + histogram bundles

echo "Creating figure bundles for forward diffusion analysis..."

# Ensure we're in the right directory
cd "$(dirname "$0")/.."

# Parse command line arguments
DATASET=${1:-mnist}  # Default to MNIST if no argument provided
CONFIG="configs/${DATASET}.yaml"

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file $CONFIG not found!"
    echo "Usage: $0 [mnist|cifar10]"
    exit 1
fi

echo "Using dataset: $DATASET"
echo "Config file: $CONFIG"

# Create schedule comparison plots
echo "Creating schedule comparison plots..."
python -m src.cli plots.schedules --config "$CONFIG"

# Create trajectory grid
echo "Creating trajectory grid..."
python -m src.cli traj.grid --config "$CONFIG"

# Create pixel histograms
echo "Creating pixel histograms..."
python -m src.cli plots.hist --config "$CONFIG"

# Create SNR analysis
echo "Creating SNR analysis..."
python -m src.cli plots.snr --config "$CONFIG"

# Create sample animation
echo "Creating sample animation..."
python -m src.cli traj.animate --config "$CONFIG" --idx 0

echo "Figure bundle creation completed!"
echo "Results saved to outputs/ directory:"
echo "  - Schedules: outputs/plots/beta_alpha_snr.png"
echo "  - Trajectory: outputs/grids/${DATASET}_traj_grid.png"
echo "  - Histograms: outputs/plots/hist_t_*.png"
echo "  - SNR Analysis: outputs/plots/snr_analysis.png"
echo "  - Animation: outputs/animations/sample_000.gif"