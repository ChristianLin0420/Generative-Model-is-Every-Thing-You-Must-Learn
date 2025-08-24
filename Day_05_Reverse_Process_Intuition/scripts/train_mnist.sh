#!/bin/bash

# Train DDPM on MNIST dataset
# This script trains a small UNet for educational purposes

set -e  # Exit on error

echo "Training DDPM on MNIST..."
echo "=========================="

# Change to project directory
cd "$(dirname "$0")/.."

# Create output directories
mkdir -p outputs/ckpts
mkdir -p outputs/logs
mkdir -p outputs/samples

# Train the model
python -m src.cli train \
    --config configs/mnist.yaml \
    "$@"

echo "Training completed!"
echo "Checkpoints saved in: outputs/ckpts/"
echo "Logs saved in: outputs/logs/"
echo "Sample images saved in: outputs/samples/"

# Generate some sample visualizations if training succeeded
if [ -f "outputs/ckpts/model_latest.pth" ]; then
    echo ""
    echo "Generating sample visualizations..."
    
    # Generate trajectory grid
    python -m src.cli sample.traj \
        --ckpt outputs/ckpts/model_latest.pth \
        --num-samples 4 \
        --animation
    
    # Generate sample grid
    python -m src.cli sample.grid \
        --ckpt outputs/ckpts/model_latest.pth \
        --num-samples 64
    
    # Create forward vs reverse comparison
    python -m src.cli viz.compare \
        --ckpt outputs/ckpts/model_latest.pth \
        --report
    
    echo "Visualizations generated!"
    echo "Check outputs/grids/ for trajectory and comparison plots"
    echo "Check outputs/samples/ for generated samples"
    echo "Check outputs/reports/ for analysis reports"
fi