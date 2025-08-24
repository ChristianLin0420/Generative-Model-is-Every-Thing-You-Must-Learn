#!/bin/bash

# Train DDPM on CIFAR-10 dataset
# This script trains a larger UNet for more complex images

set -e  # Exit on error

echo "Training DDPM on CIFAR-10..."
echo "============================"

# Change to project directory
cd "$(dirname "$0")/.."

# Create output directories
mkdir -p outputs/ckpts
mkdir -p outputs/logs
mkdir -p outputs/samples

# Check if GPU is available
if python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
    echo "GPU detected - training will use CUDA"
else
    echo "Warning: No GPU detected - training will be slow on CPU"
    echo "Consider using a smaller model or fewer epochs for CPU training"
fi

# Train the model
python -m src.cli train \
    --config configs/cifar10.yaml \
    "$@"

echo "Training completed!"
echo "Checkpoints saved in: outputs/ckpts/"
echo "Logs saved in: outputs/logs/"
echo "Sample images saved in: outputs/samples/"

# Generate some sample visualizations if training succeeded
if [ -f "outputs/ckpts/model_latest.pth" ]; then
    echo ""
    echo "Generating sample visualizations..."
    
    # Generate trajectory grid (fewer timesteps for complex images)
    python -m src.cli sample.traj \
        --ckpt outputs/ckpts/model_latest.pth \
        --num-samples 4 \
        --sampler ddim \
        --animation
    
    # Generate sample grid
    python -m src.cli sample.grid \
        --ckpt outputs/ckpts/model_latest.pth \
        --num-samples 36 \
        --sampler ddim \
        --steps 50
    
    # Create forward vs reverse comparison
    python -m src.cli viz.compare \
        --ckpt outputs/ckpts/model_latest.pth \
        --report
    
    echo "Visualizations generated!"
    echo "Check outputs/grids/ for trajectory and comparison plots"
    echo "Check outputs/samples/ for generated samples"
    echo "Check outputs/reports/ for analysis reports"
fi