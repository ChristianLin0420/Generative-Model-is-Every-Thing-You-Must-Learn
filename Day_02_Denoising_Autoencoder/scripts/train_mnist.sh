#!/bin/bash

# Day 2: Train Denoising Autoencoder on MNIST
# Quick training script for MNIST dataset

set -e  # Exit on any error

echo "🚀 Training Denoising Autoencoder on MNIST"
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "configs/mnist.yaml" ]; then
    echo "❌ Error: Please run this script from the Day_02_Denoising_Autoencoder directory"
    exit 1
fi

# Create output directories
echo "📁 Creating output directories..."
mkdir -p outputs/{ckpts,logs,grids,panels,reports}

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
    DEVICE="cuda:0"
else
    echo "💻 No GPU detected, using CPU"
    DEVICE="cpu"
fi

# Update config for current device (optional)
if [ "$DEVICE" = "cpu" ]; then
    echo "⚙️  Configuring for CPU training..."
    # You could modify the config here if needed
fi

# Start training
echo ""
echo "🏋️  Starting DAE training on MNIST..."
echo "Config: configs/mnist.yaml"
echo "Device: $DEVICE"
echo ""

python -m src.cli train \
    --config configs/mnist.yaml \
    --device $DEVICE

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully!"
    echo ""
    echo "📊 Generated outputs:"
    echo "  - Checkpoints: outputs/ckpts/"
    echo "  - Training logs: outputs/logs/"
    echo "  - Reconstruction grids: outputs/grids/"
    echo ""
    echo "Next steps:"
    echo "  - Evaluate model: python -m src.cli eval --config configs/mnist.yaml"
    echo "  - Generate visualizations: make viz"
    echo "  - Analyze limitations: python -m src.cli compare.limitations --config configs/mnist.yaml"
    echo ""
else
    echo "❌ Training failed!"
    exit 1
fi