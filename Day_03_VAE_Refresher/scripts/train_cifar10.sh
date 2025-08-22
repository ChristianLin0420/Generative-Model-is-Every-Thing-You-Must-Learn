#!/bin/bash
# Train VAE on CIFAR-10 dataset

echo "🚀 Training VAE on CIFAR-10 dataset..."

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run training
python -m src.cli train --config configs/cifar10.yaml

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ CIFAR-10 VAE training completed successfully!"
    echo "📊 Check outputs/logs/ for training logs and tensorboard data"
    echo "💾 Model checkpoints saved in outputs/ckpts/"
else
    echo "❌ CIFAR-10 VAE training failed!"
    exit 1
fi