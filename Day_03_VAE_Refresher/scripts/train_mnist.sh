#!/bin/bash
# Train VAE on MNIST dataset

echo "🚀 Training VAE on MNIST dataset..."

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run training
python -m src.cli train --config configs/mnist.yaml

# Check if training was successful
if [ $? -eq 0 ]; then
    echo "✅ MNIST VAE training completed successfully!"
    echo "📊 Check outputs/logs/ for training logs and tensorboard data"
    echo "💾 Model checkpoints saved in outputs/ckpts/"
else
    echo "❌ MNIST VAE training failed!"
    exit 1
fi