#!/bin/bash
# One-liner script to generate MNIST sample grid using EMA checkpoint

set -e

echo "=== MNIST Sample Grid Generation ==="

# Change to project directory
cd "$(dirname "$0")/.."

# Run sampling with EMA checkpoint
python -m src.cli sample.grid \
    --config configs/mnist.yaml \
    --ckpt ema.pt

echo "Sample grid generated successfully!"
echo "Check outputs/grids/ for results."
