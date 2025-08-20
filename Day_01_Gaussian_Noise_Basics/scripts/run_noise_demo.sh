#!/bin/bash

# Day 1: Gaussian Noise Basics - Complete Demo Script
# This script runs the full demonstration pipeline

set -e  # Exit on any error

echo "🚀 Starting Day 1: Gaussian Noise Basics Demo"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "configs/default.yaml" ]; then
    echo "❌ Error: Please run this script from the Day_01_Gaussian_Noise_Basics directory"
    exit 1
fi

# Create output directories
echo "📁 Creating output directories..."
mkdir -p outputs/{grids,animations,logs}

# Step 1: Generate theoretical statistics
echo ""
echo "📊 Step 1: Computing theoretical noise statistics..."
python -m src.cli stats --config configs/default.yaml

# Step 2: Generate noised image grids
echo ""
echo "🖼️  Step 2: Generating noised image grids..."
python -m src.cli add-noise --config configs/default.yaml --num-batches 2

# Step 3: Create noise progression animation
echo ""
echo "🎬 Step 3: Creating noise progression animation..."
python -m src.cli animate --config configs/default.yaml --num-images 16 --out outputs/animations/mnist_noise.gif

# Step 4: List generated outputs
echo ""
echo "✅ Demo completed! Generated outputs:"
echo "📁 Grids:"
ls -la outputs/grids/ | grep -E '\.(png|jpg)$' || echo "   No image files found"

echo ""
echo "📁 Animations:"
ls -la outputs/animations/ | grep -E '\.(gif|mp4)$' || echo "   No animation files found"

echo ""
echo "📁 Logs:"
ls -la outputs/logs/ | grep -E '\.(csv|png)$' || echo "   No log files found"

echo ""
echo "🎉 Day 1 demonstration completed successfully!"
echo ""
echo "Next steps:"
echo "  - View image grids in outputs/grids/"
echo "  - Watch animation at outputs/animations/mnist_noise.gif"  
echo "  - Check metrics in outputs/logs/noise_metrics.csv"
echo "  - Open Jupyter notebook: notebooks/01_gaussian_noise_exploration.ipynb"
echo ""