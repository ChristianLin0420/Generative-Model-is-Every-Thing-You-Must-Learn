#!/bin/bash

# Day 2: Generate all required figures and visualizations
# Comprehensive visualization pipeline for DAE analysis

set -e

echo "📊 Generating all DAE visualizations and figures"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "configs/mnist.yaml" ]; then
    echo "❌ Error: Please run this script from the Day_02_Denoising_Autoencoder directory"
    exit 1
fi

# Check if model exists
if [ ! -f "outputs/ckpts/best_model.pth" ]; then
    echo "❌ Error: No trained model found. Please run training first:"
    echo "   make train  # or ./scripts/train_mnist.sh"
    exit 1
fi

# Use MNIST by default, but allow override
CONFIG=${1:-"configs/mnist.yaml"}
DATASET=$(basename "$CONFIG" .yaml)

echo "📋 Using configuration: $CONFIG"
echo "📊 Dataset: $DATASET"
echo ""

# Create all output directories
mkdir -p outputs/{grids,panels,reports}

echo "1️⃣  Generating reconstruction grids..."
# Multiple reconstruction grids with different noise levels
python -m src.cli viz.recon_grid \
    --config "$CONFIG" \
    --num-batches 5 \
    --num-images 8

if [ $? -eq 0 ]; then
    echo "   ✅ Reconstruction grids saved to outputs/grids/"
else
    echo "   ❌ Failed to generate reconstruction grids"
fi

echo ""
echo "2️⃣  Creating sigma sweep panels..."
# Panels showing same image across different noise levels
python -m src.cli viz.sigma_panel \
    --config "$CONFIG" \
    --num-images 6

if [ $? -eq 0 ]; then
    echo "   ✅ Sigma sweep panels saved to outputs/panels/"
else
    echo "   ❌ Failed to generate sigma sweep panels"
fi

echo ""
echo "3️⃣  Running comprehensive evaluation..."
# Full evaluation with metrics
python -m src.cli eval --config "$CONFIG"

if [ $? -eq 0 ]; then
    echo "   ✅ Evaluation results saved to outputs/logs/"
else
    echo "   ❌ Evaluation failed"
fi

echo ""
echo "4️⃣  Analyzing limitations vs generative models..."
# Comprehensive limitations analysis
python -m src.cli compare.limitations --config "$CONFIG"

if [ $? -eq 0 ]; then
    echo "   ✅ Limitations analysis saved to outputs/reports/"
else
    echo "   ❌ Limitations analysis failed"
fi

echo ""
echo "5️⃣  Creating training curves visualization..."
# Plot training curves if logs exist
if [ -f "outputs/logs/train_metrics.csv" ] && [ -f "outputs/logs/val_metrics.csv" ]; then
    python << EOF
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

try:
    train_df = pd.read_csv('outputs/logs/train_metrics.csv')
    val_df = pd.read_csv('outputs/logs/val_metrics.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss
    axes[0,0].plot(train_df['epoch'], train_df['loss'], 'b-', label='Train')
    axes[0,0].plot(val_df['epoch'], val_df['loss'], 'r-', label='Validation')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].set_title('Training Loss')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # PSNR
    axes[0,1].plot(train_df['epoch'], train_df['psnr'], 'b-', label='Train')
    axes[0,1].plot(val_df['epoch'], val_df['psnr'], 'r-', label='Validation')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('PSNR (dB)')
    axes[0,1].set_title('PSNR Progress')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # SSIM
    axes[1,0].plot(train_df['epoch'], train_df['ssim'], 'b-', label='Train')
    axes[1,0].plot(val_df['epoch'], val_df['ssim'], 'r-', label='Validation')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('SSIM')
    axes[1,0].set_title('SSIM Progress')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # MSE
    axes[1,1].plot(train_df['epoch'], train_df['mse'], 'b-', label='Train')
    axes[1,1].plot(val_df['epoch'], val_df['mse'], 'r-', label='Validation')
    axes[1,1].set_xlabel('Epoch')
    axes[1,1].set_ylabel('MSE')
    axes[1,1].set_title('Mean Squared Error')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path('outputs/reports').mkdir(exist_ok=True)
    plt.savefig('outputs/reports/training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("   ✅ Training curves saved to outputs/reports/training_curves.png")
except Exception as e:
    print(f"   ⚠️  Could not generate training curves: {e}")
EOF
else
    echo "   ⚠️  Training logs not found, skipping training curves"
fi

echo ""
echo "📋 Summary of generated figures:"
echo "================================"

# List all generated files
echo ""
echo "🖼️  Reconstruction Grids:"
if ls outputs/grids/*.png 1> /dev/null 2>&1; then
    ls -la outputs/grids/*.png | awk '{print "   " $9 " (" $5 " bytes)"}'
else
    echo "   None found"
fi

echo ""
echo "📊 Sigma Sweep Panels:"
if ls outputs/panels/*.png 1> /dev/null 2>&1; then
    ls -la outputs/panels/*.png | awk '{print "   " $9 " (" $5 " bytes)"}'
else
    echo "   None found"
fi

echo ""
echo "📈 Analysis Reports:"
if ls outputs/reports/*.{png,html} 1> /dev/null 2>&1; then
    ls -la outputs/reports/*.{png,html} 2>/dev/null | awk '{print "   " $9 " (" $5 " bytes)"}'
else
    echo "   None found"
fi

echo ""
echo "📋 Metrics and Logs:"
if ls outputs/logs/*.{csv,json} 1> /dev/null 2>&1; then
    ls -la outputs/logs/*.{csv,json} 2>/dev/null | awk '{print "   " $9 " (" $5 " bytes)"}'
else
    echo "   None found"
fi

echo ""
echo "✅ Figure generation complete!"
echo ""
echo "🎯 Key deliverables for Day 2:"
echo "  1. Reconstruction grids showing clean/noisy/reconstructed triplets"
echo "  2. Sigma sweep panels demonstrating noise robustness"
echo "  3. Comprehensive limitations analysis vs generative models"
echo "  4. Training progress curves and metrics"
echo ""
echo "💡 Next steps:"
echo "  - Open outputs/reports/limitations_analysis.html for detailed analysis"
echo "  - Review reconstruction quality in outputs/grids/"
echo "  - Check training progress in outputs/reports/training_curves.png"
echo "  - Compare with Day 1 noise analysis"
echo "  - Proceed to Day 3: VAE Refresher to see generative model advantages"