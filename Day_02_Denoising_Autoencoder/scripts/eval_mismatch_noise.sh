#!/bin/bash

# Day 2: Evaluate model robustness to unseen noise levels
# Test how well the model generalizes to noise levels not seen during training

set -e

echo "🧪 Testing DAE robustness to unseen noise levels"
echo "==============================================="

# Check if we're in the right directory
if [ ! -f "configs/mnist.yaml" ]; then
    echo "❌ Error: Please run this script from the Day_02_Denoising_Autoencoder directory"
    exit 1
fi

# Check if model exists
if [ ! -f "outputs/ckpts/best_model.pth" ]; then
    echo "❌ Error: No trained model found. Please run training first:"
    echo "   ./scripts/train_mnist.sh"
    exit 1
fi

CONFIG=${1:-"configs/mnist.yaml"}
DATASET=$(basename "$CONFIG" .yaml)

echo "📋 Configuration: $CONFIG"
echo "📊 Dataset: $DATASET"
echo ""

# Standard evaluation first
echo "1️⃣  Running standard evaluation..."
python -m src.cli eval --config "$CONFIG"

# Generate detailed visualizations
echo ""
echo "2️⃣  Generating reconstruction grids..."
python -m src.cli viz.recon_grid --config "$CONFIG" --num-batches 3

echo ""
echo "3️⃣  Creating sigma sweep panels..."
python -m src.cli viz.sigma_panel --config "$CONFIG" --num-images 4

# Analyze limitations
echo ""
echo "4️⃣  Analyzing model limitations..."
python -m src.cli compare.limitations --config "$CONFIG"

echo ""
echo "✅ Robustness evaluation complete!"
echo ""
echo "📊 Check these outputs:"
echo "  - Evaluation metrics: outputs/logs/evaluation_results.json"
echo "  - Reconstruction grids: outputs/grids/"
echo "  - Sigma sweep panels: outputs/panels/"
echo "  - Limitations analysis: outputs/reports/limitations_analysis.html"
echo ""
echo "🔍 Key things to look for:"
echo "  1. How does PSNR/SSIM degrade with higher noise levels?"
echo "  2. Are reconstructions over-smoothed?"
echo "  3. Is there diversity collapse (similar outputs from different noisy inputs)?"
echo "  4. How much high-frequency detail is preserved?"
echo ""
echo "💡 Tip: Open outputs/reports/limitations_analysis.html in your browser"
echo "     for a comprehensive analysis of DAE limitations."