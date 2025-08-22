#!/bin/bash
# Compare VAE with DAE from Day 2

VAE_CONFIG="${1:-configs/mnist.yaml}"
DAE_CHECKPOINT="${2:-../Day_02_Denoising_Autoencoder/outputs/ckpts/best.pt}"
VAE_CHECKPOINT="${3:-outputs/ckpts/best.pt}"

echo "🔍 Comparing VAE with DAE..."
echo "📝 VAE Config: $VAE_CONFIG"
echo "🤖 DAE Checkpoint: $DAE_CHECKPOINT"
echo "🧠 VAE Checkpoint: $VAE_CHECKPOINT"

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if checkpoints exist
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "❌ VAE checkpoint not found: $VAE_CHECKPOINT"
    echo "💡 Please train a VAE first using scripts/train_mnist.sh"
    exit 1
fi

if [ ! -f "$DAE_CHECKPOINT" ]; then
    echo "❌ DAE checkpoint not found: $DAE_CHECKPOINT"
    echo "💡 Please ensure Day 2 DAE training is completed"
    echo "💡 Or provide the correct path to DAE checkpoint as second argument"
    exit 1
fi

echo "⚖️ Running VAE vs DAE comparison..."
python -m src.cli compare.dae \
    --config "$VAE_CONFIG" \
    --checkpoint "$VAE_CHECKPOINT" \
    --dae_ckpt "$DAE_CHECKPOINT"

# Check if comparison was successful
if [ $? -eq 0 ]; then
    echo "✅ VAE vs DAE comparison completed successfully!"
    echo "📊 Results saved in outputs/reports/"
    echo "🎯 Check the generated report for detailed analysis"
else
    echo "❌ VAE vs DAE comparison failed!"
    echo "💡 Make sure both models are compatible and checkpoints are valid"
    exit 1
fi