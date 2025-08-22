#!/bin/bash
# Generate samples and visualizations from trained VAE

CONFIG="${1:-configs/mnist.yaml}"
CHECKPOINT="${2:-outputs/ckpts/best.pt}"

echo "🎨 Generating samples and visualizations..."
echo "📝 Config: $CONFIG"
echo "💾 Checkpoint: $CHECKPOINT"

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint not found: $CHECKPOINT"
    echo "💡 Please train a model first using scripts/train_mnist.sh or scripts/train_cifar10.sh"
    exit 1
fi

echo "1️⃣ Evaluating model metrics..."
python -m src.cli eval --config "$CONFIG" --checkpoint "$CHECKPOINT"

echo "2️⃣ Generating prior samples..."
python -m src.cli sample.prior --config "$CONFIG" --checkpoint "$CHECKPOINT"

echo "3️⃣ Creating interpolations..."
python -m src.cli sample.interpolate --config "$CONFIG" --checkpoint "$CHECKPOINT"

echo "4️⃣ Creating reconstruction grids..."
python -m src.cli viz.recon_grid --config "$CONFIG" --checkpoint "$CHECKPOINT"

echo "5️⃣ Creating latent traversals..."
python -m src.cli viz.traverse --config "$CONFIG" --checkpoint "$CHECKPOINT"

# Only create 2D scatter plot if using 2D latent space
if grep -q "latent_dim: 2" "$CONFIG"; then
    echo "6️⃣ Creating 2D latent scatter plot..."
    python -m src.cli viz.latent_scatter --config "$CONFIG" --checkpoint "$CHECKPOINT"
else
    echo "6️⃣ Skipping 2D scatter plot (latent_dim != 2)"
fi

# Check if all commands were successful
if [ $? -eq 0 ]; then
    echo "✅ All samples and visualizations generated successfully!"
    echo "🎯 Results saved in:"
    echo "   📊 outputs/grids/ - Visualization grids"
    echo "   🎨 outputs/samples/ - Generated samples"
    echo "   📈 outputs/plots/ - Analysis plots"
    echo "   📋 outputs/logs/metrics.csv - Evaluation metrics"
else
    echo "❌ Sample generation failed!"
    exit 1
fi