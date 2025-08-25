#!/bin/bash

# Generate comprehensive report with training results and visualizations
# Usage: bash scripts/make_report.sh [CONFIG]

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default config
CONFIG_FILE="$PROJECT_ROOT/configs/mnist.yaml"

# Change to project root
cd "$PROJECT_ROOT"

# Parse arguments
if [ $# -ge 1 ] && [ -f "$1" ]; then
    CONFIG_FILE="$1"
elif [ $# -ge 1 ] && [ -f "configs/$1.yaml" ]; then
    CONFIG_FILE="$PROJECT_ROOT/configs/$1.yaml"
fi

echo "📊 Generating comprehensive DDPM report"
echo "Config: $CONFIG_FILE"

# Check if training has been done
OUTPUTS_DIR="outputs"
if [ ! -d "$OUTPUTS_DIR/ckpts" ] || [ ! "$(ls -A $OUTPUTS_DIR/ckpts)" ]; then
    echo "❌ No training checkpoints found. Please train a model first."
    echo ""
    if [[ "$CONFIG_FILE" == *"mnist"* ]]; then
        echo "Run: bash scripts/train_mnist.sh"
    else
        echo "Run: bash scripts/train_cifar10.sh"  
    fi
    exit 1
fi

REPORT_DIR="$OUTPUTS_DIR/reports"
mkdir -p "$REPORT_DIR"

echo "📁 Report directory: $REPORT_DIR"
echo ""

# Step 1: Generate training curves
echo "1️⃣  Plotting training curves..."
if [ -f "$OUTPUTS_DIR/logs/training_metrics.json" ]; then
    python -m src.cli viz.curves --log_dir "$OUTPUTS_DIR/logs" --output_dir "$OUTPUTS_DIR"
    echo "   ✅ Training curves saved"
else
    echo "   ⚠️  Training metrics not found, skipping curves"
fi

# Step 2: Plot schedules
echo "2️⃣  Plotting noise schedules..."
python -m src.cli viz.schedules --config "$CONFIG_FILE" --output_dir "$OUTPUTS_DIR"
echo "   ✅ Schedule plots saved"

# Step 3: Generate sample grid (if not exists)
echo "3️⃣  Generating sample grid..."
SAMPLE_FILE="$OUTPUTS_DIR/grids/samples_report_64.png"
if [ ! -f "$SAMPLE_FILE" ]; then
    python -m src.cli sample.grid --config "$CONFIG_FILE" --num_samples 64 --output_name "samples_report_64.png"
    echo "   ✅ Sample grid generated"
else
    echo "   ✅ Sample grid already exists"
fi

# Step 4: Create trajectory animation (if not exists)
echo "4️⃣  Creating trajectory animation..."
TRAJ_FILE="$OUTPUTS_DIR/animations/trajectory_report.gif"
if [ ! -f "$TRAJ_FILE" ]; then
    python -m src.cli sample.traj --config "$CONFIG_FILE" --output_name "trajectory_report.gif"
    echo "   ✅ Trajectory animation created"
else
    echo "   ✅ Trajectory animation already exists"
fi

# Step 5: Run evaluation
echo "5️⃣  Running model evaluation..."
EVAL_FILE="$OUTPUTS_DIR/logs/evaluation_results.json"
if [ ! -f "$EVAL_FILE" ]; then
    python -m src.cli eval --config "$CONFIG_FILE" --num_samples 500
    echo "   ✅ Evaluation completed"
else
    echo "   ✅ Evaluation results already exist"
fi

# Step 6: Create markdown report
echo "6️⃣  Creating markdown report..."
REPORT_FILE="$REPORT_DIR/ddmp_report.md"

cat > "$REPORT_FILE" << EOF
# DDPM Training Report

**Generated on:** $(date)  
**Config:** \`$(basename "$CONFIG_FILE")\`  
**Model:** DDPM with UNet backbone

## Training Overview

This report summarizes the training and evaluation of a Denoising Diffusion Probabilistic Model (DDPM).

### Configuration Summary

- **Dataset:** $(grep "name:" "$CONFIG_FILE" | head -1 | cut -d':' -f2 | tr -d ' ')
- **Image Size:** $(grep "image_size:" "$CONFIG_FILE" | head -1 | cut -d':' -f2 | tr -d ' ')
- **Timesteps:** $(grep "num_timesteps:" "$CONFIG_FILE" | head -1 | cut -d':' -f2 | tr -d ' ')
- **Schedule:** $(grep "schedule_type:" "$CONFIG_FILE" | head -1 | cut -d':' -f2 | tr -d ' ')
- **Model Channels:** $(grep "model_channels:" "$CONFIG_FILE" | head -1 | cut -d':' -f2 | tr -d ' ')

## Results

### Training Curves

EOF

# Add training curves if they exist
if [ -f "$OUTPUTS_DIR/curves/train_loss_curve.png" ]; then
    echo "![Training Loss](../curves/train_loss_curve.png)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

if [ -f "$OUTPUTS_DIR/curves/val_loss_curve.png" ]; then
    echo "![Validation Loss](../curves/val_loss_curve.png)" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

### Noise Schedules

![Noise Schedules](../curves/schedules.png)

### Generated Samples

![Sample Grid](../grids/samples_report_64.png)

### Reverse Diffusion Process

![Trajectory Animation](../animations/trajectory_report.gif)

## Evaluation Metrics

EOF

# Add evaluation results if they exist
if [ -f "$EVAL_FILE" ]; then
    echo "Evaluation metrics:" >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
    echo '```json' >> "$REPORT_FILE"
    cat "$EVAL_FILE" >> "$REPORT_FILE"
    echo '```' >> "$REPORT_FILE"
    echo "" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

## Model Details

### Architecture
- UNet-based denoising network
- Time-conditional FiLM layers
- Multi-head self-attention
- Exponential Moving Average (EMA)

### Training Details
- Mixed precision training (AMP)
- Gradient clipping
- Cosine learning rate schedule
- Periodic sampling during training

## Files Generated

### Checkpoints
- \`ckpts/best.pt\` - Best validation loss checkpoint
- \`ckpts/latest.pt\` - Latest training checkpoint  
- \`ckpts/ema.pt\` - EMA weights (used for sampling)

### Visualizations
- \`grids/\` - Sample image grids
- \`animations/\` - Reverse process animations
- \`curves/\` - Training and schedule plots

### Logs
- \`logs/training_metrics.json\` - Training history
- \`logs/evaluation_results.json\` - Evaluation metrics

## Usage Examples

\`\`\`bash
# Generate more samples
python -m src.cli sample.grid --config $CONFIG_FILE --num_samples 100

# Create custom animation
python -m src.cli sample.traj --config $CONFIG_FILE --method ddim --num_steps 20

# Run evaluation
python -m src.cli eval --config $CONFIG_FILE --all_methods

# Plot training curves
python -m src.cli viz.curves --log_dir outputs/logs
\`\`\`

---

*Report generated by DDPM Day 6 training pipeline*
EOF

echo "   ✅ Markdown report created: $REPORT_FILE"

# Step 7: Create summary
echo "7️⃣  Creating summary..."

echo ""
echo "📊 REPORT SUMMARY"
echo "=================="
echo ""
echo "📁 Report location: $REPORT_DIR"
echo "📄 Main report: $REPORT_FILE"
echo ""
echo "Generated files:"
echo "  📈 Training curves: outputs/curves/"
echo "  🎨 Sample grids: outputs/grids/"
echo "  🎬 Animations: outputs/animations/"
echo "  💾 Checkpoints: outputs/ckpts/"
echo "  📊 Evaluation: outputs/logs/evaluation_results.json"
echo ""
echo "✅ Comprehensive report generation completed!"
echo ""
echo "💡 Next steps:"
echo "   1. View the markdown report: cat $REPORT_FILE"
echo "   2. Open images in: outputs/grids/ and outputs/animations/"
echo "   3. Check training curves in: outputs/curves/"
echo "   4. Share results or continue experimentation!"

# Copy important files to report directory for easy access
cp -f "$CONFIG_FILE" "$REPORT_DIR/"
if [ -f "$OUTPUTS_DIR/grids/samples_report_64.png" ]; then
    cp -f "$OUTPUTS_DIR/grids/samples_report_64.png" "$REPORT_DIR/"
fi

echo ""
echo "📋 Report files copied to $REPORT_DIR for easy sharing"