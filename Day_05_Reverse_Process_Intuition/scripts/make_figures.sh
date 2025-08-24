#!/bin/bash

# Generate all figures and visualizations for Day 5
# This creates a comprehensive set of plots and analyses

set -e  # Exit on error

echo "Generating all Day 5 figures..."
echo "==============================="

# Change to project directory
cd "$(dirname "$0")/.."

# Default checkpoint path
CHECKPOINT="${1:-outputs/ckpts/model_latest.pth}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Please train a model first or specify a valid checkpoint path"
    echo "Usage: $0 [checkpoint_path]"
    echo "Example: $0 outputs/ckpts/model_best.pth"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
echo ""

# Create output directories
mkdir -p outputs/grids
mkdir -p outputs/samples
mkdir -p outputs/animations
mkdir -p outputs/reports

# 1. Generate reverse trajectory grids
echo "1. Generating reverse trajectory grids..."
echo "   - DDPM trajectories..."
python -m src.cli sample.traj \
    --ckpt "$CHECKPOINT" \
    --num-samples 4 \
    --sampler ddpm \
    --output outputs/grids/reverse_traj_ddpm.png \
    --animation

echo "   - DDIM trajectories..."
python -m src.cli sample.traj \
    --ckpt "$CHECKPOINT" \
    --num-samples 4 \
    --sampler ddim \
    --output outputs/grids/reverse_traj_ddim.png \
    --animation

# 2. Generate sample grids
echo ""
echo "2. Generating sample grids..."
echo "   - DDPM samples (64 images)..."
python -m src.cli sample.grid \
    --ckpt "$CHECKPOINT" \
    --num-samples 64 \
    --sampler ddpm \
    --output outputs/samples/samples_ddpm_64.png

echo "   - DDIM samples (64 images, 50 steps)..."
python -m src.cli sample.grid \
    --ckpt "$CHECKPOINT" \
    --num-samples 64 \
    --sampler ddim \
    --steps 50 \
    --output outputs/samples/samples_ddim_64.png

echo "   - DDIM samples (64 images, 20 steps - fast)..."
python -m src.cli sample.grid \
    --ckpt "$CHECKPOINT" \
    --num-samples 64 \
    --sampler ddim \
    --steps 20 \
    --output outputs/samples/samples_ddim_fast_64.png

# 3. Generate forward vs reverse comparison
echo ""
echo "3. Generating forward vs reverse comparisons..."
python -m src.cli viz.compare \
    --ckpt "$CHECKPOINT" \
    --output outputs/grids/forward_vs_reverse.png \
    --report

# 4. Run comprehensive evaluation
echo ""
echo "4. Running evaluation metrics..."
python -m src.cli eval \
    --ckpt "$CHECKPOINT" \
    --num-samples 200 \
    --sampler ddpm \
    --output outputs/reports/evaluation_ddpm

python -m src.cli eval \
    --ckpt "$CHECKPOINT" \
    --num-samples 200 \
    --sampler ddim \
    --output outputs/reports/evaluation_ddim

echo ""
echo "=========================================="
echo "All figures generated successfully!"
echo "=========================================="
echo ""
echo "Generated files:"
echo ""
echo "Trajectory Visualizations:"
echo "  📊 outputs/grids/reverse_traj_ddpm.png"
echo "  🎬 outputs/grids/reverse_traj_ddpm.gif"
echo "  📊 outputs/grids/reverse_traj_ddim.png"
echo "  🎬 outputs/grids/reverse_traj_ddim.gif"
echo ""
echo "Sample Grids:"
echo "  🖼️  outputs/samples/samples_ddpm_64.png"
echo "  🖼️  outputs/samples/samples_ddim_64.png"
echo "  ⚡ outputs/samples/samples_ddim_fast_64.png"
echo ""
echo "Analysis and Comparisons:"
echo "  📈 outputs/grids/forward_vs_reverse.png"
echo "  📄 outputs/grids/forward_vs_reverse_report/"
echo ""
echo "Evaluation Results:"
echo "  📊 outputs/reports/evaluation_ddpm_*.csv"
echo "  📊 outputs/reports/evaluation_ddim_*.csv"
echo "  📄 outputs/reports/evaluation_*_summary.txt"
echo ""
echo "Key Observations:"
echo "- Compare DDPM vs DDIM sampling quality and speed"
echo "- Check forward_vs_reverse_report/ for detailed analysis"
echo "- Review evaluation summaries for quantitative metrics"
echo ""
echo "Next steps:"
echo "- Examine the generated visualizations"
echo "- Read the analysis reports"
echo "- Experiment with different hyperparameters"