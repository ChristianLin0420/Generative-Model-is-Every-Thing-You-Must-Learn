#!/bin/bash
# Sweep checkpoints → quality curves + report

set -e

echo "=== DDPM Checkpoint Comparison ==="

# Change to project directory  
cd "$(dirname "$0")/.."

# Parse arguments
DATASET=${1:-mnist}

echo "Dataset: $DATASET"
echo "Comparing all checkpoints from config..."

# Run checkpoint comparison
echo "1. Generating sample grids for comparison..."
python -m src.cli compare.ckpts \
    --config "configs/${DATASET}.yaml"

# Run quality evaluation
echo "2. Evaluating sample quality..."
python -m src.cli eval.quality \
    --config "configs/${DATASET}.yaml" \
    --num-samples 200

echo "Checkpoint comparison completed successfully!"
echo ""
echo "Results available in:"
echo "  - outputs/grids/checkpoint_comparison.png"
echo "  - outputs/curves/quality_vs_checkpoint.png"  
echo "  - outputs/logs/quality_metrics.csv"
echo "  - outputs/reports/sampling_checkpoints.md"
