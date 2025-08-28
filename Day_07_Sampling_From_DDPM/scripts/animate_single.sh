#!/bin/bash
# Generate animation of single sample reverse trajectory (T→0)

set -e

echo "=== DDPM Trajectory Animation ==="

# Change to project directory
cd "$(dirname "$0")/.."

# Parse arguments
DATASET=${1:-mnist}
CHECKPOINT=${2:-ema.pt}
SAMPLE_IDX=${3:-0}

echo "Dataset: $DATASET"
echo "Checkpoint: $CHECKPOINT"
echo "Sample index: $SAMPLE_IDX"

# Run trajectory animation
python -m src.cli sample.traj \
    --config "configs/${DATASET}.yaml" \
    --ckpt "$CHECKPOINT" \
    --idx "$SAMPLE_IDX" \
    --record-every 20 \
    --fps 8

echo "Animation generated successfully!"
echo "Check outputs/animations/ for results."
