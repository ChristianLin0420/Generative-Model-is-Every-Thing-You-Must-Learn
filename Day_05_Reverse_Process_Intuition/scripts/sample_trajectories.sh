#!/bin/bash

# Generate sampling trajectories and animations
# Shows the reverse diffusion process step by step

set -e  # Exit on error

echo "Generating DDPM sampling trajectories..."
echo "========================================"

# Change to project directory
cd "$(dirname "$0")/.."

# Default checkpoint path
CHECKPOINT="${1:-outputs/ckpts/model_latest.pth}"
NUM_SAMPLES="${2:-4}"
SAMPLER="${3:-ddpm}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Usage: $0 [checkpoint_path] [num_samples] [sampler]"
    echo "Example: $0 outputs/ckpts/model_best.pth 6 ddim"
    exit 1
fi

echo "Using checkpoint: $CHECKPOINT"
echo "Number of samples: $NUM_SAMPLES"
echo "Sampler: $SAMPLER"
echo ""

# Create output directory
mkdir -p outputs/grids
mkdir -p outputs/animations

# Generate trajectory grid
echo "Generating trajectory grid..."
python -m src.cli sample.traj \
    --ckpt "$CHECKPOINT" \
    --num-samples "$NUM_SAMPLES" \
    --sampler "$SAMPLER" \
    --output "outputs/grids/reverse_traj_${SAMPLER}_${NUM_SAMPLES}samples.png" \
    --animation

echo "Done!"
echo ""
echo "Generated files:"
echo "  - Trajectory grid: outputs/grids/reverse_traj_${SAMPLER}_${NUM_SAMPLES}samples.png"
echo "  - Animation: outputs/grids/reverse_traj_${SAMPLER}_${NUM_SAMPLES}samples.gif"

# Generate additional animations with different samplers if requested
if [ "$SAMPLER" = "ddpm" ]; then
    echo ""
    echo "Also generating DDIM animation for comparison..."
    
    python -m src.cli sample.traj \
        --ckpt "$CHECKPOINT" \
        --num-samples "$NUM_SAMPLES" \
        --sampler ddim \
        --output "outputs/grids/reverse_traj_ddim_${NUM_SAMPLES}samples.png" \
        --animation
    
    echo "DDIM animation generated: outputs/grids/reverse_traj_ddim_${NUM_SAMPLES}samples.gif"
fi

echo ""
echo "Trajectory generation completed!"