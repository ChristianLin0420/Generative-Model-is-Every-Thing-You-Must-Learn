#!/bin/bash

# Create reverse trajectory animation from trained DDPM model
# Usage: bash scripts/animate_traj.sh [CONFIG] [OPTIONS]

set -e  # Exit on error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default config (MNIST)
CONFIG_FILE="$PROJECT_ROOT/configs/mnist.yaml"

# Change to project root
cd "$PROJECT_ROOT"

# Parse arguments
if [ $# -ge 1 ] && [ -f "$1" ]; then
    CONFIG_FILE="$1"
    shift
elif [ $# -ge 1 ] && [ -f "configs/$1.yaml" ]; then
    CONFIG_FILE="$PROJECT_ROOT/configs/$1.yaml"
    shift
fi

METHOD=""
NUM_STEPS=""
CHECKPOINT=""
OUTPUT_NAME=""
FPS=10

while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            METHOD="--method $2"
            shift 2
            ;;
        --num_steps)
            NUM_STEPS="--num_steps $2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="--checkpoint $2"
            shift 2
            ;;
        --output_name)
            OUTPUT_NAME="--output_name $2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [CONFIG] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  CONFIG              Config file path or name (mnist, cifar10)"
            echo ""
            echo "Options:"
            echo "  --method METHOD     Sampling method (ddpm, ddim) [default: from config]"
            echo "  --num_steps N       Number of sampling steps [default: from config]"
            echo "  --checkpoint PATH   Checkpoint file path [default: latest EMA]"
            echo "  --output_name NAME  Output filename [default: reverse_trajectory.gif]"
            echo "  --fps FPS           Animation FPS (default: 10)"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                         # Create MNIST trajectory"
            echo "  $0 cifar10                 # Create CIFAR-10 trajectory"
            echo "  $0 --method ddpm           # Use DDPM (slow but full process)"
            echo "  $0 --method ddim --num_steps 50  # DDIM with 50 steps"
            echo "  $0 --fps 20                # Faster animation"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🎬 Creating reverse trajectory animation"
echo "Config: $CONFIG_FILE"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if checkpoint exists
CHECKPOINT_DIR="outputs/ckpts"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "Please train a model first."
    exit 1
fi

# Build command
CMD="python -m src.cli sample.traj --config $CONFIG_FILE --fps $FPS"

if [ ! -z "$METHOD" ]; then
    CMD="$CMD $METHOD"
fi

if [ ! -z "$NUM_STEPS" ]; then
    CMD="$CMD $NUM_STEPS"
fi

if [ ! -z "$CHECKPOINT" ]; then
    CMD="$CMD $CHECKPOINT"
fi

if [ ! -z "$OUTPUT_NAME" ]; then
    CMD="$CMD $OUTPUT_NAME"
fi

echo "Running command: $CMD"
echo ""

# Estimate time based on method
if [[ "$METHOD" == *"ddpm"* ]]; then
    echo "⏱️  DDPM trajectory generation may take several minutes (1000 steps)..."
else
    echo "⏱️  DDIM trajectory generation should complete in ~30-60 seconds..."
fi
echo ""

# Run trajectory generation
$CMD

echo ""
echo "✅ Trajectory animation completed!"
echo "Check outputs in: outputs/animations/"

# List generated files
echo ""
echo "Generated files:"
ls -la outputs/animations/ | tail -3

echo ""
echo "💡 Tips:"
echo "  - View with any image viewer or web browser"
echo "  - DDPM shows full 1000-step process (slower generation, more steps)"
echo "  - DDIM shows compressed process (faster generation, fewer steps)"
echo "  - Try different FPS values for different animation speeds"