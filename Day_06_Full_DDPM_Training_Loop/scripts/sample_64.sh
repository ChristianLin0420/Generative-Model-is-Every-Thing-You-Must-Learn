#!/bin/bash

# Generate 64 sample images from trained DDPM model
# Usage: bash scripts/sample_64.sh [CONFIG] [OPTIONS]

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

NUM_SAMPLES=64
METHOD=""
NUM_STEPS=""
CHECKPOINT=""
OUTPUT_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --num_samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
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
        -h|--help)
            echo "Usage: $0 [CONFIG] [OPTIONS]"
            echo ""
            echo "Arguments:"
            echo "  CONFIG              Config file path or name (mnist, cifar10)"
            echo ""
            echo "Options:"
            echo "  --num_samples N     Number of samples to generate (default: 64)"
            echo "  --method METHOD     Sampling method (ddpm, ddim)"
            echo "  --num_steps N       Number of sampling steps"
            echo "  --checkpoint PATH   Checkpoint file path"
            echo "  --output_name NAME  Output filename"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                            # Use MNIST config, 64 samples"
            echo "  $0 cifar10                    # Use CIFAR-10 config"
            echo "  $0 mnist --method ddpm        # Use DDPM sampling"
            echo "  $0 --num_samples 100          # Generate 100 samples"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "🎨 Generating $NUM_SAMPLES samples"
echo "Config: $CONFIG_FILE"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if any checkpoint exists
CHECKPOINT_DIR="outputs/ckpts"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    echo "Please train a model first:"
    if [[ "$CONFIG_FILE" == *"mnist"* ]]; then
        echo "  bash scripts/train_mnist.sh"
    else
        echo "  bash scripts/train_cifar10.sh"
    fi
    exit 1
fi

# Build command
CMD="python -m src.cli sample.grid --config $CONFIG_FILE --num_samples $NUM_SAMPLES"

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

# Run sampling
$CMD

echo ""
echo "✅ Sample generation completed!"
echo "Check outputs in: outputs/grids/"

# List generated files
echo ""
echo "Generated files:"
ls -la outputs/grids/ | tail -5