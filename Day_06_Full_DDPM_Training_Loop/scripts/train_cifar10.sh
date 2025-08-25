#!/bin/bash

# Train DDPM on CIFAR-10 dataset
# Usage: bash scripts/train_cifar10.sh [OPTIONS]

set -e  # Exit on error

# Script configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_ROOT/configs/cifar10.yaml"

echo "🚀 Starting DDPM training on CIFAR-10"
echo "Project root: $PROJECT_ROOT"
echo "Config file: $CONFIG_FILE"

# Change to project root
cd "$PROJECT_ROOT"

# Parse command line arguments
EPOCHS=""
BATCH_SIZE=""
LR=""
RESUME=""
DEVICE=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="--epochs $2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="--batch_size $2"
            shift 2
            ;;
        --learning_rate)
            LR="--learning_rate $2"
            shift 2
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --device)
            DEVICE="--device $2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="--output_dir $2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs N          Number of training epochs (default: from config)"
            echo "  --batch_size N      Batch size (default: from config)"
            echo "  --learning_rate F   Learning rate (default: from config)"
            echo "  --resume PATH       Resume from checkpoint"
            echo "  --device DEVICE     Device to use (cpu, cuda, mps, auto)"
            echo "  --output_dir DIR    Output directory"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "⚠️  CIFAR-10 training requires significant computational resources:"
echo "   - Recommended: GPU with ≥8GB VRAM"
echo "   - Training time: 12-24 hours on modern GPU"
echo "   - Disk space: ~2GB for checkpoints and outputs"
echo ""

# Ask for confirmation
read -p "Continue with CIFAR-10 training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Create command
CMD="python -m src.cli train --config $CONFIG_FILE"

if [ ! -z "$EPOCHS" ]; then
    CMD="$CMD $EPOCHS"
fi

if [ ! -z "$BATCH_SIZE" ]; then
    CMD="$CMD $BATCH_SIZE"
fi

if [ ! -z "$LR" ]; then
    CMD="$CMD $LR"
fi

if [ ! -z "$RESUME" ]; then
    CMD="$CMD $RESUME"
fi

if [ ! -z "$DEVICE" ]; then
    CMD="$CMD $DEVICE"
fi

if [ ! -z "$OUTPUT_DIR" ]; then
    CMD="$CMD $OUTPUT_DIR"
fi

echo "Running command: $CMD"
echo ""

# Run training with nice priority to avoid system overload
nice -n 10 $CMD

echo ""
echo "✅ CIFAR-10 training completed!"
echo "Check outputs in: $(pwd)/outputs/"
echo ""
echo "Next steps:"
echo "  1. Generate samples: bash scripts/sample_64.sh"
echo "  2. Create trajectory: bash scripts/animate_traj.sh"
echo "  3. Run evaluation: python -m src.cli eval --config $CONFIG_FILE --all_methods"
echo "  4. Plot curves: python -m src.cli viz.curves --log_dir outputs/logs"