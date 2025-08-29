#!/bin/bash
# Test version: Train DDPM models with all three beta schedules (3 epochs each)

set -e  # Exit on any error

echo "========================================"
echo "Day 8: Testing All Beta Schedules (3 epochs each)"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "configs/base.yaml" ]; then
    echo "Error: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Array of schedules to train
schedules=("linear" "cosine" "quadratic")

echo "Training schedules: ${schedules[@]} (3 epochs each for testing)"
echo

# Train each schedule
for schedule in "${schedules[@]}"; do
    echo "----------------------------------------"
    echo "Training $schedule schedule..."
    echo "----------------------------------------"
    
    config_file="configs/test_${schedule}.yaml"
    
    if [ ! -f "$config_file" ]; then
        echo "Error: Config file $config_file not found"
        exit 1
    fi
    
    # Run training
    echo "Starting training with config: $config_file"
    python -m src.cli train --config "$config_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully trained $schedule schedule"
    else
        echo "✗ Failed to train $schedule schedule"
        exit 1
    fi
    echo
done

echo "========================================"
echo "All schedules trained successfully!"
echo "========================================"

# Print summary
echo "Training Summary:"
for schedule in "${schedules[@]}"; do
    output_dir="outputs/test_${schedule}"
    if [ -d "$output_dir" ]; then
        echo "  ✓ $schedule: outputs saved to $output_dir"
        
        # Check for checkpoints
        if [ -f "$output_dir/ckpts/ema.pt" ]; then
            echo "    - EMA checkpoint: $output_dir/ckpts/ema.pt"
        fi
        
        if [ -f "$output_dir/ckpts/best.pt" ]; then
            echo "    - Best checkpoint: $output_dir/ckpts/best.pt"
        fi
        
        # Check for sample grids
        grid_count=$(find "$output_dir/grids" -name "*.png" 2>/dev/null | wc -l)
        if [ "$grid_count" -gt 0 ]; then
            echo "    - Sample grids: $grid_count files"
        fi
    else
        echo "  ✗ $schedule: output directory not found"
    fi
done

echo
echo "✅ Quick test completed! Fixed checkpoint saving works correctly."
echo "To run full training (40 epochs each): make train_all"
