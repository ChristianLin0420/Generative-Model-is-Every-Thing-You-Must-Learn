#!/bin/bash
# Generate samples and animations for all trained beta schedules

set -e  # Exit on any error

echo "========================================"
echo "Day 8: Sampling All Beta Schedules"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "configs/base.yaml" ]; then
    echo "Error: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Array of schedules to sample from
schedules=("linear" "cosine" "quadratic")

echo "Generating samples for schedules: ${schedules[@]}"
echo

# First, plot schedule comparison
echo "----------------------------------------"
echo "Plotting schedule comparison..."
echo "----------------------------------------"

config_files=""
for schedule in "${schedules[@]}"; do
    config_files="$config_files configs/${schedule}.yaml"
done

echo "Creating schedule overlay plot..."
python -m src.cli plot.schedules --configs $config_files

if [ $? -eq 0 ]; then
    echo "✓ Schedule plots saved to outputs/plots/schedules_overlay.png"
else
    echo "✗ Failed to create schedule plots"
    exit 1
fi
echo

# Generate samples for each schedule
for schedule in "${schedules[@]}"; do
    echo "----------------------------------------"
    echo "Sampling from $schedule schedule..."
    echo "----------------------------------------"
    
    config_file="configs/${schedule}.yaml"
    output_dir="outputs/${schedule}"
    
    # Check if model was trained
    if [ ! -d "$output_dir/ckpts" ]; then
        echo "Warning: No checkpoints found for $schedule schedule in $output_dir/ckpts"
        echo "Please run training first: bash scripts/train_all.sh"
        continue
    fi
    
    # Check for EMA or best checkpoint
    if [ ! -f "$output_dir/ckpts/ema.pt" ] && [ ! -f "$output_dir/ckpts/best.pt" ]; then
        echo "Warning: No suitable checkpoint found for $schedule schedule"
        continue
    fi
    
    echo "Generating DDPM sample grid for $schedule..."
    python -m src.cli sample.grid --config "$config_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ DDPM samples generated"
    else
        echo "✗ Failed to generate DDPM samples for $schedule"
    fi
    
    echo "Generating DDIM sample grid for $schedule..."
    python -m src.cli sample.grid --config "$config_file" --ddim
    
    if [ $? -eq 0 ]; then
        echo "✓ DDIM samples generated"
    else
        echo "✗ Failed to generate DDIM samples for $schedule"
    fi
    
    echo "Generating trajectory visualization for $schedule..."
    python -m src.cli sample.traj --config "$config_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Trajectory visualization generated"
    else
        echo "✗ Failed to generate trajectory for $schedule"
    fi
    
    echo "✓ Completed sampling for $schedule schedule"
    echo
done

echo "========================================"
echo "Sampling completed for all schedules!"
echo "========================================"

# Print summary
echo "Sampling Summary:"
echo "  📊 Schedule comparison: outputs/plots/schedules_overlay.png"
echo

for schedule in "${schedules[@]}"; do
    output_dir="outputs/${schedule}"
    if [ -d "$output_dir" ]; then
        echo "  📁 $schedule schedule:"
        
        # Check for grids
        if [ -f "$output_dir/grids/samples_ddpm.png" ]; then
            echo "    ✓ DDPM samples: $output_dir/grids/samples_ddpm.png"
        fi
        
        if [ -f "$output_dir/grids/samples_ddim.png" ]; then
            echo "    ✓ DDIM samples: $output_dir/grids/samples_ddim.png"
        fi
        
        if [ -f "$output_dir/grids/trajectory_grid.png" ]; then
            echo "    ✓ Trajectory grid: $output_dir/grids/trajectory_grid.png"
        fi
        
        # Check for animations
        if [ -f "$output_dir/animations/reverse_traj.gif" ]; then
            echo "    ✓ Animation: $output_dir/animations/reverse_traj.gif"
        fi
        
        # Count all generated files
        grid_count=$(find "$output_dir/grids" -name "*.png" 2>/dev/null | wc -l)
        anim_count=$(find "$output_dir/animations" -name "*.gif" 2>/dev/null | wc -l)
        echo "    📈 Total: $grid_count grids, $anim_count animations"
    else
        echo "  ✗ $schedule: output directory not found"
    fi
done

echo
echo "Generated outputs:"
echo "  🔬 Sample grids (DDPM & DDIM) for quality comparison"
echo "  🎬 Trajectory animations showing reverse diffusion T→0"
echo "  📊 Schedule comparison plots (β, ᾱ, SNR overlays)"
echo
echo "Next steps:"
echo "  1. Compare quality metrics: bash scripts/compare.sh"
echo "  2. Or use Makefile: make compare"
echo "  3. View results in outputs/ directories"
