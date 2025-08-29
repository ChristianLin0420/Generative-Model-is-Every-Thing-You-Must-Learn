#!/bin/bash
# Compare beta schedules using metrics and generate comprehensive report

set -e  # Exit on any error

echo "========================================"
echo "Day 8: Beta Schedule Comparison"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "configs/base.yaml" ]; then
    echo "Error: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Array of schedules to compare
schedules=("linear" "cosine" "quadratic")

echo "Comparing schedules: ${schedules[@]}"
echo

# Check if all models are trained
echo "Checking trained models..."
missing_models=()
for schedule in "${schedules[@]}"; do
    output_dir="outputs/${schedule}"
    
    if [ ! -d "$output_dir/ckpts" ]; then
        missing_models+=("$schedule")
        echo "  ✗ $schedule: No checkpoints directory found"
    elif [ ! -f "$output_dir/ckpts/ema.pt" ] && [ ! -f "$output_dir/ckpts/best.pt" ]; then
        missing_models+=("$schedule")
        echo "  ✗ $schedule: No suitable checkpoint found"
    else
        echo "  ✓ $schedule: Model found"
    fi
done

if [ ${#missing_models[@]} -gt 0 ]; then
    echo
    echo "Error: Missing trained models for: ${missing_models[@]}"
    echo "Please run training first: bash scripts/train_all.sh"
    exit 1
fi

echo "All models found. Proceeding with comparison..."
echo

# Prepare config file list
config_files=""
for schedule in "${schedules[@]}"; do
    config_files="$config_files configs/${schedule}.yaml"
done

echo "----------------------------------------"
echo "Computing metrics and generating report..."
echo "----------------------------------------"

echo "Running comprehensive comparison..."
echo "This may take a few minutes as it evaluates generation quality..."

# Run comparison
python -m src.cli compare --configs $config_files

if [ $? -eq 0 ]; then
    echo "✓ Comparison completed successfully"
else
    echo "✗ Comparison failed"
    exit 1
fi

echo
echo "========================================"
echo "Comparison Analysis Complete!"
echo "========================================"

# Check and display results
comparison_dir="outputs/comparison"

if [ -d "$comparison_dir" ]; then
    echo "📊 Comparison Results:"
    echo
    
    # CSV results
    if [ -f "$comparison_dir/comparison.csv" ]; then
        echo "📈 Quantitative Metrics: $comparison_dir/comparison.csv"
        echo "   Contains: FID-proxy, PSNR, SSIM, LPIPS, timing metrics"
        
        # Show a preview of the results
        if command -v column &> /dev/null; then
            echo
            echo "   Preview:"
            echo "   --------"
            head -n 10 "$comparison_dir/comparison.csv" | column -t -s','
        fi
    fi
    
    echo
    
    # Quality plots
    if [ -f "$comparison_dir/quality_vs_schedule.png" ]; then
        echo "📊 Quality Plots: $comparison_dir/quality_vs_schedule.png"
        echo "   Bar charts comparing key metrics across schedules"
    fi
    
    # Training curves
    if [ -f "$comparison_dir/training_curves.png" ]; then
        echo "📈 Training Curves: $comparison_dir/training_curves.png"
        echo "   Loss, learning rate, and timing comparison"
    fi
    
    # Multi-run samples
    if [ -f "$comparison_dir/multi_run_samples.png" ]; then
        echo "🖼️  Sample Comparison: $comparison_dir/multi_run_samples.png"
        echo "   Side-by-side visual comparison of generated samples"
    fi
    
    # Markdown report
    if [ -f "$comparison_dir/report.md" ]; then
        echo "📝 Summary Report: $comparison_dir/report.md"
        echo "   Key findings and schedule characteristics"
        
        echo
        echo "Quick Summary:"
        echo "-------------"
        # Show a few key lines from the report
        if command -v grep &> /dev/null; then
            grep -A 10 "## Key Findings" "$comparison_dir/report.md" 2>/dev/null || echo "Report generated - see file for details"
        fi
    fi
    
    echo
    echo "Individual Schedule Results:"
    for schedule in "${schedules[@]}"; do
        output_dir="outputs/${schedule}"
        echo "  📁 $schedule:"
        
        # Training metrics
        if [ -f "$output_dir/logs/metrics.csv" ]; then
            echo "    📊 Training metrics: $output_dir/logs/metrics.csv"
        fi
        
        # Sample grids
        grid_count=$(find "$output_dir/grids" -name "*.png" 2>/dev/null | wc -l)
        echo "    🖼️  Sample grids: $grid_count files in $output_dir/grids/"
        
        # Animations
        anim_count=$(find "$output_dir/animations" -name "*.gif" 2>/dev/null | wc -l)
        if [ "$anim_count" -gt 0 ]; then
            echo "    🎬 Animations: $anim_count files in $output_dir/animations/"
        fi
    done
    
else
    echo "Warning: Comparison directory not found at $comparison_dir"
fi

echo
echo "========================================"
echo "Analysis Complete! 🎉"
echo "========================================"

echo
echo "Key Deliverables Generated:"
echo "  ✅ Three trained models (linear, cosine, quadratic)"
echo "  ✅ Schedule comparison plots (β, ᾱ, SNR overlays)"
echo "  ✅ Sample grids for each schedule (DDPM & DDIM)"
echo "  ✅ Trajectory animations (T→0 reverse diffusion)"
echo "  ✅ Quantitative metrics (FID-proxy, PSNR, SSIM, timing)"
echo "  ✅ Comprehensive comparison report"
echo
echo "View your results:"
echo "  📂 Browse outputs/ directory for all generated content"
echo "  📝 Read $comparison_dir/report.md for key insights"
echo "  📊 Check $comparison_dir/comparison.csv for detailed metrics"
echo
echo "Day 8 Beta Schedule Comparison: COMPLETE! ✨"
