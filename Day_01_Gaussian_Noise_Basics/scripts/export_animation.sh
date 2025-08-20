#!/bin/bash

# Export noise progression animation in different formats
# Usage: ./scripts/export_animation.sh [gif|mp4|both]

set -e

FORMAT=${1:-gif}  # Default to gif
CONFIG=${2:-configs/default.yaml}
NUM_IMAGES=${3:-16}

echo "🎬 Exporting noise progression animation..."
echo "Format: $FORMAT"
echo "Config: $CONFIG" 
echo "Images: $NUM_IMAGES"

# Create animations directory
mkdir -p outputs/animations

case $FORMAT in
    gif)
        echo "Creating GIF animation..."
        python -m src.cli animate \
            --config $CONFIG \
            --num-images $NUM_IMAGES \
            --out outputs/animations/mnist_noise_progression.gif
        echo "✅ GIF saved to outputs/animations/mnist_noise_progression.gif"
        ;;
    
    mp4)
        echo "Creating MP4 animation..."
        python -m src.cli animate \
            --config $CONFIG \
            --num-images $NUM_IMAGES \
            --out outputs/animations/mnist_noise_progression.mp4
        echo "✅ MP4 saved to outputs/animations/mnist_noise_progression.mp4"
        ;;
        
    both)
        echo "Creating both GIF and MP4 animations..."
        
        # GIF
        python -m src.cli animate \
            --config $CONFIG \
            --num-images $NUM_IMAGES \
            --out outputs/animations/mnist_noise_progression.gif
        
        # MP4
        python -m src.cli animate \
            --config $CONFIG \
            --num-images $NUM_IMAGES \
            --out outputs/animations/mnist_noise_progression.mp4
            
        echo "✅ Both formats saved to outputs/animations/"
        ;;
        
    *)
        echo "❌ Unknown format: $FORMAT"
        echo "Usage: $0 [gif|mp4|both] [config_path] [num_images]"
        exit 1
        ;;
esac

# List generated files
echo ""
echo "📁 Generated animations:"
ls -la outputs/animations/ | grep -E '\.(gif|mp4)$'

echo ""
echo "🎉 Animation export completed!"