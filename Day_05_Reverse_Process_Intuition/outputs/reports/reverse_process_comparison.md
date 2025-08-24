# Forward vs Reverse Process Comparison

**Dataset:** mnist
**Generated:** 2025-08-23 09:32:50

## Summary

This report compares the forward diffusion process (noise addition) with the learned reverse process (denoising) to evaluate how well the model has learned to invert the noise addition process.

### Key Metrics

- **Average PSNR:** 8.06 dB
- **Average SSIM:** 0.2900
- **Minimum PSNR:** 6.59 dB
- **Maximum SSIM:** 1.0000

## Reconstruction Quality by Timestep

| Timestep | PSNR (dB) | SSIM |
|----------|-----------|------|
|   0 | ∞ | 1.0000 |
|  14 | 11.95 | 0.3818 |
|  28 | 9.23 | 0.2676 |
|  42 | 8.01 | 0.2056 |
|  56 | 7.03 | 0.1436 |
|  70 | 6.96 | 0.1282 |
|  84 | 6.65 | 0.1071 |
|  99 | 6.59 | 0.0860 |

## Schedule Sensitivity Analysis

Comparison of different noise schedules:

| Schedule | PSNR (dB) | SSIM |
|----------|-----------|------|
| linear | 7.60 | 0.3381 |
| cosine | 5.57 | 0.1983 |
| quadratic | 8.77 | 0.3823 |

## Observations

- **Best PSNR Schedule:** quadratic (8.77 dB)
- **Best SSIM Schedule:** quadratic (0.3823)
- Reconstruction quality is **poor** (PSNR < 15 dB)
- Structural similarity is **needs improvement** (SSIM < 0.6)

## Figures

1. **trajectory_comparison.png**: Side-by-side comparison of forward and reverse trajectories
2. **schedule_sensitivity.png**: Performance comparison across different noise schedules
3. **forward_vs_reverse_panel.png**: Detailed forward vs reverse process visualization

## Technical Details

- **Model Type:** UNetTiny
- **Scheduler Type:** DDPMScheduler
- **Number of Timesteps:** 100
- **Beta Schedule:** Linear (β_start=0.0001, β_end=0.02)
- **Test Images:** 4 samples
