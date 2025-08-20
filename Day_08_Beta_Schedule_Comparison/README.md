# Day 8: β-Schedule Comparison

## 🎯 Objective
Compare different noise schedules (β-schedules) and analyze their impact on diffusion model training and sample quality.

## 📋 Tasks

### Schedule Implementation
- **Implement linear, cosine, and quadratic noise schedules**
  - Linear: β_t increases linearly from β₁ to β_T
  - Cosine: Based on cosine function for smoother transitions
  - Quadratic: Quadratic progression for different noise characteristics

### Comparative Analysis
- **Compare sample quality across schedules**
  - Train models with each schedule (or use existing checkpoints)
  - Generate samples and evaluate quality metrics
  - Analyze training convergence and stability

## 🧮 Noise Schedule Formulations

### Linear Schedule
```
β_t = β_start + (β_end - β_start) * t/T
```
- **Typical values**: β_start = 0.0001, β_end = 0.02

### Cosine Schedule
```
α̅_t = cos²(π/2 * (t/T + s)/(1 + s))
β_t = 1 - α̅_t/α̅_{t-1}
```
- **Parameter**: s = 0.008 (small offset)

### Quadratic Schedule
```
β_t = β_start + (β_end - β_start) * (t/T)²
```

## 🔧 Implementation Tips
- Visualize each schedule before training
- Ensure β_t ∈ (0,1) and ∑β_t doesn't exceed reasonable bounds
- Precompute α_t, α̅_t for each schedule
- Use same model architecture for fair comparison
- Track both training metrics and sample quality

## 📊 Expected Outputs
- Visualization of β_t, α_t, α̅_t for each schedule
- Training loss comparisons across schedules
- Sample quality evaluation (FID scores if possible)
- Analysis of which schedule works best for your dataset
- Investigation of schedule impact on different timestep ranges

## 🎓 Learning Outcomes
- Understanding noise schedule importance
- Experience with hyperparameter sensitivity analysis
- Insight into training dynamics of diffusion models

## 📖 Resources
- Improved DDPM (Nichol & Dhariwal, 2021) - cosine schedule
- Original DDPM paper - linear schedule analysis
- Schedule sensitivity studies in diffusion literature

---
**Time Estimate**: 3-4 hours  
**Difficulty**: ⭐⭐⭐☆☆