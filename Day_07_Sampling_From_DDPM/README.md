# Day 7: Sampling from DDPM

## 🎯 Objective
Implement the complete DDPM sampling procedure and evaluate sample quality using trained models from Day 6.

## 📋 Tasks

### Ancestral Sampling Implementation
- **Implement ancestral sampling step by step**
  - Start from pure noise x_T ~ N(0,I)
  - Iteratively denoise using p_θ(x_{t-1}|x_t)
  - Handle the sampling variance properly

### Quality Assessment
- **Compare sample quality with training checkpoints**
  - Generate samples using different training epochs
  - Evaluate sample diversity and quality
  - Compare with training data distribution

## 🧮 DDPM Sampling Algorithm

### Ancestral Sampling
```
x_T ~ N(0, I)
for t = T, T-1, ..., 1:
    if t > 1:
        z ~ N(0, I)
    else:
        z = 0
    
    x_{t-1} = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t)) + σ_t * z
```

### Variance Schedule
- **σ_t**: Sampling variance (can be fixed or learned)
- **Options**: σ_t = β_t or σ_t = √((1-ᾱ_{t-1})/(1-ᾱ_t)) * β_t

## 🔧 Implementation Tips
- Precompute all sampling coefficients for efficiency
- Use proper numerical precision (avoid numerical instability)
- Implement progress tracking for long sampling chains
- Add option for deterministic sampling (set z=0)
- Consider CUDA memory management for batch sampling

## 📊 Expected Outputs
- Generated sample galleries from different checkpoints
- Sampling progression visualizations (every 100 steps)
- Quantitative metrics: FID, IS (if computing resources allow)
- Comparison of sampling variance schedules
- Analysis of sampling time vs quality trade-offs

## 🎓 Learning Outcomes
- Complete DDPM pipeline understanding
- Experience with generative model evaluation
- Preparation for advanced sampling methods (DDIM, etc.)

## 📖 Resources
- DDPM paper (Ho et al., 2020) - Algorithm 2
- Sampling variance schedule analysis
- Generative model evaluation metrics

---
**Time Estimate**: 4-5 hours  
**Difficulty**: ⭐⭐⭐⭐☆

## 🎉 Milestone: Foundations Complete!
Congratulations! You now have a complete working DDPM implementation and deep understanding of diffusion model fundamentals.