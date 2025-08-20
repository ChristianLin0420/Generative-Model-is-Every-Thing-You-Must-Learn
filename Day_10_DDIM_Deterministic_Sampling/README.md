# Day 10: DDIM (Deterministic Sampling)

## 🎯 Objective
Implement DDIM (Denoising Diffusion Implicit Models) to achieve faster, deterministic sampling with fewer steps.

## 📋 Tasks

### DDIM Implementation
- **Implement DDIM sampling**
  - Use deterministic sampling trajectory
  - Support arbitrary sampling timesteps (not just consecutive)
  - Enable both deterministic and stochastic variants

### Performance Comparison
- **Compare speed and quality with DDPM**
  - Benchmark sampling time for different step counts
  - Evaluate sample quality vs number of sampling steps
  - Analyze deterministic vs stochastic sampling trade-offs

## 🧮 DDIM Formulation

### DDIM Sampling Update
```
x_{t-1} = √(α_{t-1}) * pred_x0 + √(1-α_{t-1}-σ_t²) * ε_θ(x_t,t) + σ_t * ε_t

where:
pred_x0 = (x_t - √(1-α_t) * ε_θ(x_t,t)) / √(α_t)
```

### Sampling Schedule
- **Deterministic**: σ_t = 0 (fully deterministic)
- **Stochastic**: σ_t = η * √((1-α_{t-1})/(1-α_t)) * √(1-α_t/α_{t-1})
- **η parameter**: Controls stochasticity (η=0: deterministic, η=1: DDPM-like)

## 🔧 Implementation Tips
- Precompute sampling timesteps (e.g., [0, 50, 100, 150, ..., 1000])
- Support arbitrary timestep schedules (uniform, quadratic, etc.)
- Implement both x0-prediction and ε-prediction parameterizations
- Handle numerical stability around t=0

## 📊 Expected Outputs
- Speed comparison: DDIM (10, 20, 50 steps) vs DDPM (1000 steps)
- Quality analysis across different step counts
- Deterministic vs stochastic sampling comparison (η=0 vs η=1)
- Sampling trajectory visualizations
- Memory usage and computational efficiency analysis

## 🎓 Learning Outcomes
- Understanding of non-Markovian sampling processes
- Trade-offs between sampling speed and quality
- Foundation for fast sampling techniques

## 📖 Resources
- DDIM paper (Song et al., 2021)
- Analysis of sampling trajectory choices
- Speed-quality trade-offs in diffusion models

---
**Time Estimate**: 3-4 hours  
**Difficulty**: ⭐⭐⭐⭐☆