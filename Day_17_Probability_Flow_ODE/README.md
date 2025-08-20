# Day 17: Probability Flow ODE

## 🎯 Objective
Implement sampling via ODE solvers using the probability flow formulation of diffusion models for deterministic and faster generation.

## 📋 Tasks

### ODE Formulation
- **Implement sampling via ODE solver**
  - Convert DDPM to continuous-time ODE
  - Implement neural ODE solver (Euler, Runge-Kutta)
  - Handle score function parameterization

### Sampling Comparison
- **Compare ODE vs ancestral sampling**
  - Sample quality with different ODE solvers
  - Computational efficiency analysis
  - Deterministic vs stochastic sampling trade-offs

## 🧮 Probability Flow ODE

### ODE Formulation
```
dx = [f(x,t) - ½g²(t)∇_x log p_t(x)] dt
```

### Score-Based Parameterization
```
dx = [f(x,t) - ½g²(t)s_θ(x,t)] dt
where s_θ(x,t) ≈ ∇_x log p_t(x)
```

### DDPM-specific Form
```
dx = -½β(t)[x + 2s_θ(x,t)] dt
```

## 🔧 Implementation Tips
- Convert ε-prediction to score: s_θ = -ε_θ/√(1-ᾱ_t)
- Use adaptive step size ODE solvers (dopri5, euler)
- Implement both fixed and adaptive step size methods
- Handle numerical stability near t=0
- Consider NFE (Number of Function Evaluations) for efficiency

## 📊 Expected Outputs
- ODE sampling trajectories visualization
- Comparison of different ODE solvers (Euler, RK4, dopri5)
- NFE vs sample quality trade-offs
- Deterministic sampling consistency tests
- Computational time analysis

## 🎓 Learning Outcomes
- Understanding continuous formulations of diffusion
- Experience with neural ODEs
- Trade-offs in deterministic vs stochastic sampling

## 📖 Resources
- Score-Based Generative Models (Song et al., 2021)
- Neural ODE implementations
- ODE solver theory and practice

---
**Time Estimate**: 4-5 hours  
**Difficulty**: ⭐⭐⭐⭐☆