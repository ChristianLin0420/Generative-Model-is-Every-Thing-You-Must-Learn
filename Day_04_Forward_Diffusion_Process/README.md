# Day 4: Forward Diffusion Process

## 🎯 Objective
Implement the mathematical foundation of diffusion models by coding the forward noising process and understanding the diffusion trajectory.

## 📋 Tasks

### Core Implementation
- **Code the forward diffusion process q(x_t|x_{t-1})**
  - Implement step-by-step Markov chain noising
  - Use the closed-form solution q(x_t|x_0)
  - Define noise schedule β_t (start with linear)

### Analysis & Visualization
- **Visualize diffusion trajectory**
  - Show images at different timesteps t
  - Plot noise level progression
  - Analyze convergence to pure noise

## 🧮 Mathematical Foundation

### Forward Process
- **Step-by-step**: q(x_t|x_{t-1}) = N(x_t; √(1-β_t)x_{t-1}, β_t I)
- **Closed-form**: q(x_t|x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
- **Where**: ᾱ_t = ∏(1-β_s) for s from 1 to t

### Implementation Details
- **Noise Schedule**: β_1, β_2, ..., β_T (e.g., linear from 0.0001 to 0.02)
- **Sampling**: x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε where ε ~ N(0,I)

## 🔧 Implementation Tips
- Precompute α_t, ᾱ_t, √(ᾱ_t), √(1-ᾱ_t) for efficiency
- Use proper numerical stability (avoid sqrt of small numbers)
- Implement both iterative and direct sampling methods
- Visualize intermediate results frequently

## 📊 Expected Outputs
- Forward diffusion visualizations (t=0,100,200,...,1000)
- Noise schedule plots (β_t, α_t, ᾱ_t vs t)
- Signal-to-noise ratio analysis over timesteps
- Comparison of iterative vs closed-form sampling

## 🎓 Learning Outcomes
- Deep understanding of diffusion process mechanics
- Mathematical foundation for DDPM
- Preparation for reverse process implementation

## 📖 Resources
- DDPM paper (Ho et al., 2020) - Section 2
- Forward process mathematical derivation
- Gaussian process and Markov chain concepts

---
**Time Estimate**: 3-4 hours  
**Difficulty**: ⭐⭐⭐☆☆