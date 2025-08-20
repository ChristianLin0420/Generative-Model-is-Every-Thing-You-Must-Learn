# Day 15: DDPM with EMA Weights

## 🎯 Objective
Implement Exponential Moving Average (EMA) of model weights to improve training stability and sample quality in diffusion models.

## 📋 Tasks

### EMA Implementation
- **Implement exponential moving average (EMA) of weights**
  - Track EMA weights alongside training weights
  - Update EMA weights after each optimization step
  - Use EMA weights for sampling/evaluation

### Comparative Analysis
- **Compare training stability and sample quality**
  - Compare training curves with/without EMA
  - Evaluate sample quality using EMA vs regular weights
  - Analyze convergence behavior and final performance

## 🧮 EMA Formulation

### EMA Update Rule
```
θ_ema = β * θ_ema + (1 - β) * θ

where:
- θ: current training weights
- θ_ema: exponential moving average weights
- β: EMA decay rate (typically 0.995 - 0.9999)
```

### Decay Schedule
- **Fixed**: β = constant (e.g., 0.999)
- **Scheduled**: β starts lower and increases over training
- **Step-based**: β = 1 - 1/(step + 1) for automatic scheduling

## 🔧 Implementation Tips
- Initialize EMA weights as copy of initial training weights
- Update EMA weights after each optimizer step, not each batch
- Use higher β values (0.999-0.9999) for smoother averages
- Store EMA weights separately from training weights
- Use EMA weights for all sampling and evaluation

## 📊 Expected Outputs
- Training loss curves comparing EMA vs non-EMA models
- Sample quality comparison (FID scores, visual inspection)
- Analysis of different β values (0.99, 0.999, 0.9999)
- EMA weight trajectory analysis
- Training stability metrics (loss variance, convergence speed)

## 🎓 Learning Outcomes
- Understanding training stabilization techniques
- Experience with momentum-based weight averaging
- Best practices for diffusion model training

## 📖 Resources
- EMA in deep learning training
- Training stability analysis in diffusion models
- Model averaging techniques

---
**Time Estimate**: 2-3 hours  
**Difficulty**: ⭐⭐⭐☆☆

## 🎉 Milestone: Core Diffusion Complete!
You've now mastered advanced diffusion techniques and are ready for cutting-edge research topics!