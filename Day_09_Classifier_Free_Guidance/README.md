# Day 9: Classifier-Free Guidance (CFG)

## 🎯 Objective
Implement classifier-free guidance to enable controllable conditional generation without requiring a separate classifier.

## 📋 Tasks

### Conditional Training Setup
- **Add conditional training (e.g., class-labels on CIFAR-10)**
  - Modify model to accept conditioning information
  - Implement unconditional training with dropout technique
  - Train joint conditional/unconditional model

### CFG Implementation
- **Implement CFG during sampling**
  - Modify sampling to use guidance scale
  - Balance conditional and unconditional predictions
  - Experiment with different guidance strengths

## 🧮 CFG Mathematical Foundation

### Training Objective
- Train ε_θ(x_t, t, c) where c is conditioning (class label)
- During training, randomly set c = ∅ (unconditional) with probability p_uncond
- Use same loss: L = ||ε - ε_θ(x_t, t, c)||²

### CFG Sampling
```
ε̃_θ(x_t, t, c) = ε_θ(x_t, t, ∅) + w * (ε_θ(x_t, t, c) - ε_θ(x_t, t, ∅))
```
- **w**: guidance scale (w=1 gives standard conditional generation)
- **w>1**: stronger conditioning, less diversity
- **w<1**: weaker conditioning, more diversity

## 🏗️ Architecture Modifications
- **Class Embedding**: Learnable embedding layer for class labels
- **Conditioning**: Add class embedding to time embedding
- **Unconditional Token**: Special token for unconditional generation (∅)
- **Dropout**: Randomly set conditioning to ∅ during training

## 🔧 Implementation Tips
- Use unconditional probability p_uncond = 0.1-0.2
- Start with guidance scale w = 1.0, then experiment with w ∈ [0.5, 10.0]
- Monitor both conditional and unconditional losses
- Use larger model if computational resources allow

## 📊 Expected Outputs
- Conditional sample galleries for each class
- CFG strength comparison (w = 0.5, 1.0, 2.0, 5.0, 10.0)
- Analysis of sample diversity vs conditioning strength
- Quantitative metrics: class accuracy, FID per class
- Failure case analysis

## 🎓 Learning Outcomes
- Understanding guidance mechanisms in diffusion models
- Experience with conditional generation
- Foundation for text-to-image diffusion models

## 📖 Resources
- Classifier-Free Diffusion Guidance (Ho & Salimans, 2022)
- GLIDE paper for practical CFG implementation
- Analysis of guidance scale effects

---
**Time Estimate**: 4-5 hours  
**Difficulty**: ⭐⭐⭐⭐☆