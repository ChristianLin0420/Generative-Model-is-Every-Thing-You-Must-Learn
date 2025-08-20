# Day 6: Full DDPM Training Loop

## 🎯 Objective
Implement the complete DDPM training procedure with proper noise prediction objective and train on MNIST/CIFAR-10.

## 📋 Tasks

### DDPM Training Implementation
- **Implement the DDPM objective (predicting noise)**
  - Random timestep sampling for each training example
  - Noise prediction loss: L = ||ε - ε_θ(x_t, t)||²
  - Proper training loop with batching and optimization

### Full Training Pipeline
- **Train on MNIST/CIFAR-10**
  - Implement data loading and preprocessing
  - Set up proper training hyperparameters
  - Monitor training progress and loss curves
  - Save model checkpoints regularly

## 🧮 DDPM Training Algorithm

### Training Loop
```
for each training step:
    1. Sample batch of images x_0
    2. Sample timesteps t ~ Uniform(1, T)
    3. Sample noise ε ~ N(0, I)
    4. Compute noisy images x_t = √(ᾱ_t)x_0 + √(1-ᾱ_t)ε
    5. Predict noise: ε_pred = ε_θ(x_t, t)
    6. Compute loss: L = ||ε - ε_pred||²
    7. Backpropagate and update θ
```

## 🏗️ Architecture Details
- **UNet Backbone**: Multi-scale encoder-decoder with skip connections
- **Time Embedding**: Sinusoidal position encoding + MLP
- **Normalization**: GroupNorm (better than BatchNorm for diffusion)
- **Activation**: SiLU/Swish activation functions

## 🔧 Implementation Tips
- Use mixed precision training for efficiency
- Implement EMA (Exponential Moving Average) for stable training
- Set learning rate around 1e-4 to 2e-4
- Use Adam optimizer with (β₁=0.9, β₂=0.999)
- Batch size: 32-128 depending on GPU memory

## 📊 Expected Outputs
- Training loss curves over epochs
- Intermediate model checkpoints
- Memory usage and training time statistics
- Qualitative assessment during training (optional quick sampling)

## 🎓 Learning Outcomes
- Complete understanding of DDPM training
- Practical experience with diffusion model optimization
- Foundation for sampling and evaluation

## 📖 Resources
- DDPM paper (Ho et al., 2020) - Algorithm 1
- UNet architecture for diffusion models
- Training stability techniques for generative models

---
**Time Estimate**: 5-6 hours  
**Difficulty**: ⭐⭐⭐⭐⭐