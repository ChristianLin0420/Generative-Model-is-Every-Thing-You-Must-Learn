# Day 5: Reverse Process Intuition

## 🎯 Objective
Build intuition for the reverse diffusion process by implementing a toy version with a simple neural network for denoising.

## 📋 Tasks

### Reverse Process Implementation
- **Implement a toy reverse process**
  - Start with fixed Gaussian prior p(x_T) = N(0,I)
  - Design simple reverse transitions p_θ(x_{t-1}|x_t)
  - Use basic neural network parameterization

### Neural Network Training
- **Train a small UNet to denoise**
  - Implement basic UNet architecture (simplified)
  - Train to predict noise ε given noisy image x_t and timestep t
  - Use simple loss: ||ε - ε_θ(x_t, t)||²

## 🏗️ Architecture Suggestions

### Toy UNet Components
- **Encoder**: Downsampling with conv blocks
- **Decoder**: Upsampling with conv transpose blocks
- **Skip Connections**: Concatenate features across scales
- **Time Embedding**: Sinusoidal positional encoding for timestep t

### Simplified Approach
- Start with MLP if UNet seems complex
- Focus on understanding rather than performance
- Use fewer timesteps (T=100 instead of 1000)

## 📚 Key Concepts
- Reverse process parameterization
- Neural network as function approximator
- Time conditioning in neural networks
- Denoising objective vs reconstruction objective

## 🔧 Implementation Tips
- Use pre-computed forward process from Day 4
- Start training with single timestep, then expand
- Implement time embedding (sine/cosine positional encoding)
- Use small dataset (MNIST subset) for faster iteration

## 📊 Expected Outputs
- Training curves for denoising loss
- Visual comparison: noisy input → network prediction → ground truth
- Qualitative assessment of denoising at different timesteps
- Simple reverse sampling attempts (may be poor quality)

## 🎓 Learning Outcomes
- Intuition for reverse process learning
- Understanding of neural network parameterization
- Foundation for full DDPM implementation

## 📖 Resources
- UNet architecture basics
- Time conditioning techniques
- Denoising score matching concepts

---
**Time Estimate**: 4-5 hours  
**Difficulty**: ⭐⭐⭐⭐☆