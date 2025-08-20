# Day 1: Gaussian Noise Basics

## 🎯 Objective
Understand the fundamental concepts of Gaussian noise and how it affects image data through progressive noise injection.

## 📋 Tasks

### Main Implementation
- **Implement a forward Gaussian noising process for images (e.g., MNIST)**
  - Create a function that adds Gaussian noise with varying standard deviations
  - Apply noise progressively with different noise levels (σ = 0.1, 0.3, 0.5, 0.7, 1.0)
  - Ensure proper normalization and data handling

### Visualization
- **Visualize how increasing noise levels destroy image content**
  - Create side-by-side comparisons of original vs noised images
  - Plot noise level vs image quality metrics
  - Generate animations showing progressive noise addition

## 📚 Key Concepts
- Gaussian (Normal) distribution properties
- Signal-to-noise ratio (SNR)
- Image degradation through noise injection
- Statistical properties of noise

## 🔧 Implementation Tips
- Use `torch.randn()` or `np.random.normal()` for Gaussian noise
- Consider image pixel value ranges (0-1 or 0-255)
- Implement both additive and multiplicative noise variants
- Visualize distributions of noised pixels

## 📊 Expected Outputs
- Noised image samples at different noise levels
- Quantitative analysis of image degradation
- Plots showing noise distribution characteristics

## 🎓 Learning Outcomes
- Foundation understanding of noise processes
- Preparation for diffusion model concepts
- Practical experience with image data handling

## 📖 Resources
- Understanding Gaussian distributions
- Image processing fundamentals
- Statistical noise analysis

---
**Time Estimate**: 2-3 hours  
**Difficulty**: ⭐⭐☆☆☆