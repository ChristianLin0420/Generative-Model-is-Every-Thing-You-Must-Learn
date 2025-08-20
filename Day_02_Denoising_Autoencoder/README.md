# Day 2: Denoising Autoencoder (DAE)

## 🎯 Objective
Train a denoising autoencoder to reconstruct clean images from noisy inputs and understand its limitations compared to generative models.

## 📋 Tasks

### Model Implementation
- **Train a DAE on MNIST/CIFAR-10**
  - Design encoder-decoder architecture
  - Implement training loop with noisy input → clean output
  - Use MSE or L1 loss for reconstruction

### Analysis & Comparison
- **Observe limitations vs generative models**
  - Evaluate reconstruction quality across different noise levels
  - Analyze what the model learns vs memorizes
  - Compare reconstruction fidelity with original images

## 🏗️ Architecture Suggestions
- **Encoder**: Conv layers with decreasing spatial dimensions
- **Latent Space**: Bottleneck representation
- **Decoder**: Transposed conv layers for upsampling
- **Alternative**: Fully connected DAE for simpler datasets

## 📚 Key Concepts
- Autoencoder architecture principles
- Denoising vs standard autoencoders
- Reconstruction loss functions
- Overfitting in denoising tasks

## 🔧 Implementation Tips
- Add noise during training but not validation
- Use dropout or other regularization techniques
- Experiment with different noise types (Gaussian, salt-and-pepper)
- Monitor training/validation reconstruction loss

## 📊 Expected Outputs
- Training curves (loss vs epochs)
- Before/after denoising visualizations
- Comparison of reconstruction quality across noise levels
- Analysis of failure cases

## 🎓 Learning Outcomes
- Understanding autoencoder limitations
- Experience with reconstruction-based learning
- Foundation for understanding why generative models are needed

## 📖 Resources
- Denoising Autoencoders (Vincent et al., 2008)
- Autoencoder variants and applications
- Reconstruction vs generation trade-offs

---
**Time Estimate**: 3-4 hours  
**Difficulty**: ⭐⭐⭐☆☆