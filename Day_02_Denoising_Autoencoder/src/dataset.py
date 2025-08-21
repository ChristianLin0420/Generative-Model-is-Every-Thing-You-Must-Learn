"""
Dataset utilities with on-the-fly noising for Day 2: Denoising Autoencoder
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from .noise import add_gaussian_noise


class NoisyDataset(Dataset):
    """Dataset wrapper that applies noise on-the-fly during training."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        noise_sigmas: list,
        clip_range: Tuple[float, float] = (0, 1),
        training: bool = True,
        generator_seed: Optional[int] = None
    ):
        self.base_dataset = base_dataset
        self.noise_sigmas = noise_sigmas
        self.clip_range = clip_range
        self.training = training
        
        # Create generator for reproducible noise
        self.generator = torch.Generator()
        if generator_seed is not None:
            self.generator.manual_seed(generator_seed)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Returns:
            clean_image: Original clean image
            noisy_image: Image with added Gaussian noise  
            sigma: Noise level used
        """
        clean_image, label = self.base_dataset[idx]
        
        # Randomly select noise level for training
        if self.training:
            # Create a single scalar tensor and sample uniformly
            sigma = torch.empty(1).uniform_(
                min(self.noise_sigmas), 
                max(self.noise_sigmas),
                generator=self.generator
            ).item()
        else:
            # For evaluation, use fixed pattern based on index
            sigma = self.noise_sigmas[idx % len(self.noise_sigmas)]
        
        # Add noise
        noisy_image = add_gaussian_noise(
            clean_image.unsqueeze(0), 
            sigma, 
            clip_range=self.clip_range,
            generator=self.generator
        ).squeeze(0)
        
        return clean_image, noisy_image, sigma


def get_dataset_loaders(
    dataset_name: str,
    root: Union[str, Path],
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: str = "zero_one",
    train_sigmas: list = [0.1, 0.2, 0.3, 0.5],
    test_sigmas: list = [0.1, 0.3, 0.7, 1.0],
    generator_seed: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and test DataLoaders for specified dataset.
    
    Args:
        dataset_name: 'mnist' (only MNIST is supported)
        root: Root directory for data
        batch_size: Batch size for training
        num_workers: Number of worker processes
        normalize: Normalization scheme ('zero_one' or 'minus_one_one')
        train_sigmas: Noise levels for training
        test_sigmas: Noise levels for testing (can include unseen levels)
        generator_seed: Seed for noise generation
    
    Returns:
        train_loader, test_loader
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    # Define transforms based on normalization
    if normalize == "zero_one":
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1]
        ])
        clip_range = (0, 1)
    elif normalize == "minus_one_one":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # [0, 1] -> [-1, 1]
        ])
        clip_range = (-1, 1)
    else:
        raise ValueError(f"Unknown normalization: {normalize}")
    
    # Only MNIST is supported now
    
    # Create base datasets - MNIST only
    if dataset_name.lower() != "mnist":
        raise ValueError(f"Only MNIST dataset is supported, got: {dataset_name}")
    
    train_dataset = MNIST(root=root, train=True, transform=transform, download=True)
    test_dataset = MNIST(root=root, train=False, transform=transform, download=True)
    input_shape = (1, 28, 28)
    
    # Wrap with noisy dataset
    noisy_train_dataset = NoisyDataset(
        train_dataset, 
        train_sigmas, 
        clip_range, 
        training=True,
        generator_seed=generator_seed
    )
    noisy_test_dataset = NoisyDataset(
        test_dataset, 
        test_sigmas, 
        clip_range, 
        training=False,
        generator_seed=generator_seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        noisy_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    test_loader = DataLoader(
        noisy_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Created {dataset_name.upper()} loaders:")
    print(f"  - Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  - Test: {len(test_dataset)} samples, {len(test_loader)} batches")
    print(f"  - Input shape: {input_shape}")
    print(f"  - Normalization: {normalize} -> {clip_range}")
    print(f"  - Train sigmas: {train_sigmas}")
    print(f"  - Test sigmas: {test_sigmas}")
    
    return train_loader, test_loader


def get_data_stats(dataloader: DataLoader, max_batches: int = 10) -> dict:
    """Compute dataset statistics."""
    all_clean = []
    all_noisy = []
    all_sigmas = []
    
    print("Computing dataset statistics...")
    for batch_idx, (clean, noisy, sigmas) in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        all_clean.append(clean)
        all_noisy.append(noisy) 
        all_sigmas.extend(sigmas.tolist())
    
    all_clean = torch.cat(all_clean, dim=0)
    all_noisy = torch.cat(all_noisy, dim=0)
    
    stats = {
        'clean_mean': all_clean.mean().item(),
        'clean_std': all_clean.std().item(),
        'clean_min': all_clean.min().item(),
        'clean_max': all_clean.max().item(),
        'noisy_mean': all_noisy.mean().item(),
        'noisy_std': all_noisy.std().item(),
        'noisy_min': all_noisy.min().item(),
        'noisy_max': all_noisy.max().item(),
        'sigma_range': [min(all_sigmas), max(all_sigmas)],
        'shape': list(all_clean.shape[1:])
    }
    
    print(f"Dataset statistics: {stats}")
    return stats


def denormalize(tensor: torch.Tensor, normalize_type: str = "zero_one") -> torch.Tensor:
    """Denormalize tensor for visualization."""
    if normalize_type == "zero_one":
        return torch.clamp(tensor, 0, 1)
    elif normalize_type == "minus_one_one":
        return torch.clamp((tensor + 1) / 2, 0, 1)
    else:
        raise ValueError(f"Unknown normalization type: {normalize_type}")


class FixedNoiseDataset(Dataset):
    """Dataset with fixed noise patterns for evaluation consistency."""
    
    def __init__(
        self,
        base_dataset: Dataset,
        noise_sigma: float,
        clip_range: Tuple[float, float] = (0, 1),
        seed: int = 42
    ):
        self.base_dataset = base_dataset
        self.noise_sigma = noise_sigma
        self.clip_range = clip_range
        
        # Pre-generate all noise patterns for consistency
        torch.manual_seed(seed)
        self.noise_patterns = []
        for i in range(len(base_dataset)):
            img, _ = base_dataset[i]
            noise = torch.randn_like(img) * noise_sigma
            self.noise_patterns.append(noise)
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        clean_image, label = self.base_dataset[idx]
        noise = self.noise_patterns[idx]
        noisy_image = torch.clamp(clean_image + noise, *self.clip_range)
        
        return clean_image, noisy_image, self.noise_sigma