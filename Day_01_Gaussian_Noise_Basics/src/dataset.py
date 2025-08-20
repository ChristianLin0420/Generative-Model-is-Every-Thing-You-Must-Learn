"""
Dataset utilities for Day 1: Gaussian Noise Basics
"""

from pathlib import Path
from typing import Tuple, Union

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def get_mnist_loader(
    root: Union[str, Path],
    split: str = "train",
    batch_size: int = 64,
    normalize_range: Tuple[float, float] = (0, 1),
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True
) -> DataLoader:
    """
    Get MNIST DataLoader with specified normalization range.
    
    Args:
        root: Root directory for dataset
        split: 'train' or 'test'
        batch_size: Batch size
        normalize_range: Target range for normalization, either (0, 1) or (-1, 1)
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to use pinned memory
    
    Returns:
        DataLoader for MNIST dataset
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    
    # Define transforms based on normalization range
    if normalize_range == (0, 1):
        # Standard normalization to [0, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),  # This already gives [0, 1]
        ])
    elif normalize_range == (-1, 1):
        # Normalize to [-1, 1]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # (x - 0.5) / 0.5
        ])
    else:
        raise ValueError(f"Unsupported normalize_range: {normalize_range}")
    
    # Determine training flag
    train = (split == "train")
    
    # Create dataset
    dataset = MNIST(
        root=str(root),
        train=train,
        transform=transform,
        download=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # For consistent batch sizes
    )
    
    print(f"Created MNIST {split} DataLoader:")
    print(f"  - Dataset size: {len(dataset)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Number of batches: {len(dataloader)}")
    print(f"  - Normalization range: {normalize_range}")
    
    return dataloader


def get_sample_batch(dataloader: DataLoader, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a single batch from dataloader."""
    images, labels = next(iter(dataloader))
    return images.to(device), labels.to(device)


def denormalize_tensor(
    tensor: torch.Tensor,
    normalize_range: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    """
    Denormalize tensor back to [0, 1] range for visualization.
    
    Args:
        tensor: Input tensor
        normalize_range: Original normalization range
    
    Returns:
        Denormalized tensor in [0, 1] range
    """
    if normalize_range == (0, 1):
        return tensor.clamp(0, 1)
    elif normalize_range == (-1, 1):
        return (tensor + 1) / 2  # [-1, 1] -> [0, 1]
    else:
        raise ValueError(f"Unsupported normalize_range: {normalize_range}")


def get_data_statistics(dataloader: DataLoader) -> dict:
    """Compute basic statistics of the dataset."""
    all_data = []
    
    print("Computing dataset statistics...")
    for batch_idx, (images, _) in enumerate(dataloader):
        all_data.append(images)
        if batch_idx >= 10:  # Sample first few batches for efficiency
            break
    
    all_data = torch.cat(all_data, dim=0)
    
    stats = {
        'mean': all_data.mean().item(),
        'std': all_data.std().item(), 
        'min': all_data.min().item(),
        'max': all_data.max().item(),
        'shape': list(all_data.shape[1:])  # Exclude batch dimension
    }
    
    print(f"Data statistics: {stats}")
    return stats