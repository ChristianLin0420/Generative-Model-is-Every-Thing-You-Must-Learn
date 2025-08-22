"""
Dataset utilities for MNIST and CIFAR-10.
Handles data loading, normalization, and train/test splits.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np


class NormalizeToRange(torch.nn.Module):
    """Normalize tensor to specified range."""
    
    def __init__(self, target_range: str = "zero_one"):
        super().__init__()
        self.target_range = target_range
    
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.target_range == "zero_one":
            # Normalize to [0, 1]
            return tensor
        elif self.target_range == "minus_one_one":
            # Normalize to [-1, 1]
            return 2.0 * tensor - 1.0
        else:
            raise ValueError(f"Unknown target range: {self.target_range}")
    
    def __repr__(self):
        return f"{self.__class__.__name__}(target_range='{self.target_range}')"


def get_mnist_transforms(normalize: str = "zero_one") -> Tuple[transforms.Compose, transforms.Compose]:
    """Get MNIST transforms for train and test sets."""
    
    # Base transforms
    base_transforms = [
        transforms.ToTensor(),  # Converts PIL to tensor and scales to [0,1]
        NormalizeToRange(normalize)
    ]
    
    # Training transforms (with optional augmentation)
    train_transforms = transforms.Compose([
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    ] + base_transforms)
    
    # Test transforms (no augmentation)
    test_transforms = transforms.Compose(base_transforms)
    
    return train_transforms, test_transforms


def get_cifar10_transforms(normalize: str = "zero_one") -> Tuple[transforms.Compose, transforms.Compose]:
    """Get CIFAR-10 transforms for train and test sets."""
    
    # Base transforms
    base_transforms = [
        transforms.ToTensor(),  # Converts PIL to tensor and scales to [0,1]
        NormalizeToRange(normalize)
    ]
    
    # Training transforms (with augmentation)
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ] + base_transforms)
    
    # Test transforms (no augmentation)
    test_transforms = transforms.Compose(base_transforms)
    
    return train_transforms, test_transforms


def get_dataset_stats(dataset_name: str) -> Dict[str, Any]:
    """Get dataset statistics and metadata."""
    stats = {
        "mnist": {
            "num_classes": 10,
            "input_shape": (1, 28, 28),
            "train_size": 60000,
            "test_size": 10000,
            "mean": [0.1307],
            "std": [0.3081]
        },
        "cifar10": {
            "num_classes": 10,
            "input_shape": (3, 32, 32),
            "train_size": 50000,
            "test_size": 10000,
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2470, 0.2435, 0.2616]
        }
    }
    
    if dataset_name not in stats:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return stats[dataset_name]


def create_dataloaders(
    dataset: str,
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: str = "zero_one",
    train_subset_ratio: Optional[float] = None,
    val_split: float = 0.0
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    Create train, test, and optional validation dataloaders.
    
    Args:
        dataset: 'mnist' or 'cifar10'
        root: Data directory path
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        normalize: Normalization scheme ('zero_one' or 'minus_one_one')
        train_subset_ratio: If provided, use only this fraction of training data
        val_split: Fraction of training data to use for validation
    
    Returns:
        train_loader, test_loader, val_loader (None if val_split=0)
    """
    
    if dataset == "mnist":
        train_transforms, test_transforms = get_mnist_transforms(normalize)
        
        # Load datasets
        train_dataset = torchvision.datasets.MNIST(
            root=root, train=True, download=True, transform=train_transforms
        )
        test_dataset = torchvision.datasets.MNIST(
            root=root, train=False, download=True, transform=test_transforms
        )
        
    elif dataset == "cifar10":
        train_transforms, test_transforms = get_cifar10_transforms(normalize)
        
        # Load datasets
        train_dataset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=train_transforms
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_transforms
        )
        
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Handle train subset
    if train_subset_ratio is not None:
        subset_size = int(len(train_dataset) * train_subset_ratio)
        indices = torch.randperm(len(train_dataset))[:subset_size]
        train_dataset = data.Subset(train_dataset, indices)
    
    # Handle validation split
    val_loader = None
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        
        train_dataset, val_dataset = data.random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True  # For stable batch norm
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, test_loader, val_loader


class UnlabeledDataset(Dataset):
    """Wrapper to remove labels from a dataset (for unsupervised learning)."""
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data, _ = self.dataset[idx]  # Ignore label
        return data


def create_unlabeled_dataloaders(
    dataset: str,
    root: str,
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: str = "zero_one"
) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders without labels (for unsupervised VAE training)."""
    
    train_loader, test_loader, _ = create_dataloaders(
        dataset=dataset,
        root=root,
        batch_size=batch_size,
        num_workers=num_workers,
        normalize=normalize,
        val_split=0.0
    )
    
    # Convert to unlabeled
    train_dataset = UnlabeledDataset(train_loader.dataset)
    test_dataset = UnlabeledDataset(test_loader.dataset)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
    
    return train_loader, test_loader


def sample_batch(dataloader: DataLoader, device: torch.device) -> torch.Tensor:
    """Sample a single batch from dataloader."""
    dataiter = iter(dataloader)
    batch = next(dataiter)
    
    # Handle both labeled and unlabeled data
    if isinstance(batch, (list, tuple)) and len(batch) == 2:
        # Labeled data: (images, labels)
        images, _ = batch
    else:
        # Unlabeled data: just images
        images = batch
    
    return images.to(device)


def get_sample_images(
    dataset: str,
    root: str,
    num_samples: int = 64,
    normalize: str = "zero_one"
) -> torch.Tensor:
    """Get a fixed set of sample images for visualization."""
    
    # Create test dataset
    if dataset == "mnist":
        _, test_transforms = get_mnist_transforms(normalize)
        test_dataset = torchvision.datasets.MNIST(
            root=root, train=False, download=True, transform=test_transforms
        )
    elif dataset == "cifar10":
        _, test_transforms = get_cifar10_transforms(normalize)
        test_dataset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=test_transforms
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    # Sample indices
    torch.manual_seed(42)  # For reproducible samples
    indices = torch.randperm(len(test_dataset))[:num_samples]
    
    # Collect images
    images = []
    for idx in indices:
        image, _ = test_dataset[idx]
        images.append(image)
    
    return torch.stack(images)


def compute_dataset_mean_std(dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute mean and standard deviation of dataset."""
    
    # Accumulate statistics
    channels_sum = 0
    channels_squared_sum = 0
    num_batches = 0
    
    for batch in dataloader:
        # Handle both labeled and unlabeled data
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            images, _ = batch
        else:
            images = batch
        
        # Flatten spatial dimensions
        images = images.view(images.size(0), images.size(1), -1)
        
        channels_sum += torch.mean(images, dim=[0, 2])
        channels_squared_sum += torch.mean(images**2, dim=[0, 2])
        num_batches += 1
    
    mean = channels_sum / num_batches
    std = torch.sqrt(channels_squared_sum / num_batches - mean**2)
    
    return mean, std