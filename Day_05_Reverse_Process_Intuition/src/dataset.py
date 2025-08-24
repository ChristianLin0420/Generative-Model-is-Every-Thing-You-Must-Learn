"""Dataset loading and preprocessing for DDPM training

Supports MNIST and CIFAR-10 datasets with proper normalization
and optional class conditioning.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image


class NormalizeToRange:
    """Custom transform to normalize images to [0, 1] range."""
    
    def __init__(self, min_val: float = 0.0, max_val: float = 1.0):
        self.min_val = min_val
        self.max_val = max_val
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Normalize from [0, 1] to [min_val, max_val]
        return tensor * (self.max_val - self.min_val) + self.min_val


class ConditionalDataset(Dataset):
    """Wrapper to add class conditioning to existing dataset."""
    
    def __init__(self, dataset: Dataset, return_labels: bool = True):
        self.dataset = dataset
        self.return_labels = return_labels
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if hasattr(self.dataset, '__getitem__'):
            data = self.dataset[idx]
            if isinstance(data, tuple) and len(data) == 2:
                image, label = data
                if self.return_labels:
                    return image, label
                else:
                    return (image,)
            else:
                return (data,)
        else:
            raise ValueError("Dataset must support indexing")


def get_mnist_transforms(
    image_size: int = 32,
    normalize_range: Tuple[float, float] = (0.0, 1.0)
) -> transforms.Compose:
    """Get MNIST preprocessing transforms.
    
    Args:
        image_size: Target image size (MNIST will be resized)
        normalize_range: Target normalization range
    
    Returns:
        Composed transforms
    """
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # This normalizes to [0, 1]
    ]
    
    # Add custom normalization if needed
    if normalize_range != (0.0, 1.0):
        transform_list.append(NormalizeToRange(normalize_range[0], normalize_range[1]))
    
    return transforms.Compose(transform_list)


def get_cifar10_transforms(
    image_size: int = 32,
    normalize_range: Tuple[float, float] = (0.0, 1.0),
    augment: bool = False
) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get CIFAR-10 preprocessing transforms.
    
    Args:
        image_size: Target image size
        normalize_range: Target normalization range
        augment: Whether to apply data augmentation
    
    Returns:
        Train and test transforms
    """
    # Base transforms
    base_transforms = [
        transforms.Resize((image_size, image_size)) if image_size != 32 else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
    ]
    
    # Add custom normalization if needed
    if normalize_range != (0.0, 1.0):
        base_transforms.append(NormalizeToRange(normalize_range[0], normalize_range[1]))
    
    test_transform = transforms.Compose(base_transforms)
    
    # Training transforms with optional augmentation
    train_transforms = base_transforms.copy()
    if augment:
        train_transforms.insert(-2, transforms.RandomHorizontalFlip(p=0.5))
        # Note: Avoid too much augmentation as it can hurt diffusion training
    
    train_transform = transforms.Compose(train_transforms)
    
    return train_transform, test_transform


def create_mnist_dataset(
    root: str = "data",
    download: bool = True,
    image_size: int = 32,
    normalize_range: Tuple[float, float] = (0.0, 1.0),
    return_labels: bool = False
) -> Tuple[Dataset, Dataset]:
    """Create MNIST train and test datasets.
    
    Args:
        root: Data directory
        download: Whether to download if not exists
        image_size: Target image size
        normalize_range: Normalization range
        return_labels: Whether to return class labels
    
    Returns:
        Train and test datasets
    """
    transform = get_mnist_transforms(image_size, normalize_range)
    
    train_dataset = datasets.MNIST(
        root=root, train=True, download=download, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=root, train=False, download=download, transform=transform
    )
    
    # Wrap with conditional dataset if needed
    if not return_labels:
        train_dataset = ConditionalDataset(train_dataset, return_labels=False)
        test_dataset = ConditionalDataset(test_dataset, return_labels=False)
    else:
        train_dataset = ConditionalDataset(train_dataset, return_labels=True)
        test_dataset = ConditionalDataset(test_dataset, return_labels=True)
    
    return train_dataset, test_dataset


def create_cifar10_dataset(
    root: str = "data",
    download: bool = True,
    image_size: int = 32,
    normalize_range: Tuple[float, float] = (0.0, 1.0),
    augment: bool = False,
    return_labels: bool = False
) -> Tuple[Dataset, Dataset]:
    """Create CIFAR-10 train and test datasets.
    
    Args:
        root: Data directory
        download: Whether to download if not exists
        image_size: Target image size
        normalize_range: Normalization range
        augment: Whether to apply data augmentation
        return_labels: Whether to return class labels
    
    Returns:
        Train and test datasets
    """
    train_transform, test_transform = get_cifar10_transforms(
        image_size, normalize_range, augment
    )
    
    train_dataset = datasets.CIFAR10(
        root=root, train=True, download=download, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=root, train=False, download=download, transform=test_transform
    )
    
    # Wrap with conditional dataset if needed
    if not return_labels:
        train_dataset = ConditionalDataset(train_dataset, return_labels=False)
        test_dataset = ConditionalDataset(test_dataset, return_labels=False)
    else:
        train_dataset = ConditionalDataset(train_dataset, return_labels=True)
        test_dataset = ConditionalDataset(test_dataset, return_labels=True)
    
    return train_dataset, test_dataset


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True
) -> DataLoader:
    """Create DataLoader with sensible defaults.
    
    Args:
        dataset: PyTorch dataset
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        drop_last: Whether to drop last incomplete batch
    
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0
    )


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get dataset information.
    
    Args:
        dataset_name: Name of dataset ('mnist' or 'cifar10')
    
    Returns:
        Dictionary with dataset info
    """
    if dataset_name.lower() == "mnist":
        return {
            "name": "MNIST",
            "num_classes": 10,
            "input_channels": 1,
            "native_size": (28, 28),
            "class_names": [str(i) for i in range(10)]
        }
    elif dataset_name.lower() == "cifar10":
        return {
            "name": "CIFAR-10",
            "num_classes": 10,
            "input_channels": 3,
            "native_size": (32, 32),
            "class_names": [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dataset_from_config(config: Dict[str, Any]) -> Tuple[Dataset, Dataset, DataLoader, DataLoader]:
    """Create datasets and dataloaders from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        train_dataset, test_dataset, train_loader, test_loader
    """
    dataset_config = config["dataset"]
    training_config = config["training"]
    
    dataset_name = dataset_config["name"].lower()
    
    # Create datasets
    if dataset_name == "mnist":
        train_dataset, test_dataset = create_mnist_dataset(
            root=dataset_config["root"],
            download=dataset_config.get("download", True),
            image_size=dataset_config.get("image_size", 32),
            normalize_range=(0.0, 1.0),
            return_labels=dataset_config.get("return_labels", False)
        )
    elif dataset_name == "cifar10":
        train_dataset, test_dataset = create_cifar10_dataset(
            root=dataset_config["root"],
            download=dataset_config.get("download", True),
            image_size=dataset_config.get("image_size", 32),
            normalize_range=(0.0, 1.0),
            augment=training_config.get("augment", False),
            return_labels=dataset_config.get("return_labels", False)
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = create_dataloader(
        test_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False,
        num_workers=training_config.get("num_workers", 4),
        pin_memory=True,
        drop_last=False
    )
    
    return train_dataset, test_dataset, train_loader, test_loader