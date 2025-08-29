"""
Dataset loaders for MNIST and CIFAR with normalization and optional label loading.
"""

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional, Any
from pathlib import Path


class NormalizedDataset(Dataset):
    """Wrapper to apply custom normalization to dataset."""
    
    def __init__(self, dataset: Dataset, normalize_mode: str = "minus_one_one"):
        self.dataset = dataset
        self.normalize_mode = normalize_mode
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        image, label = self.dataset[idx]
        
        # Apply normalization
        if self.normalize_mode == "minus_one_one":
            image = image * 2.0 - 1.0  # [0, 1] -> [-1, 1]
        elif self.normalize_mode == "zero_one":
            pass  # Keep [0, 1]
        else:
            raise ValueError(f"Unknown normalization mode: {self.normalize_mode}")
        
        return image, label


def get_mnist_dataloader(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: str = "minus_one_one",
    train: bool = True,
    download: bool = True,
    shuffle: Optional[bool] = None
) -> DataLoader:
    """
    Get MNIST dataloader with specified normalization.
    
    Args:
        root: Data root directory
        batch_size: Batch size
        num_workers: Number of worker processes
        normalize: Normalization mode ("minus_one_one" or "zero_one")
        train: Whether to load training set
        download: Whether to download data if not found
        shuffle: Whether to shuffle data (defaults to train value)
        
    Returns:
        DataLoader for MNIST
    """
    if shuffle is None:
        shuffle = train
    
    # Basic transforms to convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts PIL to [0, 1] tensor
    ])
    
    # Load dataset
    dataset = torchvision.datasets.MNIST(
        root=root,
        train=train,
        transform=transform,
        download=download
    )
    
    # Apply custom normalization
    dataset = NormalizedDataset(dataset, normalize_mode=normalize)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train  # Drop last batch for training to ensure consistent batch size
    )
    
    return dataloader


def get_cifar10_dataloader(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: str = "minus_one_one",
    train: bool = True,
    download: bool = True,
    shuffle: Optional[bool] = None,
    augment: bool = False
) -> DataLoader:
    """
    Get CIFAR-10 dataloader with specified normalization.
    
    Args:
        root: Data root directory
        batch_size: Batch size
        num_workers: Number of worker processes
        normalize: Normalization mode ("minus_one_one" or "zero_one")
        train: Whether to load training set
        download: Whether to download data if not found
        shuffle: Whether to shuffle data (defaults to train value)
        augment: Whether to apply data augmentation (only for training)
        
    Returns:
        DataLoader for CIFAR-10
    """
    if shuffle is None:
        shuffle = train
    
    # Base transforms
    transform_list = []
    
    # Data augmentation for training
    if train and augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
        ])
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    transform = transforms.Compose(transform_list)
    
    # Load dataset
    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        transform=transform,
        download=download
    )
    
    # Apply custom normalization
    dataset = NormalizedDataset(dataset, normalize_mode=normalize)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train
    )
    
    return dataloader


def get_dataloader(
    dataset: str,
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    normalize: str = "minus_one_one",
    train: bool = True,
    download: bool = True,
    shuffle: Optional[bool] = None,
    **kwargs
) -> DataLoader:
    """
    Get dataloader for specified dataset.
    
    Args:
        dataset: Dataset name ("mnist" or "cifar10")
        root: Data root directory
        batch_size: Batch size
        num_workers: Number of worker processes
        normalize: Normalization mode
        train: Whether to load training set
        download: Whether to download data
        shuffle: Whether to shuffle data
        **kwargs: Additional dataset-specific arguments
        
    Returns:
        DataLoader for specified dataset
    """
    dataset = dataset.lower()
    
    if dataset == "mnist":
        return get_mnist_dataloader(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            train=train,
            download=download,
            shuffle=shuffle
        )
    elif dataset == "cifar10":
        return get_cifar10_dataloader(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            normalize=normalize,
            train=train,
            download=download,
            shuffle=shuffle,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_dataset_info(dataset: str) -> dict:
    """Get dataset information."""
    dataset = dataset.lower()
    
    if dataset == "mnist":
        return {
            'name': 'MNIST',
            'channels': 1,
            'height': 28,
            'width': 28,
            'num_classes': 10,
            'train_size': 60000,
            'test_size': 10000
        }
    elif dataset == "cifar10":
        return {
            'name': 'CIFAR-10',
            'channels': 3,
            'height': 32,
            'width': 32,
            'num_classes': 10,
            'train_size': 50000,
            'test_size': 10000
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def test_dataloader(dataset: str = "mnist", batch_size: int = 4) -> None:
    """Test dataloader by loading a few batches."""
    print(f"Testing {dataset} dataloader...")
    
    dataloader = get_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=0,  # Avoid multiprocessing issues in testing
        download=True
    )
    
    dataset_info = get_dataset_info(dataset)
    print(f"Dataset: {dataset_info['name']}")
    print(f"Image shape: {dataset_info['channels']}x{dataset_info['height']}x{dataset_info['width']}")
    print(f"Number of classes: {dataset_info['num_classes']}")
    
    # Load first batch
    for batch_idx, (images, labels) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
        print(f"  Labels: {labels.tolist()}")
        
        if batch_idx >= 2:  # Test first 3 batches
            break
    
    print("Dataloader test completed!")


if __name__ == "__main__":
    # Test both datasets
    test_dataloader("mnist")
    print()
    test_dataloader("cifar10")
