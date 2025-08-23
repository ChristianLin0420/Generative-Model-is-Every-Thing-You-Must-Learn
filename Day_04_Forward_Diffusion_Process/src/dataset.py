"""Dataset loading and preprocessing for MNIST and CIFAR-10."""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, Optional, Callable
from omegaconf import DictConfig

from .utils import normalize_to_neg_one_to_one


class NormalizeTransform:
    """Custom normalization transform."""
    
    def __init__(self, mode: str = "zero_one"):
        """
        Args:
            mode: "zero_one" for [0,1] or "minus_one_one" for [-1,1]
        """
        self.mode = mode
    
    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.mode == "minus_one_one":
            return normalize_to_neg_one_to_one(tensor)
        return tensor  # Already in [0,1] from ToTensor()


def get_mnist_transforms(normalize_mode: str = "zero_one") -> transforms.Compose:
    """Get MNIST transforms."""
    transform_list = [
        transforms.ToTensor(),  # Converts to [0,1]
    ]
    
    if normalize_mode == "minus_one_one":
        transform_list.append(NormalizeTransform("minus_one_one"))
    
    return transforms.Compose(transform_list)


def get_cifar10_transforms(normalize_mode: str = "zero_one") -> transforms.Compose:
    """Get CIFAR-10 transforms."""
    transform_list = [
        transforms.ToTensor(),  # Converts to [0,1]
    ]
    
    if normalize_mode == "minus_one_one":
        transform_list.append(NormalizeTransform("minus_one_one"))
    
    return transforms.Compose(transform_list)


def get_dataset(
    dataset_name: str,
    root: str,
    train: bool = True,
    normalize_mode: str = "zero_one"
) -> Dataset:
    """Get dataset by name.
    
    Args:
        dataset_name: "mnist" or "cifar10"
        root: Root directory to store dataset
        train: Whether to load training set
        normalize_mode: "zero_one" or "minus_one_one"
    
    Returns:
        Dataset object
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "mnist":
        transform = get_mnist_transforms(normalize_mode)
        dataset = torchvision.datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
    elif dataset_name == "cifar10":
        transform = get_cifar10_transforms(normalize_mode)
        dataset = torchvision.datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset


def get_dataloader(
    dataset_name: str,
    root: str,
    batch_size: int,
    train: bool = True,
    num_workers: int = 4,
    normalize_mode: str = "zero_one",
    shuffle: bool = True
) -> DataLoader:
    """Get dataloader for dataset.
    
    Args:
        dataset_name: "mnist" or "cifar10"
        root: Root directory to store dataset
        batch_size: Batch size
        train: Whether to load training set
        num_workers: Number of worker processes
        normalize_mode: "zero_one" or "minus_one_one"
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader object
    """
    dataset = get_dataset(dataset_name, root, train, normalize_mode)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def get_dataloader_from_config(config: DictConfig, train: bool = True) -> DataLoader:
    """Get dataloader from config.
    
    Args:
        config: Configuration object with data section
        train: Whether to load training set
    
    Returns:
        DataLoader object
    """
    return get_dataloader(
        dataset_name=config.data.dataset,
        root=config.data.root,
        batch_size=config.data.batch_size,
        train=train,
        num_workers=config.data.num_workers,
        normalize_mode=config.data.normalize,
        shuffle=train
    )


def get_sample_batch(
    dataset_name: str,
    root: str,
    batch_size: int = 16,
    normalize_mode: str = "zero_one"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get a sample batch from dataset.
    
    Args:
        dataset_name: "mnist" or "cifar10"
        root: Root directory to store dataset
        batch_size: Number of samples
        normalize_mode: "zero_one" or "minus_one_one"
    
    Returns:
        Tuple of (images, labels)
    """
    dataloader = get_dataloader(
        dataset_name=dataset_name,
        root=root,
        batch_size=batch_size,
        train=True,
        num_workers=0,  # No multiprocessing for single batch
        normalize_mode=normalize_mode,
        shuffle=True
    )
    
    return next(iter(dataloader))