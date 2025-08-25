"""
Dataset loaders for MNIST and CIFAR-10 with proper normalization
Optional class labels for conditional generation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path


class NormalizeInverse(nn.Module):
    """Inverse normalization transform"""
    
    def __init__(self, mean: Tuple[float, ...], std: Tuple[float, ...]):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
        
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)


def get_mnist_transforms(image_size: int = 32) -> Dict[str, transforms.Compose]:
    """Get MNIST transforms for training and validation"""
    
    # MNIST normalization: mean=0.5, std=0.5 to get [-1, 1] range
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [0, 1] -> [-1, 1]
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    return {
        "train": train_transform,
        "val": val_transform,
        "inverse": NormalizeInverse([0.5], [0.5])
    }


def get_cifar10_transforms(image_size: int = 32) -> Dict[str, transforms.Compose]:
    """Get CIFAR-10 transforms for training and validation"""
    
    # CIFAR-10 normalization to [-1, 1]
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    
    train_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),  # Slight rotation
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    return {
        "train": train_transform,
        "val": val_transform,
        "inverse": NormalizeInverse(mean, std)
    }


class ConditionalDataset(Dataset):
    """
    Wrapper for conditional generation with labels
    Can drop labels with some probability for classifier-free guidance
    """
    
    def __init__(
        self, 
        dataset: Dataset,
        num_classes: int,
        label_dropout: float = 0.1,
        unconditional_label: int = -1
    ):
        self.dataset = dataset
        self.num_classes = num_classes
        self.label_dropout = label_dropout
        self.unconditional_label = unconditional_label
        
    def __len__(self) -> int:
        return len(self.dataset)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image, label = self.dataset[idx]
        
        # Randomly drop labels for CFG training
        if torch.rand(1) < self.label_dropout:
            label = self.unconditional_label
            
        return image, label


def get_dataset(
    dataset_name: str,
    data_dir: str = "./data",
    image_size: int = 32,
    train: bool = True,
    download: bool = True,
    conditional: bool = False,
    label_dropout: float = 0.1
) -> Dataset:
    """
    Get dataset by name
    
    Args:
        dataset_name: "mnist" or "cifar10"
        data_dir: directory to store/load data
        image_size: target image size
        train: whether to get training set
        download: whether to download if not exists
        conditional: whether to return labels for conditional generation
        label_dropout: probability of dropping labels (for CFG)
        
    Returns:
        Dataset instance
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = dataset_name.lower()
    
    if dataset_name == "mnist":
        transforms_dict = get_mnist_transforms(image_size)
        transform = transforms_dict["train"] if train else transforms_dict["val"]
        
        dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=train,
            transform=transform,
            download=download
        )
        num_classes = 10
        
    elif dataset_name == "cifar10":
        transforms_dict = get_cifar10_transforms(image_size)
        transform = transforms_dict["train"] if train else transforms_dict["val"]
        
        dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=train,
            transform=transform,
            download=download
        )
        num_classes = 10
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    # Wrap with conditional dataset if requested
    if conditional:
        dataset = ConditionalDataset(
            dataset=dataset,
            num_classes=num_classes,
            label_dropout=label_dropout
        )
        
    return dataset


def get_dataloader(
    dataset_name: str,
    data_dir: str = "./data", 
    batch_size: int = 32,
    image_size: int = 32,
    train: bool = True,
    shuffle: Optional[bool] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    download: bool = True,
    conditional: bool = False,
    label_dropout: float = 0.1
) -> DataLoader:
    """
    Get DataLoader for specified dataset
    
    Args:
        dataset_name: "mnist" or "cifar10"
        data_dir: directory to store/load data
        batch_size: batch size
        image_size: target image size  
        train: whether to get training set
        shuffle: whether to shuffle (defaults to train)
        num_workers: number of dataloader workers
        pin_memory: whether to pin memory for faster GPU transfer
        download: whether to download if not exists
        conditional: whether to return labels
        label_dropout: probability of dropping labels (for CFG)
        
    Returns:
        DataLoader instance
    """
    if shuffle is None:
        shuffle = train
        
    dataset = get_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        image_size=image_size,
        train=train,
        download=download,
        conditional=conditional,
        label_dropout=label_dropout
    )
    
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train  # Drop last incomplete batch for training
    )


def get_dataset_stats(dataset_name: str) -> Dict[str, Any]:
    """Get dataset statistics"""
    
    stats = {
        "mnist": {
            "channels": 1,
            "image_size": 28,
            "num_classes": 10,
            "train_samples": 60000,
            "test_samples": 10000,
            "class_names": [str(i) for i in range(10)]
        },
        "cifar10": {
            "channels": 3,
            "image_size": 32,
            "num_classes": 10,
            "train_samples": 50000,
            "test_samples": 10000,
            "class_names": [
                "airplane", "automobile", "bird", "cat", "deer",
                "dog", "frog", "horse", "ship", "truck"
            ]
        }
    }
    
    dataset_name = dataset_name.lower()
    if dataset_name not in stats:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    return stats[dataset_name]


def get_inverse_transform(dataset_name: str, image_size: int = 32) -> nn.Module:
    """Get inverse normalization transform for visualization"""
    
    if dataset_name.lower() == "mnist":
        return get_mnist_transforms(image_size)["inverse"]
    elif dataset_name.lower() == "cifar10":
        return get_cifar10_transforms(image_size)["inverse"] 
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


class InfiniteDataLoader:
    """
    Wrapper that creates an infinite dataloader
    Useful for training with arbitrary number of steps
    """
    
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.iter = None
        
    def __iter__(self):
        return self
        
    def __next__(self):
        try:
            if self.iter is None:
                self.iter = iter(self.dataloader)
            return next(self.iter)
        except StopIteration:
            # Restart the dataloader
            self.iter = iter(self.dataloader)
            return next(self.iter)
            
    def __len__(self):
        return len(self.dataloader)


def create_dataloaders(config: Dict[str, Any]) -> Dict[str, DataLoader]:
    """Create train/val dataloaders from config"""
    
    dataset_config = config["dataset"]
    dataset_name = dataset_config["name"]
    
    # Training dataloader
    train_loader = get_dataloader(
        dataset_name=dataset_name,
        data_dir=dataset_config.get("data_dir", "./data"),
        batch_size=config["training"]["batch_size"],
        image_size=dataset_config.get("image_size", 32),
        train=True,
        num_workers=dataset_config.get("num_workers", 4),
        conditional=dataset_config.get("conditional", False),
        label_dropout=dataset_config.get("label_dropout", 0.1)
    )
    
    # Validation dataloader (smaller batch size, no augmentation)
    val_batch_size = min(config["training"]["batch_size"], 64)
    val_loader = get_dataloader(
        dataset_name=dataset_name,
        data_dir=dataset_config.get("data_dir", "./data"),
        batch_size=val_batch_size,
        image_size=dataset_config.get("image_size", 32),
        train=False,
        shuffle=False,
        num_workers=dataset_config.get("num_workers", 4),
        conditional=dataset_config.get("conditional", False),
        label_dropout=0.0  # No label dropout for validation
    )
    
    return {
        "train": train_loader,
        "val": val_loader,
        "infinite_train": InfiniteDataLoader(train_loader)
    }