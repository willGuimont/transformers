import math

import numpy as np
import torchvision.transforms as T
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CIFAR10


def split_dataset(dataset, split: float):
    # Split data
    num_data = len(dataset)
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    split = math.floor(split * num_data)
    train_idx, valid_idx = indices[:split], indices[split:]
    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)

    return train_dataset, valid_dataset


def get_cifar10_dataloaders(train_split: float, batch_size: int, num_workers: int = 4):
    # Data augmentation
    transform_train = T.Compose([
        T.RandAugment(4, 14),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # Load data
    cifar, cifar_test = CIFAR10('data/', train=True, download=True), CIFAR10('data/', train=False, download=True)
    cifar.transform = transform_train
    cifar_test.transform = transform_test

    # Split data
    train_dataset, valid_dataset = split_dataset(cifar, train_split)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(cifar_test, batch_size=batch_size)

    return train_loader, valid_loader, test_loader
