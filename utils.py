# utils.py

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

def load_data_cifar10(num_clients: int, batch_size: int = 32):
    """Load CIFAR-10 dataset and partition it into Non-IID subsets."""
    
    # 1. Load CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = CIFAR10(root="./data", train=False, download=True, transform=transform)

    # 2. Create Non-IID partitions
    # Sort dataset by labels
    labels = trainset.targets
    sorted_indices = sorted(range(len(labels)), key=lambda k: labels[k])
    
    # Partition indices among clients
    partition_size = len(sorted_indices) // num_clients
    client_indices = [sorted_indices[i * partition_size : (i + 1) * partition_size] for i in range(num_clients)]

    # 3. Create DataLoaders for each client
    trainloaders = []
    for indices in client_indices:
        subset = Subset(trainset, indices)
        trainloaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
        
    # The test set is shared among all clients for evaluation
    testloader = DataLoader(testset, batch_size=batch_size * 2)

    return trainloaders, testloader

def get_device():
    """Get the appropriate device (GPU or CPU)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
