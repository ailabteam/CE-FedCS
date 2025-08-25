# utils.py

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision.datasets import CIFAR10
import numpy as np

def load_data_cifar10(num_clients: int, batch_size: int = 32, shards_per_client: int = 2):
    """Load CIFAR-10 and partition it into Non-IID subsets."""
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = CIFAR10(root="./data", train=False, download=True, transform=transform)

    # --- NEW, MORE ROBUST Non-IID PARTITIONING ---
    num_shards = num_clients * shards_per_client
    shard_size = len(trainset) // num_shards
    
    # Sort data by labels
    labels = np.array(trainset.targets)
    sorted_indices = np.argsort(labels)
    
    # Create shards
    shards = [sorted_indices[i * shard_size : (i + 1) * shard_size] for i in range(num_shards)]
    
    # Shuffle shards and assign to clients
    np.random.shuffle(shards)
    client_shards = [shards[i * shards_per_client : (i + 1) * shards_per_client] for i in range(num_clients)]

    # Create DataLoaders for each client
    trainloaders = []
    for shard_list in client_shards:
        client_indices = np.concatenate(shard_list)
        subset = Subset(trainset, client_indices)
        trainloaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True))
        
    testloader = DataLoader(testset, batch_size=batch_size * 2)

    return trainloaders, testloader

def get_device():
    """Get the appropriate device (GPU or CPU)."""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
