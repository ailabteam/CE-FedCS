# app.py

print("--- FEDERATED LEARNING EXPERIMENT RUNNER (v17) ---")

import argparse # THÊM MỚI: Để đọc tham số dòng lệnh
import pickle   # THÊM MỚI: Để lưu kết quả
from collections import OrderedDict
import warnings
import os
from typing import List, Tuple, Dict, Optional

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from utils import load_data_cifar10, get_device

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 1. SHARED COMPONENTS (MODEL, CLIENT, TRAIN/TEST FUNCTIONS)
#    (Không có thay đổi nào trong phần này)
# =============================================================================
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def train(net, trainloader, epochs, device, optimizer_name, lr):
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def train_fedprox(net, trainloader, epochs, device, global_params, mu, optimizer_name, lr):
    criterion = torch.nn.CrossEntropyLoss()
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    global_params_torch = [torch.from_numpy(p).to(device) for p in global_params]
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            proximal_term = 0.0
            for local_param, global_param in zip(net.parameters(), global_params_torch):
                proximal_term += torch.square((local_param - global_param).norm(2))
            loss += (mu / 2) * proximal_term
            loss.backward()
            optimizer.step()

def test(net, testloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss / len(testloader), accuracy

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, testloader, device):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]
    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.net.load_state_dict(state_dict, strict=True)
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train_method = config.get("train_method", "standard")
        epochs = config.get("local_epochs", 1)
        lr = config.get("lr", 0.01)
        mu = config.get("mu", 0.01)
        optimizer = config.get("optimizer", "sgd")
        if train_method == "fedprox":
            train_fedprox(self.net, self.trainloader, epochs, self.device, parameters, mu, optimizer, lr)
        else:
            train(self.net, self.trainloader, epochs, self.device, optimizer, lr)
        return self.get_parameters(config={}), len(self.trainloader.dataset), {}
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}

def client_fn_factory(trainloaders, testloader, device):
    def client_fn(cid: str) -> fl.client.Client:
        net = Net().to(device)
        trainloader = trainloaders[int(cid)]
        return FlowerClient(net, trainloader, testloader, device).to_client()
    return client_fn

def get_evaluate_fn(testloader, device):
    def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict) -> Optional[Tuple[float, Dict]]:
        net = Net().to(device)
        params_dict = zip(net.state_dict().keys(), [torch.from_numpy(p) for p in parameters])
        state_dict = OrderedDict(params_dict)
        net.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(net, testloader, device)
        print(f"Round {server_round} | Accuracy: {accuracy:.4f} | Loss: {loss:.4f}")
        return loss, {"accuracy": accuracy}
    return evaluate

# =============================================================================
# 2. MAIN EXECUTION LOGIC
# =============================================================================

def main(args):
    """Load data, define and run a single experiment."""
    # Common settings
    NUM_ROUNDS = 50
    NUM_CLIENTS = 100
    CLIENTS_PER_ROUND = 10
    
    # Load data once
    DEVICE = get_device()
    print(f"Running on device: {DEVICE}")
    trainloaders, testloader = load_data_cifar10(num_clients=NUM_CLIENTS, batch_size=32, shards_per_client=5)
    
    # Create a generic client_fn and evaluate_fn
    client_fn = client_fn_factory(trainloaders, testloader, DEVICE)
    evaluate_fn = get_evaluate_fn(testloader, DEVICE)

    # Define all experiment configurations
    experiments = {
        "fedavg": {
            "name": "FedAvg (Baseline, E=5)",
            "fit_config_fn": lambda sr: {"train_method": "standard", "local_epochs": 5, "lr": 0.01, "optimizer": "sgd"},
        },
        "proposed": {
            "name": "Proposed (Balanced FedProx, E=5, mu=0.1)",
            "fit_config_fn": lambda sr: {
                "train_method": "fedprox", 
                "local_epochs": 5,
                "lr": 0.01 * (0.1 ** (sr // 20)), 
                "mu": 0.1,
                "optimizer": "adam"
            },
        },
    }

    # Select the experiment to run based on command-line argument
    exp_config = experiments[args.experiment]

    print(f"\n{'='*30}\n[RUNNING EXPERIMENT]: {exp_config['name']}\n{'='*30}")
    
    # Configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=CLIENTS_PER_ROUND / NUM_CLIENTS,
        min_fit_clients=CLIENTS_PER_ROUND,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=exp_config["fit_config_fn"],
    )

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_gpus": 1.0 if get_device().type == "cuda" else 0.0},
    )
    
    # Save the results
    if not os.path.exists("results"):
        os.makedirs("results")
    
    save_path = f"results/{args.experiment}_history.pkl"
    with open(save_path, "wb") as f:
        pickle.dump(history, f)
    
    print(f"\nExperiment '{exp_config['name']}' finished. Results saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Experiment Runner")
    parser.add_argument(
        "experiment", 
        type=str, 
        choices=["fedavg", "proposed"],
        help="The name of the experiment to run."
    )
    args = parser.parse_args()
    main(args)
