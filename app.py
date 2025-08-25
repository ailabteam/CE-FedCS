# app.py

print("--- FINAL PROPOSED METHOD: Balanced FedProx + Adam + LR Decay (v16) ---")

from collections import OrderedDict
import warnings
import os
from typing import List, Tuple, Dict, Optional

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from utils import load_data_cifar10, get_device

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# 1. EXPERIMENT CONFIGURATION
# =============================================================================

# This dictionary will hold the results of all experiments
all_histories = {}

# --- NEW BALANCED HYPERPARAMETERS for the Proposed Method ---
BALANCED_CONFIG = {
    "name": "Proposed (Balanced FedProx)",
    "fit_config_fn": lambda sr: {
        "train_method": "fedprox", 
        "local_epochs": 5,          # TĂNG: Cho phép học sâu hơn
        "lr": 0.01 * (0.1 ** (sr // 20)), # Giữ LR Decay
        "mu": 0.1,                  # GIẢM: Regularization vừa phải
        "optimizer": "adam"         # Giữ Adam
    },
}
# --- END NEW ---

# =============================================================================
# 2. SHARED COMPONENTS (MODEL, CLIENT, TRAIN/TEST FUNCTIONS)
# =============================================================================

class Net(nn.Module):
    # ... (no change)
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
    # ... (no change)
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
    # ... (no change)
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
    # ... (no change)
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
    # ... (no change)
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
    # ... (no change)
    def client_fn(cid: str) -> fl.client.Client:
        net = Net().to(device)
        trainloader = trainloaders[int(cid)]
        return FlowerClient(net, trainloader, testloader, device).to_client()
    return client_fn

def get_evaluate_fn(testloader, device):
    # ... (no change)
    def evaluate(server_round: int, parameters: List[np.ndarray], config: Dict) -> Optional[Tuple[float, Dict]]:
        net = Net().to(device)
        params_dict = zip(net.state_dict().keys(), [torch.from_numpy(p) for p in parameters])
        state_dict = OrderedDict(params_dict)
        net.load_state_dict(state_dict, strict=True)
        loss, accuracy = test(net, testloader, device)
        print(f"Server-side evaluation round {server_round} / accuracy {accuracy}")
        return loss, {"accuracy": accuracy}
    return evaluate

# =============================================================================
# 3. EXPERIMENT ORCHESTRATION
# =============================================================================

def run_experiment(experiment_config, client_fn, evaluate_fn):
    # ... (no change)
    print(f"\n{'='*30}\n[RUNNING EXPERIMENT]: {experiment_config['name']}\n{'='*30}")
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=experiment_config["clients_per_round"] / experiment_config["num_clients"],
        min_fit_clients=experiment_config["clients_per_round"],
        min_available_clients=experiment_config["num_clients"],
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=experiment_config["fit_config_fn"],
    )
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=experiment_config["num_clients"],
        config=fl.server.ServerConfig(num_rounds=experiment_config["num_rounds"]),
        strategy=strategy,
        client_resources={"num_gpus": 1.0 if get_device().type == "cuda" else 0.0},
    )
    return history

def plot_results(results: Dict):
    # ... (no change)
    plt.figure(figsize=(12, 8))
    for name, history in results.items():
        rounds = [int(r) for r, _ in history.metrics_centralized["accuracy"]]
        accuracies = [float(acc) for _, acc in history.metrics_centralized["accuracy"]]
        plt.plot(rounds, accuracies, marker='o', linestyle='-', label=name, markersize=4)
    plt.title("Comparison of FL Algorithms on Moderate Non-IID CIFAR-10")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Model Accuracy")
    plt.grid(True)
    plt.legend()
    plt.xticks(range(0, 51, 5))
    plt.ylim(0.1, 0.7) # Chỉnh lại ylim để thấy rõ sự khác biệt
    if not os.path.exists("figures"):
        os.makedirs("figures")
    figure_path = "figures/main_comparison.png"
    plt.savefig(figure_path, dpi=600, bbox_inches='tight')
    print(f"\nComparison plot saved to {figure_path}")


if __name__ == "__main__":
    NUM_ROUNDS = 50
    NUM_CLIENTS = 100
    CLIENTS_PER_ROUND = 10
    
    DEVICE = get_device()
    print(f"Running on device: {DEVICE}")
    trainloaders, testloader = load_data_cifar10(num_clients=NUM_CLIENTS, batch_size=32, shards_per_client=5)
    
    client_fn = client_fn_factory(trainloaders, testloader, DEVICE)
    evaluate_fn = get_evaluate_fn(testloader, DEVICE)

    # --- CHỈ CHẠY 2 THỬ NGHIỆM QUAN TRỌNG NHẤT ĐỂ TIẾT KIỆM THỜI GIAN ---
    experiments = [
        {
            "name": "FedAvg (Baseline)",
            "fit_config_fn": lambda sr: {"train_method": "standard", "local_epochs": 5, "lr": 0.01, "optimizer": "sgd"},
        },
        BALANCED_CONFIG, # Chạy cấu hình "cân bằng" mà chúng ta vừa định nghĩa
    ]

    for exp_config in experiments:
        exp_config["num_rounds"] = NUM_ROUNDS
        exp_config["num_clients"] = NUM_CLIENTS
        exp_config["clients_per_round"] = CLIENTS_PER_ROUND
        history = run_experiment(exp_config, client_fn, evaluate_fn)
        all_histories[exp_config["name"]] = history

    plot_results(all_histories)
