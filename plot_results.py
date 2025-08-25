# plot_results.py

import pickle
import matplotlib.pyplot as plt
import os
from typing import Dict

def plot_results(results: Dict):
    """Plot accuracy curves for all experiments."""
    plt.figure(figsize=(12, 8))
    
    for name, history in results.items():
        # Check if history object is valid and has the required metric
        if not history or not hasattr(history, 'metrics_centralized') or 'accuracy' not in history.metrics_centralized:
            print(f"Warning: Skipping '{name}' due to missing or invalid history data.")
            continue
            
        rounds = [int(r) for r, _ in history.metrics_centralized["accuracy"]]
        accuracies = [float(acc) for _, acc in history.metrics_centralized["accuracy"]]
        plt.plot(rounds, accuracies, marker='o', linestyle='-', label=name, markersize=4)

    plt.title("Comparison of FL Algorithms on Moderate Non-IID CIFAR-10")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Model Accuracy")
    plt.grid(True)
    plt.legend()
    plt.xticks(range(0, 51, 5))
    plt.ylim(0.1, 0.7)
    
    if not os.path.exists("figures"):
        os.makedirs("figures")
    
    figure_path = "figures/final_comparison.png"
    plt.savefig(figure_path, dpi=600, bbox_inches='tight')
    print(f"\nComparison plot saved to {figure_path}")

if __name__ == "__main__":
    results_dir = "results"
    experiment_files = {
        "FedAvg (Baseline)": "fedavg_history.pkl",
        "Proposed (Balanced FedProx)": "proposed_history.pkl",
    }
    
    all_histories = {}
    
    for name, filename in experiment_files.items():
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            print(f"Loading results for '{name}' from {path}")
            with open(path, "rb") as f:
                history = pickle.load(f)
                all_histories[name] = history
        else:
            print(f"Warning: Result file not found for '{name}' at {path}")

    if all_histories:
        plot_results(all_histories)
    else:
        print("No result files found to plot.")
