# strategy.py

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

class FedPoCStrategy(fl.server.strategy.FedAvg):
    def __init__(self, num_candidates: int = 20, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Dictionary to store last known loss of each client
        self.client_losses: Dict[str, float] = {}
        # Number of clients to consider in each round
        self.num_candidates = num_candidates
        print(f"FedPoC Strategy initialized with {self.num_candidates} candidates per round.")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Implement Power-of-Choice client selection."""
        
        # 1. Candidate Selection
        num_available = client_manager.num_available()
        sample_size = min(self.num_candidates, num_available)
        
        if sample_size < self.min_fit_clients:
            print(f"Not enough clients available ({num_available}) to select from.")
            return []

        candidates = client_manager.sample(
            num_clients=sample_size, 
            min_num_clients=sample_size
        )
        
        print(f"[Round {server_round}] Step 1: Sampled {len(candidates)} candidates for evaluation.")

        # 2. Rank candidates based on their last known loss
        candidate_losses = []
        # Use a high default loss for clients that have never trained
        avg_loss = np.mean(list(self.client_losses.values())) if self.client_losses else 2.3 
        for client in candidates:
            loss = self.client_losses.get(client.cid, avg_loss * 1.5) # Prioritize new clients
            candidate_losses.append((client, loss))
            
        # 3. Final Selection: Choose top-K clients with the highest loss
        candidate_losses.sort(key=lambda x: x[1], reverse=True)
        
        num_to_select = self.min_fit_clients
        selected_clients = [client for client, loss in candidate_losses[:num_to_select]]
        
        print(f"[Round {server_round}] Step 2: Selected top {num_to_select} clients for training: {[c.cid for c in selected_clients]}")
        
        # 4. Return instructions for the selected clients
        fit_ins = FitIns(parameters, {})
        return [(client, fit_ins) for client in selected_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results and update client losses."""
        
        # Update client losses from the results of this round
        for client, fit_res in results:
            if fit_res.metrics and "loss" in fit_res.metrics:
                self.client_losses[client.cid] = fit_res.metrics["loss"]
        
        # Call the parent class (FedAvg) method to aggregate model weights
        return super().aggregate_fit(server_round, results, failures)
