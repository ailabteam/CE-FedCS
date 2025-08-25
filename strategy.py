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
        self.client_losses: Dict[str, float] = {}
        self.num_candidates = num_candidates
        print(f"FedPoC Strategy initialized with {self.num_candidates} candidates per round.")

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Implement Power-of-Choice client selection."""
        
        num_available = client_manager.num_available()
        sample_size = min(self.num_candidates, num_available)
        
        if sample_size < self.min_fit_clients:
            return []

        candidates = client_manager.sample(
            num_clients=sample_size, 
            min_num_clients=sample_size
        )
        
        avg_loss = np.mean(list(self.client_losses.values())) if self.client_losses else 2.3
        candidate_losses = []
        for client in candidates:
            loss = self.client_losses.get(client.cid, avg_loss * 1.5)
            candidate_losses.append((client, loss))
            
        candidate_losses.sort(key=lambda x: x[1], reverse=True)
        
        num_to_select = self.min_fit_clients
        selected_clients = [client for client, loss in candidate_losses[:num_to_select]]
        
        print(f"[Round {server_round}] Selected top {num_to_select} clients from {sample_size} candidates.")
        
        fit_ins = FitIns(parameters, {})
        return [(client, fit_ins) for client in selected_clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        for client, fit_res in results:
            if fit_res.metrics and "loss" in fit_res.metrics:
                self.client_losses[client.cid] = fit_res.metrics["loss"]
        
        return super().aggregate_fit(server_round, results, failures)
