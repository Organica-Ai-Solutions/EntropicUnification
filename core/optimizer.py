"""Training loop for the entropic-unification engine."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

from .coupling_layer import CouplingLayer
from .entropy_module import EntropyModule
from .geometry_engine import GeometryEngine
from .loss_functions import LossFunctions
from .quantum_engine import QuantumEngine


@dataclass
class OptimizerConfig:
    learning_rate: float = 1e-3
    steps: int = 1000
    checkpoint_interval: int = 100
    log_interval: int = 10
    metric_grad_clip: Optional[float] = 10.0
    results_path: str = "results"


class EntropicOptimizer:
    def __init__(
        self,
        quantum_engine: QuantumEngine,
        geometry_engine: GeometryEngine,
        entropy_module: EntropyModule,
        coupling_layer: CouplingLayer,
        loss_functions: LossFunctions,
        config: OptimizerConfig,
    ) -> None:
        self.quantum = quantum_engine
        self.geometry = geometry_engine
        self.entropy = entropy_module
        self.coupling = coupling_layer
        self.loss = loss_functions
        self.config = config

        self.history: Dict[str, List[float]] = {
            "total_loss": [],
            "einstein_loss": [],
            "entropy_loss": [],
            "curvature_loss": [],
            "smoothness_loss": [],
            "consistency": [],
        }

        self.results_path = Path(config.results_path)
        self.results_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Training step and loop
    # ------------------------------------------------------------------
    def optimization_step(
        self,
        state: torch.Tensor,
        partition: List[int],
        target_gradient: Optional[torch.Tensor],
        weights: Dict[str, float],
    ) -> Dict[str, torch.Tensor]:
        terms = self.coupling.compute_coupling_terms(state, partition)
        losses = self.loss.total_loss(terms, target_gradient, weights)

        update_info = self.coupling.update_coupling(
            state,
            partition,
            learning_rate=self.config.learning_rate,
            metric_grad_clip=self.config.metric_grad_clip,
        )

        return {**terms.__dict__, **losses, **update_info}

    def train(
        self,
        parameters: torch.Tensor,
        times: torch.Tensor,
        partition: List[int],
        initial_states: Optional[torch.Tensor] = None,
        target_gradient: Optional[torch.Tensor] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        if weights is None:
            weights = {
                "einstein": 1.0,
                "entropy": 1.0,
                "curvature": 0.1,
                "smoothness": 0.1,
            }

        states = self.quantum.time_evolve_batch(parameters, times, initial_states)
        state = states[0]

        start = time.time()
        for step in tqdm(range(self.config.steps), desc="Entropic optimisation"):
            results = self.optimization_step(state, partition, target_gradient, weights)

            if (step + 1) % self.config.log_interval == 0:
                self._update_history(results)

            if (step + 1) % self.config.checkpoint_interval == 0:
                self.save_checkpoint(step + 1, state, results)
                self.save_training_curves()

        end = time.time()

        final_results = {
            "final_state": state.detach(),
            "final_metric": self.geometry.metric.detach(),
            "training_time": end - start,
            "history": self.history,
        }

        self.save_results(final_results)
        return final_results

    # ------------------------------------------------------------------
    # Logging utilities
    # ------------------------------------------------------------------
    def _update_history(self, results: Dict[str, torch.Tensor]) -> None:
        for key in self.history:
            tensor = results.get(key)
            if tensor is not None:
                self.history[key].append(float(tensor.detach().cpu()))

    def save_checkpoint(self, step: int, state: torch.Tensor, results: Dict[str, torch.Tensor]) -> None:
        payload = {
            "step": step,
            "state": state.detach().cpu(),
            "metric_field": self.geometry.metric_field.detach().cpu(),
            "results": {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in results.items()},
        }
        torch.save(payload, self.results_path / f"checkpoint_{step}.pt")

    def save_training_curves(self) -> None:
        with open(self.results_path / "training_logs.json", "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)

    def save_results(self, results: Dict[str, torch.Tensor]) -> None:
        tensor_results = {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in results.items()
        }
        torch.save(tensor_results, self.results_path / "final_results.pt")
