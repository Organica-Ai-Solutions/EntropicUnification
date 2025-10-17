"""Entropy measurements for quantum subsystems."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch

from .quantum_engine import QuantumEngine


class EntropyModule:
    def __init__(self, quantum_engine: QuantumEngine, epsilon: float = 1e-12) -> None:
        self.quantum_engine = quantum_engine
        self.epsilon = epsilon

    def compute_density_matrix(self, state: torch.Tensor) -> torch.Tensor:
        state = state.reshape(-1, 1)
        return state @ state.conj().t()

    def partial_trace(self, state: torch.Tensor, keep_qubits: Sequence[int]) -> torch.Tensor:
        return self.quantum_engine.reduced_density_matrix(state, keep_qubits)

    def von_neumann_entropy(self, rho: torch.Tensor) -> torch.Tensor:
        eigenvals = torch.linalg.eigvalsh(rho)
        eigenvals = eigenvals.clamp(min=self.epsilon)
        return -torch.sum(eigenvals * torch.log(eigenvals))

    def compute_entanglement_entropy(self, state: torch.Tensor, partition: Sequence[int]) -> torch.Tensor:
        rho_A = self.partial_trace(state, partition)
        return self.von_neumann_entropy(rho_A)

    def entropy_gradient(self, state: torch.Tensor, partition: Sequence[int]) -> torch.Tensor:
        entropy = self.compute_entanglement_entropy(state, partition)
        grad = torch.autograd.grad(entropy, state, create_graph=True, retain_graph=True)[0]
        return grad

    def entropy_flow(self, states: torch.Tensor, partition: Sequence[int], times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        entropies = torch.stack(
            [self.compute_entanglement_entropy(state, partition) for state in states]
        )
        dS_dt = torch.gradient(entropies, spacing=(times,), edge_order=2)[0]
        return entropies, dS_dt
