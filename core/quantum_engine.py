"""Differentiable quantum engine built on PennyLane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pennylane as qml
import torch


@dataclass
class QuantumConfig:
    num_qubits: int
    depth: int
    device: str = "default.qubit"
    interface: str = "torch"
    shots: Optional[int] = None
    seed: Optional[int] = None


class QuantumEngine:
    """Quantum circuit constructor and state-evolution helper."""

    def __init__(self, config: QuantumConfig) -> None:
        self.config = config
        self.num_qubits = config.num_qubits
        self.depth = config.depth
        self.support_tensor = 2 * self.num_qubits

        self.device = qml.device(
            config.device,
            wires=self.num_qubits,
            shots=config.shots,
            seed=config.seed,
        )

        self.ansatz = self._build_ansatz()

    # ------------------------------------------------------------------
    # Circuit construction
    # ------------------------------------------------------------------
    def _build_ansatz(self):
        num_qubits = self.num_qubits
        depth = self.depth

        @qml.qnode(self.device, interface=self.config.interface, diff_method="best")
        def circuit(weights: torch.Tensor, time: float, psi0: Optional[torch.Tensor] = None):
            if psi0 is not None:
                qml.QubitStateVector(psi0, wires=range(num_qubits))
            else:
                qml.BasisState(torch.zeros(num_qubits, dtype=torch.int64), wires=range(num_qubits))

            param_block = weights.reshape(depth, 2, num_qubits)

            for layer in range(depth):
                for wire in range(num_qubits):
                    qml.RY(param_block[layer, 0, wire], wires=wire)

                for wire in range(num_qubits - 1):
                    qml.CNOT(wires=[wire, wire + 1])

                for wire in range(num_qubits):
                    qml.RZ(param_block[layer, 1, wire] * time, wires=wire)

            return qml.state()

        return circuit

    # ------------------------------------------------------------------
    # State utilities
    # ------------------------------------------------------------------
    def evolve_state(
        self,
        parameters: torch.Tensor,
        time: float,
        initial_state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if parameters.numel() != self.depth * 2 * self.num_qubits:
            raise ValueError("Parameter tensor has incorrect shape for ansatz")

        state = self.ansatz(parameters, time, initial_state)
        return torch.as_tensor(state, dtype=torch.complex128)

    def time_evolve_batch(
        self,
        parameters: torch.Tensor,
        times: torch.Tensor,
        initial_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if initial_states is None:
            initial_states = torch.zeros(
                (len(times), 2**self.num_qubits), dtype=torch.complex128
            )
            initial_states[:, 0] = 1.0

        states = []
        for t, psi0 in zip(times, initial_states):
            states.append(self.evolve_state(parameters, float(t.item()), psi0))
        return torch.stack(states)

    def compute_state_overlap(self, state1: torch.Tensor, state2: torch.Tensor) -> torch.Tensor:
        return torch.abs(torch.dot(state1.conj(), state2)) ** 2

    def reduced_density_matrix(self, state: torch.Tensor, keep_qubits: Sequence[int]) -> torch.Tensor:
        num_qubits = self.num_qubits
        rho = torch.outer(state, state.conj())

        full_axes = list(range(num_qubits))
        trace_axes = [ax for ax in full_axes if ax not in keep_qubits]

        reshaped = rho.reshape([2] * (2 * num_qubits))
        for axis in reversed(trace_axes):
            reshaped = torch.trace(reshaped, dim1=axis, dim2=axis + num_qubits)

        final_dim = 2 ** len(keep_qubits)
        return reshaped.reshape(final_dim, final_dim)

    def random_parameters(self, scale: float = 1.0) -> torch.Tensor:
        return scale * torch.randn(self.depth, 2, self.num_qubits, dtype=torch.float64)

    def bell_state(self) -> torch.Tensor:
        psi = torch.zeros(2 ** self.num_qubits, dtype=torch.complex128)
        psi[0] = psi[-1] = 1 / np.sqrt(2)
        return psi

    def ghz_state(self) -> torch.Tensor:
        psi = torch.zeros(2 ** self.num_qubits, dtype=torch.complex128)
        psi[0] = psi[-1] = 1 / np.sqrt(2)
        return psi
