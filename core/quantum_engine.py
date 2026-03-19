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
        depth = self.config.depth
        
        # Calculate number of parameters needed for the enhanced ansatz
        # 3 rotations (RY, RZ, RX) per qubit per layer
        # Plus 2 parameters for each entangling gate (CRX, CRZ)
        # Plus 1 parameter for time evolution per qubit per layer
        self.params_per_qubit = 4  # RY, RZ, RX, time evolution
        self.params_per_entangler = 2  # CRX, CRZ
        
        # Parameters: [rotations, entanglers, time_evolution]
        total_params = (self.params_per_qubit * num_qubits + 
                       self.params_per_entangler * (num_qubits - 1)) * depth
        
        @qml.qnode(self.device, interface=self.config.interface, diff_method="best")
        def circuit(weights: torch.Tensor, time: float, psi0: Optional[torch.Tensor] = None):
            if psi0 is not None:
                qml.StatePrep(psi0, wires=range(num_qubits))
            else:
                # Initialize in a more complex state than just |0...0>
                for i in range(num_qubits):
                    qml.Hadamard(wires=i)
                
                # Add some initial entanglement
                for i in range(0, num_qubits - 1, 2):
                    qml.CZ(wires=[i, i+1])
            
            # Reshape weights for the circuit
            weights = weights.reshape(depth, -1)
            param_idx = 0
            
            for layer in range(depth):
                # Single-qubit rotations
                for qubit in range(num_qubits):
                    # More expressive single-qubit rotations
                    qml.RY(weights[layer, param_idx], wires=qubit)
                    param_idx += 1
                    qml.RZ(weights[layer, param_idx], wires=qubit)
                    param_idx += 1
                    qml.RX(weights[layer, param_idx], wires=qubit)
                    param_idx += 1
                
                # Entangling layers with different connectivity
                # Nearest-neighbor entanglement
                for qubit in range(num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                    qml.CRZ(weights[layer, param_idx], wires=[qubit, qubit + 1])
                    param_idx += 1
                    qml.CRX(weights[layer, param_idx], wires=[qubit, qubit + 1])
                    param_idx += 1
                
                # Add long-range entanglement every other layer
                if layer % 2 == 0 and num_qubits > 2:
                    for offset in range(2, min(4, num_qubits)):  # Connect up to 3 qubits apart
                        for qubit in range(num_qubits - offset):
                            qml.CZ(wires=[qubit, qubit + offset])
                
                # Time evolution in the computational basis
                for qubit in range(num_qubits):
                    qml.RZ(weights[layer, param_idx] * time, wires=qubit)
                    param_idx += 1
                
                # Reset parameter index for next layer
                param_idx = 0
            
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
        # Calculate expected number of parameters
        total_params = (self.params_per_qubit * self.num_qubits + 
                       self.params_per_entangler * (self.num_qubits - 1)) * self.depth
        
        if parameters.numel() != total_params:
            raise ValueError(f"Parameter tensor has incorrect shape for ansatz. Expected {total_params} parameters, got {parameters.numel()}")

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
        """Compute reduced density matrix by tracing out all qubits not in keep_qubits.

        Uses tensor-reshape partial trace: O(2^n) instead of the O(2^{3n}) triple loop.
        Works for arbitrary qubit orderings.
        """
        n = self.num_qubits
        keep = sorted(list(keep_qubits))
        trace_out = [i for i in range(n) if i not in keep]

        if not trace_out:
            # Keeping everything — full density matrix
            return torch.outer(state, state.conj())

        # Reshape state to (2, 2, ..., 2): one axis per qubit
        state_tensor = state.reshape([2] * n)

        # Permute: kept qubits first, traced qubits last
        perm = keep + trace_out
        state_tensor = state_tensor.permute(perm).contiguous()

        # Reshape to (dim_keep, dim_trace)
        dim_keep = 2 ** len(keep)
        dim_trace = 2 ** len(trace_out)
        state_matrix = state_tensor.reshape(dim_keep, dim_trace)

        # ρ_A = Tr_B(|ψ⟩⟨ψ|) = M M†  where M = state_matrix
        return state_matrix @ state_matrix.conj().t()

    def random_parameters(self, scale: float = 1.0) -> torch.Tensor:
        """Return a flat parameter tensor compatible with evolve_state()."""
        total_params = (
            self.params_per_qubit * self.num_qubits
            + self.params_per_entangler * (self.num_qubits - 1)
        ) * self.depth
        return scale * torch.randn(total_params, dtype=torch.float64)

    def bell_state(self) -> torch.Tensor:
        """Return a maximally entangled two-qubit Bell state embedded in num_qubits space.
        For num_qubits > 2 this is (|00...0⟩ + |11...1⟩)/√2 (a GHZ-like state)."""
        psi = torch.zeros(2 ** self.num_qubits, dtype=torch.complex128)
        psi[0] = psi[-1] = 1 / np.sqrt(2)
        return psi

    def ghz_state(self) -> torch.Tensor:
        """Return the GHZ state (|00...0⟩ + |11...1⟩)/√2 for num_qubits qubits."""
        psi = torch.zeros(2 ** self.num_qubits, dtype=torch.complex128)
        psi[0] = psi[-1] = 1 / np.sqrt(2)
        return psi
