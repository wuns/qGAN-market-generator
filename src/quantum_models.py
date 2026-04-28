"""Hybrid quantum generator: variational circuit + classical post-processing head.

The quantum part produces n_qubits expectation values from a latent vector.
A classical linear layer then maps these to the flat output space (window * n_assets).

Why a classical post-processing head:
- The output dim (window * n_assets) typically far exceeds n_qubits
- The literature (Zoufal+2019, Coyle+2021) uses similar patches/heads
- Keeps the quantum simulation tractable: state-vector size = 2^n_qubits

PennyLane 0.44 / Qiskit 2.2 compliant per project rules.
"""
from __future__ import annotations

import pennylane as qml
import torch
import torch.nn as nn


def make_quantum_node(n_qubits: int, n_layers: int):
    """Build the QNode and weight_shapes dict for a small variational circuit.

    Architecture:
      AngleEmbedding(inputs)        # encodes the latent vector into RY angles
      for each layer:
          RY(theta_l) on every qubit
          ring of CNOTs (entanglement)
      measure <Z> on every qubit

    The 'ring of CNOTs' connectivity is hardware-friendly (nearest-neighbor)
    and avoids the all-to-all entanglement that StronglyEntanglingLayers would add.

    Returns: (qnode, weight_shapes_dict).
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="best")
    def circuit(inputs, weights):
        # Encode latent noise into rotation angles. AngleEmbedding pads/truncates
        # to fit n_qubits; we'll size the latent vector to n_qubits anyway.
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

        for layer in range(n_layers):
            for w in range(n_qubits):
                qml.RY(weights[layer, w], wires=w)
            # Ring of CNOTs
            for w in range(n_qubits):
                qml.CNOT(wires=[w, (w + 1) % n_qubits])

        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    weight_shapes = {"weights": (n_layers, n_qubits)}
    return circuit, weight_shapes


class QuantumGenerator(nn.Module):
    """Hybrid quantum generator.

    Forward pass:
      latent z (batch, latent_dim=n_qubits)
        -> quantum circuit (batch, n_qubits) of <Z> expectations
        -> classical linear head (batch, window * n_assets)
        -> tanh into [-1, 1]

    The latent_dim is forced to equal n_qubits so the AngleEmbedding consumes
    exactly the noise vector. This keeps the comparison with the classical
    generator interpretable: same input dimensionality, different transformation.
    """

    def __init__(self, n_qubits: int, n_layers: int, window: int, n_assets: int = 1):
        super().__init__()
        self.n_qubits  = n_qubits
        self.n_layers  = n_layers
        self.window    = window
        self.n_assets  = n_assets
        self.latent_dim = n_qubits        # see docstring
        self.out_dim    = window * n_assets

        circuit, weight_shapes = make_quantum_node(n_qubits, n_layers)
        self.qlayer = qml.qnn.TorchLayer(circuit, weight_shapes)

        # Classical post-processing head: n_qubits -> window * n_assets
        self.head = nn.Sequential(
            nn.Linear(n_qubits, self.out_dim),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z shape: (batch, n_qubits)
        q_out = self.qlayer(z)              # (batch, n_qubits) of <Z> values in [-1, 1]
        return self.head(q_out)             # (batch, window * n_assets)


def count_quantum_parameters(qgen: QuantumGenerator) -> dict:
    """Detailed parameter count for the quantum generator."""
    n_quantum   = qgen.n_qubits * qgen.n_layers
    n_classical = sum(p.numel() for name, p in qgen.named_parameters()
                      if "qlayer" not in name)
    return {
        "quantum_params":   n_quantum,
        "classical_params": n_classical,
        "total_params":     n_quantum + n_classical,
    }
