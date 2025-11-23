"""Quantum policy built with PennyLane for reinforcement learning.

A parameterized quantum circuit (PQC) plays the role of the policy. The circuit
produces a probability distribution over actions via measurement. The trainable
rotation angles are the policy parameters, updated by a policy gradient method.
"""

from __future__ import annotations

import numpy as np
import pennylane as qml


def create_q_policy_device(n_qubits: int) -> qml.Device:
    """Create a PennyLane device for the quantum policy.

    Args:
        n_qubits: Number of qubits used in the circuit.

    Returns:
        Configured PennyLane device.
    """

    return qml.device("default.qubit", wires=n_qubits, shots=None)


def quantum_policy_circuit(weights: np.ndarray) -> np.ndarray:
    """Quantum circuit that outputs action probabilities.

    The circuit is intentionally small so it runs quickly on CPU. It uses
    rotation gates to create a superposition, applies a simple entangling
    pattern, and measures probabilities in the computational basis.

    Args:
        weights: Array of shape (n_layers, n_qubits, 3) containing rotation
            angles for RX, RY, and RZ on each qubit.

    Returns:
        Array of probabilities over all computational basis states, which map to
        actions in the bandit environment.
    """

    n_layers, n_qubits, _ = weights.shape

    for layer in range(n_layers):
        for wire in range(n_qubits):
            qml.RX(weights[layer, wire, 0], wires=wire)
            qml.RY(weights[layer, wire, 1], wires=wire)
            qml.RZ(weights[layer, wire, 2], wires=wire)
        # Add a simple chain of CNOTs to create entanglement when n_qubits > 1
        for wire in range(n_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])

    return qml.probs(wires=range(n_qubits))


class QuantumPolicy:
    """Wrapper around a PennyLane quantum policy circuit.

    This class provides helper methods to sample actions and compute log
    probabilities needed for policy gradient updates.
    """

    def __init__(self, n_arms: int, n_layers: int = 1) -> None:
        self.n_arms = n_arms
        self.n_qubits = int(np.ceil(np.log2(n_arms))) or 1
        self.n_layers = n_layers
        self.device = create_q_policy_device(self.n_qubits)
        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(n_layers, self.n_qubits, 3))

        # Bind the qnode to this instance so that weights can be updated.
        self.qnode = qml.QNode(quantum_policy_circuit, self.device, interface="autograd")

    def action_probabilities(self, weights: np.ndarray | None = None) -> np.ndarray:
        """Compute probabilities over actions for the given weights."""

        current_weights = self.weights if weights is None else weights
        probs_full = self.qnode(current_weights)
        # There may be more computational basis states than actions. We keep the
        # first n_arms probabilities and renormalize so they sum to one.
        truncated = probs_full[: self.n_arms]
        truncated = truncated / np.sum(truncated)
        return truncated

    def select_action(self) -> tuple[int, np.ndarray]:
        """Sample an action according to the quantum probabilities.

        Returns:
            Tuple of (action index, probability vector used for sampling).
        """

        probs = self.action_probabilities()
        action = int(np.random.choice(len(probs), p=probs))
        return action, probs

    def log_prob(self, action: int, weights: np.ndarray | None = None) -> float:
        """Return log probability of taking a specific action."""

        probs = self.action_probabilities(weights)
        if action < 0 or action >= len(probs):
            raise ValueError("Action index out of range.")
        return float(np.log(probs[action] + 1e-10))

    def grad_log_prob(self, action: int) -> np.ndarray:
        """Compute gradient of log probability with respect to weights.

        PennyLane's ``grad`` function can differentiate the QNode directly. We
        apply the chain rule to obtain the gradient of log π(a|θ).
        """

        def log_prob_fn(param):
            probs = self.action_probabilities(param)
            return np.log(probs[action] + 1e-10)

        grad_fn = qml.grad(log_prob_fn)
        return grad_fn(self.weights)

    def update(self, grad: np.ndarray, lr: float) -> None:
        """Update weights with gradient ascent."""

        self.weights = self.weights + lr * grad
