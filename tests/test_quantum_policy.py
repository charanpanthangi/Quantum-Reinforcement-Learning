"""Tests for the PennyLane quantum policy."""

import pytest

np = pytest.importorskip("numpy")
qml = pytest.importorskip("pennylane")

from app.quantum_policy import QuantumPolicy


def test_probabilities_sum_to_one():
    np.random.seed(1)
    policy = QuantumPolicy(n_arms=2, n_layers=1)
    probs = policy.action_probabilities()
    assert np.isclose(np.sum(probs), 1.0, atol=1e-6)


def test_select_action_outputs_valid_action():
    policy = QuantumPolicy(n_arms=3, n_layers=1)
    action, probs = policy.select_action()
    assert 0 <= action < 3
    assert probs.shape[0] == 3
