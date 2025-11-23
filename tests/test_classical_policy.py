"""Tests for the classical softmax policy."""

import pytest

np = pytest.importorskip("numpy")

from app.classical_policy import ClassicalSoftmaxPolicy


def test_softmax_probabilities_valid():
    policy = ClassicalSoftmaxPolicy(3)
    policy.preferences = np.array([0.0, 1.0, -1.0])
    probs = policy.action_probabilities()
    assert np.all(probs > 0)
    assert np.isclose(np.sum(probs), 1.0)


def test_action_sample_in_range():
    np.random.seed(2)
    policy = ClassicalSoftmaxPolicy(2)
    action, probs = policy.select_action()
    assert 0 <= action < 2
    assert probs.shape[0] == 2
