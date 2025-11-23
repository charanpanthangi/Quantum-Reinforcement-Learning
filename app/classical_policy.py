"""Classical policy baselines for comparison with the quantum policy.

The baseline here is a simple softmax over trainable preferences. It mirrors the
quantum policy interface so that both can be trained with the same REINFORCE
loop.
"""

from __future__ import annotations

import numpy as np


class ClassicalSoftmaxPolicy:
    """Softmax policy with one learnable preference per arm."""

    def __init__(self, n_arms: int) -> None:
        self.n_arms = n_arms
        self.preferences = np.zeros(n_arms, dtype=float)

    def action_probabilities(self) -> np.ndarray:
        """Compute softmax probabilities over actions."""

        prefs = self.preferences - np.max(self.preferences)
        exp_prefs = np.exp(prefs)
        probs = exp_prefs / np.sum(exp_prefs)
        return probs

    def select_action(self) -> tuple[int, np.ndarray]:
        """Sample an action according to the softmax policy."""

        probs = self.action_probabilities()
        action = int(np.random.choice(self.n_arms, p=probs))
        return action, probs

    def log_prob(self, action: int) -> float:
        """Return log probability of the given action."""

        probs = self.action_probabilities()
        if action < 0 or action >= self.n_arms:
            raise ValueError("Action index out of range.")
        return float(np.log(probs[action] + 1e-10))

    def grad_log_prob(self, action: int) -> np.ndarray:
        """Gradient of log probability with respect to preferences.

        For a softmax distribution, the gradient is (one-hot - probs).
        """

        probs = self.action_probabilities()
        grad = -probs
        grad[action] += 1.0
        return grad

    def update(self, grad: np.ndarray, lr: float) -> None:
        """Update preferences using gradient ascent."""

        self.preferences = self.preferences + lr * grad
