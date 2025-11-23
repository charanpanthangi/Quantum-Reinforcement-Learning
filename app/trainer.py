"""Training utilities for quantum and classical reinforcement learning agents.

Both agents follow the REINFORCE policy gradient idea: if an action leads to a
reward, increase the probability of taking that action again. The difference is
only in how the policy represents probabilities (quantum circuit vs. softmax).
"""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .classical_policy import ClassicalSoftmaxPolicy
from .envs import MultiArmedBandit
from .quantum_policy import QuantumPolicy


def moving_average(values: Iterable[float], window: int = 20) -> np.ndarray:
    """Compute a simple moving average for smoothing curves."""

    values = np.array(list(values), dtype=float)
    if values.size == 0:
        return values
    cumsum = np.cumsum(np.insert(values, 0, 0))
    window = max(1, window)
    smoothed = (cumsum[window:] - cumsum[:-window]) / window
    # Pad to match original length so plots align with episodes
    padding = np.ones(window - 1) * smoothed[0]
    return np.concatenate([padding, smoothed])


def train_qrl(
    env: MultiArmedBandit,
    n_episodes: int = 200,
    lr: float = 0.2,
    n_layers: int = 1,
) -> Tuple[np.ndarray, list[float]]:
    """Train a quantum policy on the bandit environment using REINFORCE.

    Args:
        env: MultiArmedBandit environment.
        n_episodes: Number of episodes to run.
        lr: Learning rate for gradient ascent.
        n_layers: Number of layers in the quantum circuit.

    Returns:
        Tuple of (trained weights, list of rewards per episode).
    """

    policy = QuantumPolicy(env.n_arms, n_layers=n_layers)
    rewards: list[float] = []

    for _ in range(n_episodes):
        action, _ = policy.select_action()
        reward = env.step(action)
        rewards.append(float(reward))

        # REINFORCE update: gradient of log probability scaled by reward.
        grad_logp = policy.grad_log_prob(action)
        policy.update(grad_logp * reward, lr)

    return policy.weights, rewards


def train_classical_rl(
    env: MultiArmedBandit,
    n_episodes: int = 200,
    lr: float = 0.2,
) -> Tuple[np.ndarray, list[float]]:
    """Train a classical softmax policy on the bandit environment."""

    policy = ClassicalSoftmaxPolicy(env.n_arms)
    rewards: list[float] = []

    for _ in range(n_episodes):
        action, _ = policy.select_action()
        reward = env.step(action)
        rewards.append(float(reward))

        grad_logp = policy.grad_log_prob(action)
        policy.update(grad_logp * reward, lr)

    return policy.preferences, rewards
