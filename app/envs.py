"""Simple reinforcement learning environments for quantum and classical agents.

This module currently provides a multi-armed bandit environment. A bandit has no
state; each action (arm) returns a reward with some fixed probability. The goal
for the agent is to discover which arm has the highest reward probability and
pull it most often. This tiny environment makes it easy to compare classical
and quantum policies without heavy computation.
"""

from __future__ import annotations

import numpy as np


class MultiArmedBandit:
    """Multi-armed bandit with Bernoulli rewards.

    Args:
        reward_probs: A list or array of probabilities, one for each arm.

    Attributes:
        n_arms: Number of available actions.
        true_reward_probs: Numpy array of reward probabilities.
    """

    def __init__(self, reward_probs: list[float] | np.ndarray) -> None:
        self.true_reward_probs = np.array(reward_probs, dtype=float)
        if np.any(self.true_reward_probs < 0) or np.any(self.true_reward_probs > 1):
            raise ValueError("Reward probabilities must be between 0 and 1.")
        self.n_arms = len(self.true_reward_probs)

    def reset(self) -> None:
        """Reset method for compatibility with RL patterns.

        A bandit has no state to reset, so this method simply exists for
        completeness and future extension.
        """

    def step(self, action: int) -> int:
        """Pull an arm and return a reward of 0 or 1.

        Args:
            action: The index of the arm to pull.

        Returns:
            Integer reward sampled from a Bernoulli distribution.
        """

        if action < 0 or action >= self.n_arms:
            raise ValueError("Action index out of range.")
        prob = self.true_reward_probs[action]
        reward = np.random.rand() < prob
        return int(reward)

    def sample_optimal_reward(self, num_samples: int = 1000) -> float:
        """Estimate the maximum achievable average reward.

        This helper repeatedly pulls the best arm to estimate its expected
        reward. It is useful for sanity checks in tests and demos.
        """

        best_arm = int(np.argmax(self.true_reward_probs))
        samples = np.random.rand(num_samples) < self.true_reward_probs[best_arm]
        return float(np.mean(samples))
