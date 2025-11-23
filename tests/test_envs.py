"""Tests for the multi-armed bandit environment."""

import pytest

np = pytest.importorskip("numpy")

from app.envs import MultiArmedBandit


def test_bandit_rewards_are_binary():
    env = MultiArmedBandit([0.2, 0.8])
    rewards = [env.step(1) for _ in range(20)]
    assert set(rewards).issubset({0, 1})


def test_bandit_probabilities_respected():
    np.random.seed(0)
    env = MultiArmedBandit([0.1, 0.9])
    samples = [env.step(1) for _ in range(200)]
    assert np.mean(samples) > 0.7
