"""Integration-style checks for training loops."""

import pytest

np = pytest.importorskip("numpy")
qml = pytest.importorskip("pennylane")

from app.envs import MultiArmedBandit
from app.trainer import train_classical_rl, train_qrl


def test_training_improves_average_reward():
    np.random.seed(0)
    env = MultiArmedBandit([0.1, 0.9])

    _, rewards_q = train_qrl(env, n_episodes=30, lr=0.2)
    env.reset()
    _, rewards_c = train_classical_rl(env, n_episodes=30, lr=0.2)

    first_q = np.mean(rewards_q[:10])
    last_q = np.mean(rewards_q[-10:])
    first_c = np.mean(rewards_c[:10])
    last_c = np.mean(rewards_c[-10:])

    assert last_q >= first_q
    assert last_c >= first_c
