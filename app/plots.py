"""Plot utilities that save SVG figures for the project.

SVG output keeps files lightweight and text-based, which avoids GitHub preview
issues with binary images. Each helper produces a simple visual to compare how
quantum and classical agents learn on the bandit environment.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .trainer import moving_average


# Ensure inline backend prefers SVG when used in notebooks
plt.rcParams["figure.dpi"] = 120


def plot_episode_rewards(
    rewards_q: list[float],
    rewards_c: list[float],
    output_path: str = "examples/qrl_episode_rewards.svg",
    window: int = 20,
) -> None:
    """Line plot of smoothed episode rewards for both agents."""

    plt.figure(figsize=(6, 4))
    episodes = np.arange(len(rewards_q))
    plt.plot(episodes, moving_average(rewards_q, window), label="Quantum policy")
    plt.plot(episodes, moving_average(rewards_c, window), label="Classical softmax")
    plt.xlabel("Episode")
    plt.ylabel("Moving average reward")
    plt.title("Learning curves on the bandit")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_final_performance(
    rewards_q: list[float],
    rewards_c: list[float],
    output_path: str = "examples/qrl_vs_classical_performance.svg",
    tail: int = 50,
) -> None:
    """Bar chart comparing final average rewards."""

    avg_q = float(np.mean(rewards_q[-tail:])) if rewards_q else 0.0
    avg_c = float(np.mean(rewards_c[-tail:])) if rewards_c else 0.0

    plt.figure(figsize=(4, 4))
    plt.bar(["Quantum"], [avg_q], color="#4C72B0")
    plt.bar(["Classical"], [avg_c], color="#DD8452")
    plt.ylabel(f"Average reward (last {tail} episodes)")
    plt.title("Final performance")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
