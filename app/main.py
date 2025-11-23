"""Command-line entrypoint to train quantum and classical bandit agents.

This script wires together the environment, training routines, and plotting
helpers. It is intentionally simple so newcomers can read through and run it
quickly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .envs import MultiArmedBandit
from .plots import plot_episode_rewards, plot_final_performance
from .trainer import moving_average, train_classical_rl, train_qrl


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Quantum and classical bandit demo")
    parser.add_argument("--episodes", type=int, default=300, help="Number of training episodes")
    parser.add_argument("--lr-quantum", type=float, default=0.2, help="Learning rate for quantum policy")
    parser.add_argument("--lr-classical", type=float, default=0.2, help="Learning rate for classical policy")
    parser.add_argument("--n-arms", type=int, default=2, help="Number of bandit arms")
    parser.add_argument(
        "--reward-probs",
        type=str,
        default=None,
        help="JSON list of reward probabilities per arm, e.g. '[0.1, 0.9]'. If not set, defaults are used.",
    )
    parser.add_argument("--output-dir", type=str, default="examples", help="Where to save SVG plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.reward_probs:
        reward_probs = json.loads(args.reward_probs)
    else:
        # Simple default where arm 1 is best
        reward_probs = [0.2, 0.8] if args.n_arms == 2 else list(np.linspace(0.2, 0.8, args.n_arms))

    env = MultiArmedBandit(reward_probs)
    print(f"Created bandit with reward probabilities: {env.true_reward_probs}")

    weights_q, rewards_q = train_qrl(env, n_episodes=args.episodes, lr=args.lr_quantum)
    print("Finished training quantum policy")

    # Reset is a no-op, but included for clarity
    env.reset()
    prefs_c, rewards_c = train_classical_rl(env, n_episodes=args.episodes, lr=args.lr_classical)
    print("Finished training classical policy")

    # Prepare output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_episode_rewards(rewards_q, rewards_c, output_path=str(output_dir / "qrl_episode_rewards.svg"))
    plot_final_performance(rewards_q, rewards_c, output_path=str(output_dir / "qrl_vs_classical_performance.svg"))

    avg_q = float(np.mean(moving_average(rewards_q)[-10:])) if rewards_q else 0.0
    avg_c = float(np.mean(moving_average(rewards_c)[-10:])) if rewards_c else 0.0
    best_arm = int(np.argmax(env.true_reward_probs))

    print("Summary:\n--------")
    print(f"Optimal arm index: {best_arm} with reward probability {env.true_reward_probs[best_arm]:.2f}")
    print(f"Quantum policy average (last episodes): {avg_q:.3f}")
    print(f"Classical policy average (last episodes): {avg_c:.3f}")
    print(f"SVG plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
