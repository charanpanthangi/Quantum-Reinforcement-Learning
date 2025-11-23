# Quantum Reinforcement Learning (QRL) – Quantum Policy for a Bandit

## What This Project Does
- Trains two reinforcement learning agents on a simple multi-armed bandit.
- The **quantum agent** uses a parameterized quantum circuit (PQC) built with PennyLane.
- The **classical agent** uses a softmax policy with one preference per arm.
- Both agents try to learn which arm gives the best reward probability.

## Why Quantum RL Is Interesting
- A quantum circuit can represent probability distributions over actions.
- The angles of the quantum gates are trainable, just like neural network weights.
- QRL connects quantum computing to decision-making problems in machine learning.

## Why We Use SVG Instead of PNG
> GitHub’s CODEX interface cannot preview binary image files like PNG or JPG and often shows
> “Binary files are not supported” in pull request views. To avoid this, all visualizations in
> this repository are saved as lightweight SVG (vector) images. SVGs are text-based, easy to diff,
> and render cleanly inside GitHub and CODEX.

## How the Quantum Policy Works (Plain English)
- The quantum circuit takes in a simple fixed context (no changing state in a bandit).
- Trainable rotation gates create a quantum state that encodes action probabilities.
- Measuring the qubits yields probabilities over actions; the measurement outcome is the chosen arm.
- A policy gradient update adjusts the gate angles to make rewarding actions more likely.

## Repository Structure
- `app/`: source code for environment, policies, training loop, plotting, and CLI.
- `notebooks/`: Jupyter demo notebook.
- `examples/`: saved SVG plots from sample runs.
- `tests/`: lightweight pytest checks for core components.

## How to Run
```bash
pip install -r requirements.txt
python app/main.py --episodes 300 --n-arms 2
```

How to open the notebook:
```bash
jupyter notebook notebooks/qrl_bandit_demo.ipynb
```

## What You Should See
- Rewards increasing over episodes for both agents.
- Quantum and classical learning curves on the same SVG plot.
- Final comparison bar chart of average rewards.

## Future Extensions
- Contextual bandits where the reward probabilities depend on a state.
- Small gridworld tasks with quantum policies.
- Deeper quantum circuits or more qubits.
- Running the circuit on real quantum hardware.
