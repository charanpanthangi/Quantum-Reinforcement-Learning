"""Application package for quantum reinforcement learning example."""

# Expose primary classes for convenience
from .envs import MultiArmedBandit
from .quantum_policy import QuantumPolicy, create_q_policy_device
from .classical_policy import ClassicalSoftmaxPolicy
from .trainer import train_qrl, train_classical_rl, moving_average
