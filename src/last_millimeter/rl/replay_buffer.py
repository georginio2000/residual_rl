"""Fixed-size replay buffer specialized for continuous-control transitions."""

from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        policy_output_dim: int | None = None,
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        policy_output_dim = action_dim if policy_output_dim is None else policy_output_dim
        self.state = np.empty((capacity, state_dim), dtype=np.float32)
        self.base_action = np.empty((capacity, action_dim), dtype=np.float32)
        self.policy_output = np.empty((capacity, policy_output_dim), dtype=np.float32)
        self.reward = np.empty((capacity, 1), dtype=np.float32)
        self.next_state = np.empty((capacity, state_dim), dtype=np.float32)
        self.next_base_action = np.empty((capacity, action_dim), dtype=np.float32)
        self.terminated = np.empty((capacity, 1), dtype=np.float32)
        # Defaults to 1.0 (always-applied correction), which is a no-op for
        # every mode except TRIGGERED: there, the actually-executed correction
        # is policy_output * trigger, so the critic/actor must see that
        # masked quantity rather than the actor's raw, possibly-never-applied
        # output. See ActionComposer's TRIGGERED mode and SACAgent.update.
        self.trigger = np.ones((capacity, 1), dtype=np.float32)
        self.next_trigger = np.ones((capacity, 1), dtype=np.float32)
        self._index = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        state: np.ndarray,
        base_action: np.ndarray,
        policy_output: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        next_base_action: np.ndarray,
        terminated: bool,
        trigger: float = 1.0,
        next_trigger: float = 1.0,
    ) -> None:
        index = self._index
        self.state[index] = state
        self.base_action[index] = base_action
        self.policy_output[index] = policy_output
        self.reward[index] = reward
        self.next_state[index] = next_state
        self.next_base_action[index] = next_base_action
        self.terminated[index] = float(terminated)
        self.trigger[index] = float(trigger)
        self.next_trigger[index] = float(next_trigger)
        self._index = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size > self._size:
            raise ValueError(f"cannot sample {batch_size} items from buffer of size {self._size}")
        indices = self._rng.integers(0, self._size, size=batch_size)
        return {
            "state": self.state[indices],
            "base_action": self.base_action[indices],
            "policy_output": self.policy_output[indices],
            "reward": self.reward[indices],
            "next_state": self.next_state[indices],
            "next_base_action": self.next_base_action[indices],
            "terminated": self.terminated[indices],
            "trigger": self.trigger[indices],
            "next_trigger": self.next_trigger[indices],
        }
