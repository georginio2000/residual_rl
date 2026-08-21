"""Interfaces and baseline implementations for frozen generalist policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np


class BasePolicy(ABC):
    """Black-box action interface implemented by the future OpenPI adapter."""

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Number of continuous action dimensions."""

    @abstractmethod
    def act(self, observation: Any) -> np.ndarray:
        """Return a frozen reference action for one observation."""

    def reset(self) -> None:
        """Reset episode-local policy state, such as a cached action chunk."""

    def close(self) -> None:
        """Release optional policy resources."""


class ProportionalBasePolicy(BasePolicy):
    """Useful but systematically imprecise controller for the toy environment."""

    def __init__(self, gain: float = 5.0, bias: tuple[float, float] = (0.18, -0.12)) -> None:
        self.gain = float(gain)
        self.bias = np.asarray(bias, dtype=np.float32)
        if self.bias.shape != (2,):
            raise ValueError("bias must have shape (2,)")

    @property
    def action_dim(self) -> int:
        return 2

    def act(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (4,):
            raise ValueError("toy base policy expects observation shape (4,)")
        position, goal = observation[:2], observation[2:]
        action = self.gain * (goal - position) + self.bias
        return np.clip(action, -1.0, 1.0).astype(np.float32)


class ObservationActionBasePolicy(BasePolicy):
    """Read a frozen base action supplied by an environment observation."""

    def __init__(
        self,
        action_dim: int,
        key: str = "base_action",
        offset: list[float] | tuple[float, ...] | np.ndarray | None = None,
    ) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not key:
            raise ValueError("base-action observation key cannot be empty")
        self._action_dim = int(action_dim)
        self.key = key
        if offset is None:
            self.offset = np.zeros(self.action_dim, dtype=np.float64)
        else:
            self.offset = np.asarray(offset, dtype=np.float64)
        if self.offset.shape != (self.action_dim,):
            raise ValueError(
                f"base-action offset must have shape ({self.action_dim},), "
                f"got {self.offset.shape}"
            )
        if not np.all(np.isfinite(self.offset)):
            raise ValueError("base-action offset must contain only finite values")
        self.offset = self.offset.copy()

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        if not isinstance(observation, Mapping):
            raise TypeError("observation-action policy requires a mapping observation")
        if self.key not in observation:
            raise KeyError(f"observation is missing base-action key {self.key!r}")
        action = np.asarray(observation[self.key])
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"expected {self.key!r} shape ({self.action_dim},), got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("base action must contain only finite values")
        return action + self.offset.astype(action.dtype, copy=False)
