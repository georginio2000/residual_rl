"""Interfaces and baseline implementations for frozen generalist policies."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
