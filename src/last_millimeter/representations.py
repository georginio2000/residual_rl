"""Representation boundary between environments/VLAs and reinforcement learning."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import numpy as np


class RepresentationEncoder(ABC):
    """Encode a raw observation into the state consumed by actor and critics."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Encoded feature dimension."""

    @abstractmethod
    def encode(self, observation: Any) -> np.ndarray:
        """Encode a single observation without updating model parameters."""


class IdentityStateEncoder(RepresentationEncoder):
    """State-based baseline used before VLA latent integration."""

    def __init__(self, observation_dim: int) -> None:
        self._output_dim = int(observation_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, observation: np.ndarray) -> np.ndarray:
        encoded = np.asarray(observation, dtype=np.float32)
        if encoded.shape != (self.output_dim,):
            raise ValueError(
                f"expected observation shape ({self.output_dim},), got {encoded.shape}"
            )
        return encoded.copy()


class ObservationKeyEncoder(RepresentationEncoder):
    """Read a flat state feature from a mapping observation.

    This supports state-based LIBERO experiments without importing LIBERO or
    extracting VLA latents into the main project environment.
    """

    def __init__(self, key: str, output_dim: int) -> None:
        if not key:
            raise ValueError("observation key cannot be empty")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.key = key
        self._output_dim = int(output_dim)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def encode(self, observation: Mapping[str, Any]) -> np.ndarray:
        if not isinstance(observation, Mapping):
            raise TypeError("key-based representations require a mapping observation")
        if self.key not in observation:
            raise KeyError(f"observation is missing representation key {self.key!r}")
        encoded = np.asarray(observation[self.key], dtype=np.float32)
        if encoded.shape != (self.output_dim,):
            raise ValueError(
                f"expected {self.key!r} shape ({self.output_dim},), got {encoded.shape}"
            )
        return encoded.copy()
