"""Representation boundary between environments/VLAs and reinforcement learning."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class RepresentationEncoder(ABC):
    """Encode a raw observation into the state consumed by actor and critics."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Encoded feature dimension."""

    @abstractmethod
    def encode(self, observation: np.ndarray) -> np.ndarray:
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

