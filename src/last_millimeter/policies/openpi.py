"""Dependency-free boundary for a frozen remote OpenPI policy server."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np

from last_millimeter.policies.base import BasePolicy


class InferenceClient(Protocol):
    """Small subset implemented by OpenPI's WebsocketClientPolicy."""

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...


class OpenPIClientBasePolicy(BasePolicy):
    """Consume action chunks from a frozen OpenPI inference client.

    The adapter owns no trainable model and sends exactly one observation per
    inference request. The official OpenPI server remains in its own Python
    environment; only an object implementing :class:`InferenceClient` crosses
    this boundary.
    """

    def __init__(self, client: InferenceClient, *, action_dim: int, replan_steps: int = 5) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        self.client = client
        self._action_dim = int(action_dim)
        self.replan_steps = int(replan_steps)
        self._action_plan: deque[np.ndarray] = deque()

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def reset(self) -> None:
        self._action_plan.clear()

    def act(self, observation: Mapping[str, Any]) -> np.ndarray:
        if not isinstance(observation, Mapping):
            raise TypeError("OpenPI observations must be mappings")
        if not self._action_plan:
            response = self.client.infer(observation)
            if "actions" not in response:
                raise KeyError("OpenPI response is missing 'actions'")
            actions = np.asarray(response["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != self.action_dim:
                raise ValueError(
                    "OpenPI actions must have shape "
                    f"(chunk_length, {self.action_dim}), got {actions.shape}"
                )
            if len(actions) < self.replan_steps:
                raise ValueError(
                    f"OpenPI returned {len(actions)} actions, fewer than "
                    f"replan_steps={self.replan_steps}"
                )
            self._action_plan.extend(action.copy() for action in actions[: self.replan_steps])
        return self._action_plan.popleft()
