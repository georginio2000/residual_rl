"""Gymnasium client for an isolated LIBERO simulator process."""

from __future__ import annotations

from collections.abc import Mapping
import json
from typing import Any, Protocol
from urllib import error, request

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class JsonTransport(Protocol):
    """Request/response transport used by :class:`RemoteLiberoEnv`."""

    def request(self, route: str, payload: Mapping[str, Any]) -> Mapping[str, Any]: ...


class UrllibJsonTransport:
    """Minimal dependency-free HTTP/JSON transport."""

    def __init__(self, endpoint: str, timeout: float) -> None:
        if not endpoint:
            raise ValueError("remote LIBERO endpoint cannot be empty")
        if timeout <= 0:
            raise ValueError("remote LIBERO timeout must be positive")
        self.endpoint = endpoint.rstrip("/")
        self.timeout = float(timeout)

    def request(self, route: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        body = json.dumps(dict(payload), allow_nan=False).encode("utf-8")
        http_request = request.Request(
            f"{self.endpoint}/{route.lstrip('/')}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout) as response:
                decoded = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"remote LIBERO request {route!r} failed with HTTP {exc.code}: {detail}"
            ) from exc
        except (error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"remote LIBERO request {route!r} failed: {exc}") from exc
        if not isinstance(decoded, Mapping):
            raise TypeError(f"remote LIBERO response for {route!r} must be a mapping")
        return decoded


class RemoteLiberoEnv(gym.Env[dict[str, np.ndarray], np.ndarray]):
    """Proxy a LIBERO environment without importing its legacy dependencies."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        endpoint: str,
        observation_dim: int = 8,
        action_dim: int = 7,
        action_low: float = -1.0,
        action_high: float = 1.0,
        action_dtype: str = "float64",
        timeout: float = 120.0,
        transport: JsonTransport | None = None,
    ) -> None:
        super().__init__()
        if observation_dim <= 0:
            raise ValueError("observation_dim must be positive")
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if action_low >= action_high:
            raise ValueError("action_low must be less than action_high")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.action_dtype = np.dtype(action_dtype)
        if not np.issubdtype(self.action_dtype, np.floating):
            raise ValueError("remote LIBERO action_dtype must be floating point")
        self.transport = transport or UrllibJsonTransport(endpoint, timeout)
        self._session_active = False

        self.action_space = spaces.Box(
            self.action_low,
            self.action_high,
            shape=(self.action_dim,),
            dtype=self.action_dtype,
        )
        self.observation_space = spaces.Dict(
            {
                "observation/state": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(self.observation_dim,),
                    dtype=np.float32,
                ),
                "base_action": spaces.Box(
                    self.action_low,
                    self.action_high,
                    shape=(self.action_dim,),
                    dtype=self.action_dtype,
                ),
            }
        )

    def _observation(self, value: Any) -> dict[str, np.ndarray]:
        if not isinstance(value, Mapping):
            raise TypeError("remote LIBERO observation must be a mapping")
        state = np.asarray(value.get("observation/state"), dtype=np.float32)
        base_action = np.asarray(value.get("base_action"), dtype=self.action_dtype)
        if state.shape != (self.observation_dim,):
            raise ValueError(
                f"remote LIBERO state must have shape ({self.observation_dim},), "
                f"got {state.shape}"
            )
        if base_action.shape != (self.action_dim,):
            raise ValueError(
                f"remote LIBERO base action must have shape ({self.action_dim},), "
                f"got {base_action.shape}"
            )
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(base_action)):
            raise ValueError("remote LIBERO observation must contain only finite values")
        return {
            "observation/state": state.copy(),
            "base_action": base_action.copy(),
        }

    @staticmethod
    def _info(value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise TypeError("remote LIBERO info must be a mapping")
        return dict(value)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        response = self.transport.request(
            "/reset",
            {"seed": seed, "options": options or {}},
        )
        observation = self._observation(response.get("observation"))
        info = self._info(response.get("info", {}))
        self._session_active = True
        return observation, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if not self._session_active:
            raise RuntimeError("remote LIBERO environment must be reset before stepping")
        action = np.asarray(action)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"remote LIBERO action must have shape ({self.action_dim},), got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("remote LIBERO action must contain only finite values")
        response = self.transport.request("/step", {"action": action.tolist()})
        observation = self._observation(response.get("observation"))
        info = self._info(response.get("info", {}))
        return (
            observation,
            float(response.get("reward", 0.0)),
            bool(response.get("terminated", False)),
            bool(response.get("truncated", False)),
            info,
        )

    def close(self) -> None:
        if not self._session_active:
            return
        try:
            self.transport.request("/close", {})
        except (OSError, RuntimeError):
            pass
        finally:
            self._session_active = False
