"""Gymnasium client for an isolated LIBERO simulator process."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any, Protocol
from urllib import error, request

import gymnasium as gym
import numpy as np
from gymnasium import spaces


MAX_ABS_PLANAR_SCENE_TRANSLATION_METERS = 0.20


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
        action_bias: list[float] | tuple[float, ...] | np.ndarray | None = None,
        scene_translation: Sequence[float] | np.ndarray | None = None,
        task_context_dim: int = 0,
        task_ids: Sequence[int] | None = None,
        initial_state_ids: Sequence[int] | None = None,
        sampling: str = "round_robin",
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
        if task_context_dim < 0:
            raise ValueError("task_context_dim cannot be negative")
        if sampling not in {"round_robin", "uniform"}:
            raise ValueError("remote LIBERO sampling must be 'round_robin' or 'uniform'")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.action_low = float(action_low)
        self.action_high = float(action_high)
        self.action_dtype = np.dtype(action_dtype)
        if not np.issubdtype(self.action_dtype, np.floating):
            raise ValueError("remote LIBERO action_dtype must be floating point")
        if action_bias is None:
            self.action_bias = np.zeros(self.action_dim, dtype=self.action_dtype)
        else:
            self.action_bias = np.asarray(action_bias, dtype=self.action_dtype)
        if self.action_bias.shape != (self.action_dim,):
            raise ValueError(
                f"remote LIBERO action_bias must have shape ({self.action_dim},), "
                f"got {self.action_bias.shape}"
            )
        if not np.all(np.isfinite(self.action_bias)):
            raise ValueError("remote LIBERO action_bias must contain only finite values")
        self.action_bias = self.action_bias.copy()
        self._scene_translation_configured = scene_translation is not None
        if scene_translation is None:
            self.scene_translation = np.zeros(3, dtype=np.float64)
        else:
            self.scene_translation = np.asarray(scene_translation, dtype=np.float64)
        if self.scene_translation.shape != (3,):
            raise ValueError(
                "remote LIBERO scene_translation must have shape (3,), "
                f"got {self.scene_translation.shape}"
            )
        if not np.all(np.isfinite(self.scene_translation)):
            raise ValueError(
                "remote LIBERO scene_translation must contain only finite values"
            )
        if not np.isclose(self.scene_translation[2], 0.0, atol=1e-12):
            raise ValueError("remote LIBERO scene_translation must be planar")
        if np.any(
            np.abs(self.scene_translation[:2])
            > MAX_ABS_PLANAR_SCENE_TRANSLATION_METERS
        ):
            raise ValueError(
                "remote LIBERO planar scene_translation cannot exceed "
                f"{MAX_ABS_PLANAR_SCENE_TRANSLATION_METERS:.2f} m per axis"
            )
        if np.any(self.action_bias != 0.0) and np.any(self.scene_translation != 0.0):
            raise ValueError(
                "remote LIBERO action_bias and scene_translation cannot both be nonzero"
            )
        self.scene_translation = self.scene_translation.copy()
        self.task_context_dim = int(task_context_dim)
        self.task_ids = self._validate_ids("task_ids", task_ids)
        self.initial_state_ids = self._validate_ids("initial_state_ids", initial_state_ids)
        self.sampling = sampling
        self.transport = transport or UrllibJsonTransport(endpoint, timeout)
        self._session_active = False
        self._reset_count = 0

        self.action_space = spaces.Box(
            self.action_low,
            self.action_high,
            shape=(self.action_dim,),
            dtype=self.action_dtype,
        )
        observation_spaces: dict[str, spaces.Space[Any]] = {
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
        if self.task_context_dim:
            observation_spaces["task_context"] = spaces.Box(
                0.0,
                1.0,
                shape=(self.task_context_dim,),
                dtype=np.float32,
            )
        self.observation_space = spaces.Dict(observation_spaces)

    @staticmethod
    def _validate_ids(name: str, values: Sequence[int] | None) -> tuple[int, ...]:
        if values is None:
            return ()
        ids = tuple(int(value) for value in values)
        if not ids:
            raise ValueError(f"remote LIBERO {name} cannot be empty")
        if len(set(ids)) != len(ids):
            raise ValueError(f"remote LIBERO {name} must be unique")
        if any(value < 0 for value in ids):
            raise ValueError(f"remote LIBERO {name} cannot contain negative values")
        return ids

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
        result = {
            "observation/state": state.copy(),
            "base_action": base_action.copy(),
        }
        if self.task_context_dim:
            task_context = np.asarray(value.get("task_context"), dtype=np.float32)
            if task_context.shape != (self.task_context_dim,):
                raise ValueError(
                    "remote LIBERO task_context must have shape "
                    f"({self.task_context_dim},), got {task_context.shape}"
                )
            if not np.all(np.isfinite(task_context)):
                raise ValueError("remote LIBERO task_context must contain only finite values")
            result["task_context"] = task_context.copy()
        return result

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
        reset_options = dict(options or {})
        if "scene_translation" in reset_options:
            raise ValueError(
                "remote LIBERO scene_translation is fixed by environment configuration"
            )
        if self._scene_translation_configured:
            reset_options["scene_translation"] = self.scene_translation.tolist()
        schedule_index = self._reset_count
        if self.task_ids and "task_id" not in reset_options:
            if self.sampling == "round_robin":
                task_id = self.task_ids[schedule_index % len(self.task_ids)]
            else:
                task_id = int(self.np_random.choice(self.task_ids))
            reset_options["task_id"] = task_id
        if self.initial_state_ids and "initial_state_id" not in reset_options:
            if self.sampling == "round_robin":
                task_period = len(self.task_ids) or 1
                state_index = (schedule_index // task_period) % len(self.initial_state_ids)
                initial_state_id = self.initial_state_ids[state_index]
            else:
                initial_state_id = int(self.np_random.choice(self.initial_state_ids))
            reset_options["initial_state_id"] = initial_state_id
        response = self.transport.request(
            "/reset",
            {"seed": seed, "options": reset_options},
        )
        observation = self._observation(response.get("observation"))
        info = self._info(response.get("info", {}))
        self._session_active = True
        self._reset_count += 1
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
        executed_action = np.asarray(action, dtype=self.action_dtype) + self.action_bias
        if not np.all(np.isfinite(executed_action)):
            raise ValueError("biased remote LIBERO action must contain only finite values")
        response = self.transport.request(
            "/step",
            {"action": executed_action.tolist()},
        )
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
