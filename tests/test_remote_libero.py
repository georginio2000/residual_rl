from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from last_millimeter.config import load_config
from last_millimeter.envs import RemoteLiberoEnv
from last_millimeter.factory import make_components
from last_millimeter.policies import ObservationActionBasePolicy
from last_millimeter.representations import ObservationKeyEncoder


class MockTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def request(self, route: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((route, dict(payload)))
        observation = {
            "observation/state": list(range(8)),
            "base_action": [0.1] * 7,
        }
        if route == "/reset":
            return {"observation": observation, "info": {"success": False}}
        if route == "/step":
            return {
                "observation": observation,
                "reward": 1.0,
                "terminated": True,
                "truncated": False,
                "info": {"success": True, "initial_state_id": 3},
            }
        if route == "/close":
            return {"closed": True}
        raise AssertionError(f"unexpected route {route}")


def test_remote_libero_lifecycle_and_payloads() -> None:
    transport = MockTransport()
    env = RemoteLiberoEnv(endpoint="unused", transport=transport)
    env.close()
    assert transport.calls == []

    observation, info = env.reset(seed=3, options={"tag": "test"})
    assert transport.calls[0] == ("/reset", {"seed": 3, "options": {"tag": "test"}})
    assert observation["observation/state"].shape == (8,)
    assert observation["base_action"].shape == (7,)
    assert observation["base_action"].dtype == np.float64
    assert info == {"success": False}

    next_observation, reward, terminated, truncated, info = env.step(np.zeros(7))
    assert transport.calls[1] == ("/step", {"action": [0.0] * 7})
    assert next_observation["observation/state"].dtype == np.float32
    assert reward == 1.0
    assert terminated
    assert not truncated
    assert info["initial_state_id"] == 3

    env.close()
    assert transport.calls[-1] == ("/close", {})


def test_remote_libero_rejects_step_before_reset() -> None:
    env = RemoteLiberoEnv(endpoint="unused", transport=MockTransport())

    with pytest.raises(RuntimeError, match="reset before stepping"):
        env.step(np.zeros(7))


@pytest.mark.parametrize(
    "config_name",
    [
        "frozen_state.yaml",
        "spatial_task4_frozen_state.yaml",
        "libero10_task8_frozen_state.yaml",
    ],
)
def test_frozen_libero_config_builds_without_connecting(config_name: str) -> None:
    config_path = Path(__file__).parents[1] / "configs/libero" / config_name
    config = load_config(config_path)

    components = make_components(config)

    assert isinstance(components.base_policy, ObservationActionBasePolicy)
    assert isinstance(components.encoder, ObservationKeyEncoder)
    assert components.composer.action_dim == 7
    assert components.agent is None
