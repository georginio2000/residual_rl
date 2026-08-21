from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from last_millimeter.config import load_config
from last_millimeter.envs import RemoteLiberoEnv
from last_millimeter.factory import make_components, make_environment
from last_millimeter.policies import ActionComposer, ControlMode, ObservationActionBasePolicy
from last_millimeter.representations import (
    ConcatenatedObservationEncoder,
    ObservationKeyEncoder,
)


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


class MultiTaskMockTransport(MockTransport):
    def __init__(self, task_context_dim: int) -> None:
        super().__init__()
        self.task_context_dim = task_context_dim
        self.task_id = 0

    def request(self, route: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        self.calls.append((route, dict(payload)))
        if route == "/reset":
            self.task_id = int(payload["options"]["task_id"])
        task_context = [0.0] * self.task_context_dim
        task_context[self.task_id] = 1.0
        observation = {
            "observation/state": list(range(8)),
            "base_action": [0.1] * 7,
            "task_context": task_context,
        }
        if route == "/reset":
            return {
                "observation": observation,
                "info": {
                    "success": False,
                    "task_id": self.task_id,
                    "initial_state_id": payload["options"].get("initial_state_id"),
                },
            }
        if route == "/step":
            return {
                "observation": observation,
                "reward": 0.0,
                "terminated": False,
                "truncated": False,
                "info": {"success": False, "task_id": self.task_id},
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


def test_remote_libero_applies_action_calibration_bias() -> None:
    transport = MockTransport()
    bias = [0.025, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    env = RemoteLiberoEnv(endpoint="unused", action_bias=bias, transport=transport)
    env.reset(seed=0)

    env.step(np.zeros(7))

    assert transport.calls[1] == ("/step", {"action": bias})


@pytest.mark.parametrize(
    "action_bias",
    [[0.0] * 6, [0.0] * 8, [0.0] * 6 + [float("nan")]],
)
def test_remote_libero_rejects_invalid_action_bias(action_bias: list[float]) -> None:
    with pytest.raises(ValueError, match="action_bias"):
        RemoteLiberoEnv(endpoint="unused", action_bias=action_bias, transport=MockTransport())


def test_remote_libero_config_forwards_action_bias() -> None:
    config = load_config(Path(__file__).parents[1] / "configs/libero/frozen_state.yaml")
    config.environment.options["action_bias"] = [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    environment = make_environment(config)

    np.testing.assert_array_equal(
        environment.action_bias,
        [0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )


def test_remote_libero_forwards_fixed_scene_translation_on_reset() -> None:
    transport = MockTransport()
    env = RemoteLiberoEnv(
        endpoint="unused",
        scene_translation=[0.14, 0.0, 0.0],
        transport=transport,
    )

    env.reset(seed=3, options={"tag": "test"})

    assert transport.calls[0] == (
        "/reset",
        {
            "seed": 3,
            "options": {
                "tag": "test",
                "scene_translation": [0.14, 0.0, 0.0],
            },
        },
    )


def test_remote_libero_config_forwards_scene_translation() -> None:
    config_path = (
        Path(__file__).parents[1]
        / "configs/libero/spatial_suite_scene_shift_x_0p140_frozen_state.yaml"
    )
    environment = make_environment(load_config(config_path))

    np.testing.assert_array_equal(environment.scene_translation, [0.14, 0.0, 0.0])


@pytest.mark.parametrize(
    "scene_translation",
    [
        [0.0, 0.0],
        [0.0, 0.0, 0.01],
        [float("nan"), 0.0, 0.0],
        [0.201, 0.0, 0.0],
    ],
)
def test_remote_libero_rejects_invalid_scene_translation(
    scene_translation: list[float],
) -> None:
    with pytest.raises(ValueError, match="scene_translation"):
        RemoteLiberoEnv(
            endpoint="unused",
            scene_translation=scene_translation,
            transport=MockTransport(),
        )


def test_remote_libero_rejects_scene_translation_with_action_bias() -> None:
    with pytest.raises(ValueError, match="cannot both be nonzero"):
        RemoteLiberoEnv(
            endpoint="unused",
            action_bias=[0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            scene_translation=[0.14, 0.0, 0.0],
            transport=MockTransport(),
        )


def test_remote_libero_rejects_per_reset_scene_translation_override() -> None:
    env = RemoteLiberoEnv(
        endpoint="unused",
        scene_translation=[0.14, 0.0, 0.0],
        transport=MockTransport(),
    )

    with pytest.raises(ValueError, match="fixed by environment configuration"):
        env.reset(options={"scene_translation": [0.0, 0.0, 0.0]})


def test_remote_libero_round_robin_covers_task_state_product() -> None:
    transport = MultiTaskMockTransport(task_context_dim=10)
    env = RemoteLiberoEnv(
        endpoint="unused",
        task_context_dim=10,
        task_ids=[2, 5],
        initial_state_ids=[7, 8],
        sampling="round_robin",
        transport=transport,
    )

    resets = [env.reset(seed=100 + index) for index in range(4)]

    scheduled = [call[1]["options"] for call in transport.calls if call[0] == "/reset"]
    assert scheduled == [
        {"task_id": 2, "initial_state_id": 7},
        {"task_id": 5, "initial_state_id": 7},
        {"task_id": 2, "initial_state_id": 8},
        {"task_id": 5, "initial_state_id": 8},
    ]
    assert [info["task_id"] for _, info in resets] == [2, 5, 2, 5]
    np.testing.assert_array_equal(resets[1][0]["task_context"], np.eye(10)[5])


def test_oracle_offset_cancels_remote_action_bias() -> None:
    transport = MockTransport()
    env = RemoteLiberoEnv(
        endpoint="unused",
        action_bias=[0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        transport=transport,
    )
    policy = ObservationActionBasePolicy(
        action_dim=7,
        offset=[-0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    composer = ActionComposer(
        ControlMode.FROZEN,
        action_dim=7,
        residual_scale=0.35,
        action_low=-2.0,
        action_high=2.0,
    )
    observation, _ = env.reset(seed=0)

    corrected_action = composer.compose(policy.act(observation), None).executed_action
    env.step(corrected_action)

    np.testing.assert_allclose(transport.calls[1][1]["action"], [0.1] * 7)


@pytest.mark.parametrize(
    "config_name",
    [
        "frozen_state.yaml",
        "spatial_task4_frozen_state.yaml",
        "libero10_task8_frozen_state.yaml",
        "spatial_suite_frozen_state.yaml",
        "spatial_suite_bias_x_0p150_frozen_state.yaml",
        "spatial_suite_bias_x_0p150_oracle_state.yaml",
        "spatial_suite_scene_shift_x_0p000_frozen_state.yaml",
        "spatial_suite_scene_shift_x_0p140_frozen_state.yaml",
    ],
)
def test_frozen_libero_config_builds_without_connecting(config_name: str) -> None:
    config_path = Path(__file__).parents[1] / "configs/libero" / config_name
    config = load_config(config_path)

    components = make_components(config)

    assert isinstance(components.base_policy, ObservationActionBasePolicy)
    if config_name.startswith("spatial_suite"):
        assert isinstance(components.encoder, ConcatenatedObservationEncoder)
        assert components.encoder.output_dim == 18
    else:
        assert isinstance(components.encoder, ObservationKeyEncoder)
    assert components.composer.action_dim == 7
    assert components.agent is None


def test_spatial_initial_state_split_is_disjoint_and_complete() -> None:
    split_path = (
        Path(__file__).parents[1] / "configs/libero/splits/spatial_v1.yaml"
    )
    with split_path.open(encoding="utf-8") as handle:
        split = yaml.safe_load(handle)

    partitions = [
        set(split[name]) for name in ("selection", "train", "validation", "test")
    ]
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    )
    assert set().union(*partitions) == set(range(split["states_per_task"]))
