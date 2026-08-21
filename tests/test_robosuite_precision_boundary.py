from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]


class FakePolicy:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def infer(self, observation: dict) -> dict[str, np.ndarray]:
        self.calls.append(observation)
        return {"actions": np.zeros((5, 7), dtype=np.float32)}


class FakeSimState:
    def __init__(self, step: int) -> None:
        self.step = step

    def flatten(self) -> np.ndarray:
        return np.asarray([self.step, 1.0], dtype=np.float64)


class FakeSim:
    def __init__(self, env: "FakeNutEnv") -> None:
        self.env = env

    def get_state(self) -> FakeSimState:
        return FakeSimState(self.env.steps)


class FakeNutEnv:
    action_spec = (-np.ones(7), np.ones(7))

    def __init__(self) -> None:
        self.steps = 0
        self.closed = False
        self.sim = FakeSim(self)

    def observation(self) -> dict[str, np.ndarray]:
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        return {
            "agentview_image": image,
            "robot0_eye_in_hand_image": image + 20,
            "robot0_eef_pos": np.asarray([0.1, 0.2, 0.3]),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.asarray([0.01, -0.01]),
        }

    def reset(self) -> dict[str, np.ndarray]:
        self.steps = 0
        return self.observation()

    def step(self, action: list[float]) -> tuple[dict, float, bool, dict]:
        assert len(action) == 7
        self.steps += 1
        return self.observation(), 0.0, False, {}

    def _check_success(self) -> bool:
        return self.steps >= 2

    def staged_rewards(self) -> tuple[float, float, float, float]:
        if self.steps == 0:
            return 0.0, 0.0, 0.0, 0.0
        if self.steps == 1:
            return 0.08, 0.35, 0.40, 0.60
        return 0.0, 0.0, 0.0, 0.0

    def close(self) -> None:
        self.closed = True


def load_module(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, FakePolicy]:
    imageio = ModuleType("imageio")
    imageio.mimwrite = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "imageio", imageio)

    robosuite = ModuleType("robosuite")
    robosuite.__version__ = "1.4.1"
    robosuite.make = lambda **kwargs: None
    controllers = ModuleType("robosuite.controllers")
    controllers.load_controller_config = lambda **kwargs: {}
    monkeypatch.setitem(sys.modules, "robosuite", robosuite)
    monkeypatch.setitem(sys.modules, "robosuite.controllers", controllers)

    policy = FakePolicy()
    openpi_client = ModuleType("openpi_client")
    openpi_client.image_tools = SimpleNamespace(
        resize_with_pad=lambda image, height, width: image,
        convert_to_uint8=lambda image: np.asarray(image, dtype=np.uint8),
    )
    openpi_client.websocket_client_policy = SimpleNamespace(
        WebsocketClientPolicy=lambda host, port: policy
    )
    monkeypatch.setitem(sys.modules, "openpi_client", openpi_client)

    path = ROOT / "scripts/openpi/eval_robosuite_precision.py"
    spec = importlib.util.spec_from_file_location("robosuite_precision_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, policy


def test_model_observation_preserves_pi05_libero_convention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = load_module(monkeypatch)
    env = FakeNutEnv()
    raw = env.observation()

    prepared = module.model_observation(raw, "place the nut", resize_size=224)

    np.testing.assert_array_equal(
        prepared["observation/image"], raw["agentview_image"][::-1, ::-1]
    )
    np.testing.assert_array_equal(
        prepared["observation/wrist_image"],
        raw["robot0_eye_in_hand_image"][::-1, ::-1],
    )
    np.testing.assert_allclose(
        prepared["observation/state"],
        [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.01, -0.01],
    )
    assert prepared["prompt"] == "place the nut"


def test_task_stage_snapshots_and_transient_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, _ = load_module(monkeypatch)
    env = FakeNutEnv()
    tracker = module.StageTracker()

    tracker.update(module.task_stage_snapshot(env, "NutAssemblySquare", {}, False))
    env.steps = 1
    tracker.update(module.task_stage_snapshot(env, "NutAssemblySquare", {}, False))
    env.steps = 2
    tracker.update(module.task_stage_snapshot(env, "NutAssemblySquare", {}, True))
    result = tracker.result()

    assert result["ever"] == {
        "grasped": True,
        "hovered": True,
        "lifted": True,
        "reached": True,
        "success": True,
    }
    assert result["maxima"]["hover_reward"] == pytest.approx(0.60)
    assert not result["final"]["grasped"]
    assert result["final"]["success"]

    tool = module.task_stage_snapshot(
        None,
        "ToolHang",
        {"frame_is_assembled": np.asarray([1.0]), "tool_on_frame": np.asarray([0.0])},
        False,
    )
    assert tool == {"frame_assembled": True, "tool_on_frame": False, "success": False}


def test_frozen_precision_evaluation_boundary_is_reproducible_and_incremental(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module, policy = load_module(monkeypatch)
    environments: list[FakeNutEnv] = []

    def make_environment(*args: object) -> FakeNutEnv:
        assert args == ("NutAssemblySquare", 4, 0, 0)
        env = FakeNutEnv()
        environments.append(env)
        return env

    monkeypatch.setattr(module, "make_environment", make_environment)
    args = argparse.Namespace(
        host="unused",
        port=8000,
        task="NutAssemblySquare",
        prompt=None,
        num_trials=2,
        seed=11,
        max_steps=4,
        num_steps_wait=0,
        resize_size=224,
        replan_steps=5,
        render_gpu_device_id=0,
        video_stride=2,
        no_video=True,
        output_dir=tmp_path,
    )

    result = module.evaluate(args)

    assert result["policy"] == "frozen pi05_libero"
    assert result["batch_size"] == 1
    assert result["robosuite_version"] == "1.4.1"
    assert result["trials"] == 2
    assert result["successes"] == 2
    assert result["stage_counts"] == {
        "reached": 2,
        "grasped": 2,
        "lifted": 2,
        "hovered": 2,
        "success": 2,
    }
    assert [episode["placement_seed"] for episode in result["episodes"]] == [11, 12]
    assert len(policy.calls) == 2
    assert environments[0].closed
    with (tmp_path / "result.json").open(encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted == result


@pytest.mark.parametrize(
    "response, message",
    [
        ({}, "missing 'actions'"),
        ({"actions": np.zeros((5, 6))}, "chunk_length, 7"),
        ({"actions": np.zeros((4, 7))}, "fewer than replan_steps"),
        ({"actions": np.full((5, 7), np.nan)}, "finite"),
    ],
)
def test_action_chunk_validation(
    monkeypatch: pytest.MonkeyPatch,
    response: dict,
    message: str,
) -> None:
    module, _ = load_module(monkeypatch)

    with pytest.raises((KeyError, ValueError), match=message):
        module._validated_action_chunk(response, replan_steps=5)
