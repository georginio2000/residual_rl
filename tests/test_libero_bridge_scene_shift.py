from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest


ROOT = Path(__file__).parents[1]


class FakeModel:
    def __init__(self) -> None:
        self.body_pos = np.asarray([[0.1, -0.2, 0.9]], dtype=np.float64)

    def body_name2id(self, name: str) -> int:
        assert name == "fixture_root"
        return 0


class FakeData:
    def __init__(self, model: FakeModel) -> None:
        self.model = model
        self.object_qpos = np.asarray([0.0, 0.2, 0.97, 1.0, 0.0, 0.0, 0.0])
        self.body_xpos = np.zeros((2, 3), dtype=np.float64)

    def get_joint_qpos(self, name: str) -> np.ndarray:
        assert name == "bowl_joint"
        return self.object_qpos

    def set_joint_qpos(self, name: str, value: np.ndarray) -> None:
        assert name == "bowl_joint"
        self.object_qpos = np.asarray(value).copy()


class FakeSim:
    def __init__(self) -> None:
        self.model = FakeModel()
        self.data = FakeData(self.model)
        self.forward()

    def forward(self) -> None:
        self.data.body_xpos[0] = self.data.object_qpos[:3]
        self.data.body_xpos[1] = self.model.body_pos[0]


class FakeLiberoEnv:
    def __init__(self) -> None:
        sim = FakeSim()
        movable = SimpleNamespace(name="bowl", joints=["bowl_joint"])
        fixture = SimpleNamespace(name="cabinet", root_body="fixture_root")
        self.env = SimpleNamespace(
            sim=sim,
            objects_dict={"bowl": movable},
            fixtures_dict={"cabinet": fixture},
            obj_body_id={"bowl": 0, "cabinet": 1},
        )
        self.sim = sim
        self.steps = 0

    def _observation(self) -> dict[str, np.ndarray]:
        return {
            "agentview_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3),
            "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.zeros(2),
        }

    def reset(self) -> dict[str, np.ndarray]:
        self.steps = 0
        return self._observation()

    def set_init_state(self, state: np.ndarray) -> dict[str, np.ndarray]:
        del state
        self.steps = 0
        self.sim.data.object_qpos = np.asarray(
            [0.0, 0.2, 0.97, 1.0, 0.0, 0.0, 0.0]
        )
        self.sim.model.body_pos[0] = [0.1, -0.2, 0.9]
        self.sim.forward()
        return self._observation()

    def get_sim_state(self) -> np.ndarray:
        return np.concatenate((self.sim.data.object_qpos, self.sim.model.body_pos[0]))

    def regenerate_obs_from_state(self, state: np.ndarray) -> dict[str, np.ndarray]:
        del state
        self.sim.forward()
        return self._observation()

    def step(self, action: list[float]) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        del action
        self.steps += 1
        return self._observation(), 1.0, self.steps >= 2, {}

    def close(self) -> None:
        pass


class FakeSuite:
    n_tasks = 1

    def __init__(self) -> None:
        self.task = SimpleNamespace(language="mock task")

    def get_task(self, task_id: int) -> object:
        assert task_id == 0
        return self.task

    def get_task_init_states(self, task_id: int) -> list[np.ndarray]:
        assert task_id == 0
        return [np.zeros(1)]


class FakePolicy:
    def infer(self, observation: dict) -> dict[str, np.ndarray]:
        del observation
        return {"actions": np.zeros((5, 7), dtype=np.float64)}


def load_bridge_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    perturbation_path = ROOT / "scripts/openpi/libero_scene_perturbation.py"
    perturbation_spec = importlib.util.spec_from_file_location(
        "libero_scene_perturbation", perturbation_path
    )
    assert perturbation_spec is not None and perturbation_spec.loader is not None
    perturbation = importlib.util.module_from_spec(perturbation_spec)
    perturbation_spec.loader.exec_module(perturbation)
    monkeypatch.setitem(sys.modules, "libero_scene_perturbation", perturbation)

    imageio = ModuleType("imageio")
    imageio.mimwrite = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "imageio", imageio)

    libero = ModuleType("libero")
    libero.__path__ = []
    libero_libero = ModuleType("libero.libero")
    libero_libero.__path__ = []
    libero_libero.benchmark = SimpleNamespace(
        get_benchmark_dict=lambda: {"libero_spatial": FakeSuite}
    )
    libero_libero.get_libero_path = lambda name: name
    libero_envs = ModuleType("libero.libero.envs")
    libero_envs.OffScreenRenderEnv = object
    monkeypatch.setitem(sys.modules, "libero", libero)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_libero)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", libero_envs)

    openpi_client = ModuleType("openpi_client")
    openpi_client.image_tools = SimpleNamespace(
        resize_with_pad=lambda image, height, width: image,
        convert_to_uint8=lambda image: image,
    )
    openpi_client.websocket_client_policy = SimpleNamespace(
        WebsocketClientPolicy=lambda host, port: FakePolicy()
    )
    monkeypatch.setitem(sys.modules, "openpi_client", openpi_client)

    bridge_path = ROOT / "scripts/openpi/libero_bridge_service.py"
    bridge_spec = importlib.util.spec_from_file_location(
        "libero_bridge_service_test", bridge_path
    )
    assert bridge_spec is not None and bridge_spec.loader is not None
    bridge = importlib.util.module_from_spec(bridge_spec)
    bridge_spec.loader.exec_module(bridge)
    bridge.make_environment = lambda task, seed: FakeLiberoEnv()
    return bridge


@pytest.mark.parametrize("shift_x", [0.0, 0.14])
def test_bridge_applies_and_records_scene_translation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shift_x: float,
) -> None:
    bridge_module = load_bridge_module(monkeypatch)
    args = SimpleNamespace(
        replan_steps=5,
        num_steps_wait=1,
        output_dir=tmp_path,
        seed=7,
        task_suite_name="libero_spatial",
        task_id=0,
        task_ids=None,
        policy_host="unused",
        policy_port=8000,
        resize_size=2,
    )
    bridge = bridge_module.LiberoOpenPIBridge(args)

    reset = bridge.reset(
        {
            "seed": 0,
            "options": {
                "task_id": 0,
                "initial_state_id": 0,
                "scene_translation": [shift_x, 0.0, 0.0],
            },
        }
    )

    assert reset["info"]["scene_translation_m"] == [shift_x, 0.0, 0.0]
    assert reset["info"]["maximum_root_translation_error_m"] == 0.0
    np.testing.assert_allclose(bridge.env.sim.data.object_qpos[:3], [shift_x, 0.2, 0.97])
    np.testing.assert_allclose(bridge.env.sim.model.body_pos[0], [0.1 + shift_x, -0.2, 0.9])

    response = bridge.step({"action": [0.0] * 7})
    assert response["terminated"]
    with (tmp_path / "bridge_result.json").open(encoding="utf-8") as handle:
        result = json.load(handle)
    assert result["openpi_commit"] == "15a9616a00943ada6c20a0f158e3adb39df2ccac"
    assert result["libero_commit"] == "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
    record = result["episodes"][0]["scene_translation"]
    assert record["translation_m"] == [shift_x, 0.0, 0.0]
    assert "final_sim_state_sha256" in record
    hashes_match = (
        record["nominal_sim_state_sha256"] == record["shifted_sim_state_sha256"]
    )
    assert hashes_match is (shift_x == 0.0)
