from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


MODULE_PATH = (
    Path(__file__).parents[1] / "scripts/openpi/libero_scene_perturbation.py"
)
SPEC = importlib.util.spec_from_file_location("libero_scene_perturbation", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
perturbation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(perturbation)


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


class FakeEnv:
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

    def get_sim_state(self) -> np.ndarray:
        return self.sim.data.object_qpos.copy()

    def regenerate_obs_from_state(self, state: np.ndarray) -> dict[str, bool]:
        del state
        self.sim.forward()
        return {"regenerated": True}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.201, -0.201])
def test_scene_translation_rejects_unsafe_values(value: float) -> None:
    with pytest.raises(ValueError, match="scene translation"):
        perturbation.validate_scene_translation(value)


@pytest.mark.parametrize(
    "translation",
    [
        [0.0, 0.0],
        [0.0, 0.0, 0.01],
        [0.0, float("nan"), 0.0],
        [0.0, -0.201, 0.0],
    ],
)
def test_bridge_scene_translation_vector_rejects_invalid_values(
    translation: list[float],
) -> None:
    with pytest.raises(ValueError, match="scene translation"):
        perturbation.validate_scene_translation_vector(translation)


def test_scene_translation_moves_objects_and_fixtures_together() -> None:
    env = FakeEnv()
    translation = perturbation.validate_scene_translation(0.05)

    observation, record = perturbation.translate_libero_scene(env, translation)

    assert observation == {"regenerated": True}
    np.testing.assert_allclose(env.sim.data.object_qpos[:3], [0.05, 0.2, 0.97])
    np.testing.assert_allclose(env.sim.model.body_pos[0], [0.15, -0.2, 0.9])
    assert record["translation_m"] == [0.05, 0.0, 0.0]
    assert record["maximum_root_translation_error_m"] <= 1e-6


@pytest.mark.parametrize(
    "translation",
    [np.asarray([0.1, 0.0]), np.asarray([0.1, 0.0, 0.01])],
)
def test_scene_translation_rejects_non_planar_vectors(
    translation: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="scene translation"):
        perturbation.translate_libero_scene(FakeEnv(), translation)
