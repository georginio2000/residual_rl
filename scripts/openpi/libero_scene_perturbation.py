"""Auditable scene-position perturbations for the pinned LIBERO client."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


MAX_ABS_PLANAR_SHIFT_METERS = 0.20


def validate_scene_translation(x_meters: float, y_meters: float = 0.0) -> np.ndarray:
    """Return a finite, bounded xyz translation in world-frame meters."""
    translation = np.asarray([x_meters, y_meters, 0.0], dtype=np.float64)
    if not np.all(np.isfinite(translation)):
        raise ValueError("scene translation must contain only finite values")
    if np.any(np.abs(translation[:2]) > MAX_ABS_PLANAR_SHIFT_METERS):
        raise ValueError(
            "planar scene translation cannot exceed "
            f"{MAX_ABS_PLANAR_SHIFT_METERS:.2f} m per axis"
        )
    if not math.isclose(float(translation[2]), 0.0):
        raise ValueError("scene translation must remain planar")
    return translation


def capture_scene_root_positions(env: Any) -> dict[str, dict[str, list[float]]]:
    """Capture world-frame roots for every movable object and fixture."""
    inner = env.env
    movable_objects = {
        name: np.asarray(inner.sim.data.body_xpos[inner.obj_body_id[name]])
        .astype(np.float64)
        .tolist()
        for name in sorted(inner.objects_dict)
    }
    fixtures = {
        name: np.asarray(inner.sim.data.body_xpos[inner.obj_body_id[name]])
        .astype(np.float64)
        .tolist()
        for name in sorted(inner.fixtures_dict)
    }
    return {"movable_objects": movable_objects, "fixtures": fixtures}


def translate_libero_scene(
    env: Any,
    translation: np.ndarray,
) -> tuple[dict, dict[str, Any]]:
    """Translate all movable objects and fixtures, preserving scene relations."""
    translation = np.asarray(translation, dtype=np.float64)
    if translation.shape != (3,):
        raise ValueError(f"scene translation must have shape (3,), got {translation.shape}")
    validated_translation = validate_scene_translation(
        float(translation[0]), float(translation[1])
    )
    if not np.array_equal(translation, validated_translation):
        raise ValueError("scene translation must be planar with zero z displacement")

    inner = env.env
    before = capture_scene_root_positions(env)
    for obj in inner.objects_dict.values():
        if not obj.joints:
            raise ValueError(f"movable object {obj.name!r} has no free joint")
        joint_name = obj.joints[-1]
        joint_qpos = np.asarray(inner.sim.data.get_joint_qpos(joint_name)).copy()
        if joint_qpos.shape != (7,):
            raise ValueError(
                f"movable object joint {joint_name!r} must have 7 qpos values, "
                f"got {joint_qpos.shape}"
            )
        joint_qpos[:3] += translation
        inner.sim.data.set_joint_qpos(joint_name, joint_qpos)

    for fixture in inner.fixtures_dict.values():
        body_id = inner.sim.model.body_name2id(fixture.root_body)
        inner.sim.model.body_pos[body_id, :3] += translation

    inner.sim.forward()
    observation = env.regenerate_obs_from_state(env.get_sim_state())
    after = capture_scene_root_positions(env)

    maximum_error = 0.0
    for category in ("movable_objects", "fixtures"):
        for name, nominal_position in before[category].items():
            expected = np.asarray(nominal_position) + translation
            achieved = np.asarray(after[category][name])
            maximum_error = max(maximum_error, float(np.max(np.abs(achieved - expected))))
    if maximum_error > 1e-6:
        raise RuntimeError(
            "LIBERO scene translation did not move every root as requested; "
            f"maximum error was {maximum_error:.3e} m"
        )

    return observation, {
        "translation_m": translation.tolist(),
        "maximum_root_translation_error_m": maximum_error,
        "nominal_root_positions_m": before,
        "shifted_root_positions_m": after,
    }
