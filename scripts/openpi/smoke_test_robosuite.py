"""Standalone sanity check: create each target env, reset, dump one frame."""

from __future__ import annotations

import sys
from pathlib import Path

import imageio
import numpy as np
import robosuite
from robosuite.controllers import load_controller_config

OUTPUT_DIR = Path("/data/smoke")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for env_name in ["NutAssemblySquare", "ToolHang"]:
    print(f"=== {env_name} ===")
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = robosuite.make(
        env_name=env_name,
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        reward_shaping=False,
        ignore_done=True,
        hard_reset=False,
    )
    np.random.seed(7)
    obs = env.reset()
    print("action_dim:", env.action_dim)
    print("keys:", sorted(obs.keys()))
    for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]:
        print(key, obs[key].shape, obs[key])
    for cam_key in ["agentview_image", "robot0_eye_in_hand_image"]:
        img = obs[cam_key]
        print(cam_key, img.shape, img.dtype)
        imageio.imwrite(OUTPUT_DIR / f"{env_name}_{cam_key}_raw.png", img)
        imageio.imwrite(OUTPUT_DIR / f"{env_name}_{cam_key}_vflip.png", img[::-1])
        imageio.imwrite(OUTPUT_DIR / f"{env_name}_{cam_key}_bothflip.png", img[::-1, ::-1])
    # step a few dummy actions and check success flag plumbing
    for _ in range(5):
        obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
    print("check_success:", env._check_success())
    env.close()

print("SMOKE_TEST_OK")
