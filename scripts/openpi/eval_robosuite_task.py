"""Screen a frozen OpenPI checkpoint zero-shot on a robomimic/robosuite task.

NutAssemblySquare and ToolHang are native robosuite environments, not part of
the LIBERO benchmark. There is no OpenPI fine-tune for either task, so this
script probes whatever checkpoint is being served (e.g. the pinned
``pi05_libero``) entirely out of its training distribution. It mirrors the
structure of ``eval_libero_task.py`` so results are directly comparable, but
owns a robosuite environment instead of a LIBERO ``OffScreenRenderEnv``.

Run this inside the isolated robosuite client image
(``docker/openpi/robosuite.Dockerfile``); it is not a dependency of
residual_rl itself.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import robosuite
from robosuite.controllers import load_controller_config

ROBOSUITE_ACTION_DIM = 7
ROBOSUITE_DUMMY_ACTION = [0.0] * 6 + [-1.0]

TASK_INSTRUCTIONS = {
    "NutAssemblySquare": "pick up the square nut and place it on the square peg",
    "ToolHang": "insert the hook into the base and hang the wrench on the hook",
}

# Generous caps, not authoritative robomimic benchmark horizons: this is a
# zero-shot screen, so trials are given more room than a fine-tuned policy
# would need before being called a timeout.
DEFAULT_MAX_STEPS = {
    "NutAssemblySquare": 520,
    "ToolHang": 1000,
}


def update_array_digest(digest: Any, key: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(key.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def update_observation_digest(digest: Any, observation: dict) -> None:
    update_array_digest(digest, "image", observation["observation/image"])
    update_array_digest(digest, "wrist", observation["observation/wrist_image"])
    update_array_digest(digest, "state", observation["observation/state"])
    digest.update(str(observation["prompt"]).encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--env-name", choices=sorted(TASK_INSTRUCTIONS), default="NutAssemblySquare"
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="override the default language prompt for --env-name",
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument(
        "--flip-image",
        choices=["none", "vertical", "both"],
        default="vertical",
        help="orientation correction for robosuite's offscreen camera render",
    )
    parser.add_argument(
        "--action-bias-x",
        type=float,
        default=0.0,
        help="constant bias added to Cartesian x actions after frozen inference",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/robosuite_smoke"))
    return parser.parse_args()


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / denominator


def make_environment(env_name: str, seed: int) -> robosuite.environments.base.MujocoEnv:
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
    return env


def apply_flip(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ascontiguousarray(image)
    if mode == "vertical":
        return np.ascontiguousarray(image[::-1])
    return np.ascontiguousarray(image[::-1, ::-1])


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.num_trials <= 0:
        raise ValueError("num_trials must be positive")
    if args.replan_steps <= 0:
        raise ValueError("replan_steps must be positive")
    if not math.isfinite(args.action_bias_x):
        raise ValueError("action_bias_x must be finite")
    max_steps = args.max_steps or DEFAULT_MAX_STEPS[args.env_name]
    instruction = args.instruction or TASK_INSTRUCTIONS[args.env_name]
    action_bias = np.zeros(ROBOSUITE_ACTION_DIM, dtype=np.float64)
    action_bias[0] = args.action_bias_x

    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env = make_environment(args.env_name, args.seed)
    successes = 0
    episode_results = []
    try:
        for episode_index in range(args.num_trials):
            trial_seed = args.seed + episode_index
            np.random.seed(trial_seed)
            observation = env.reset()
            action_plan: collections.deque = collections.deque()
            replay_images = []
            inference_requests = 0
            observation_digest = hashlib.sha256()
            action_digest = hashlib.sha256()
            executed_action_digest = hashlib.sha256()
            first_action = None
            first_executed_action = None
            action_dtype = None
            success = False
            step = 0

            while step < max_steps + args.num_steps_wait:
                if step < args.num_steps_wait:
                    observation, _, _, _ = env.step(ROBOSUITE_DUMMY_ACTION)
                    step += 1
                    continue

                image = apply_flip(observation["agentview_image"], args.flip_image)
                wrist_image = apply_flip(
                    observation["robot0_eye_in_hand_image"], args.flip_image
                )
                image = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(image, args.resize_size, args.resize_size)
                )
                wrist_image = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        wrist_image, args.resize_size, args.resize_size
                    )
                )
                replay_images.append(image)

                if not action_plan:
                    model_observation = {
                        "observation/image": image,
                        "observation/wrist_image": wrist_image,
                        "observation/state": np.concatenate(
                            (
                                observation["robot0_eef_pos"],
                                quat_to_axis_angle(observation["robot0_eef_quat"]),
                                observation["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": instruction,
                    }
                    # One observation per request: frozen inference batch size is 1.
                    update_observation_digest(observation_digest, model_observation)
                    action_chunk = np.asarray(client.infer(model_observation)["actions"])
                    update_array_digest(action_digest, "actions", action_chunk)
                    if first_action is None:
                        first_action = action_chunk[0].tolist()
                        action_dtype = str(action_chunk.dtype)
                    inference_requests += 1
                    if len(action_chunk) < args.replan_steps:
                        raise ValueError(
                            f"policy returned {len(action_chunk)} actions, fewer than "
                            f"replan_steps={args.replan_steps}"
                        )
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                executed_action = np.asarray(action, dtype=np.float64) + action_bias
                update_array_digest(
                    executed_action_digest,
                    "executed_action",
                    executed_action,
                )
                if first_executed_action is None:
                    first_executed_action = executed_action.tolist()
                observation, _, _, _ = env.step(executed_action.tolist())
                step += 1
                success = bool(env._check_success())
                if success:
                    break

            suffix = "success" if success else "failure"
            video_name = f"{args.env_name}_episode_{episode_index}_{suffix}.mp4"
            video_path = args.output_dir / video_name
            imageio.mimwrite(video_path, replay_images, fps=10)
            episode_results.append(
                {
                    "episode": episode_index,
                    "seed": trial_seed,
                    "success": success,
                    "steps": step,
                    "inference_requests": inference_requests,
                    "observation_trace_sha256": observation_digest.hexdigest(),
                    "action_trace_sha256": action_digest.hexdigest(),
                    "executed_action_trace_sha256": executed_action_digest.hexdigest(),
                    "action_dtype": action_dtype,
                    "first_action": first_action,
                    "first_executed_action": first_executed_action,
                    "video": str(video_path),
                }
            )
            successes += int(success)
            logging.info(
                "episode=%d success=%s steps=%d requests=%d",
                episode_index,
                success,
                step,
                inference_requests,
            )
    finally:
        env.close()

    result = {
        "policy": "frozen (zero-shot, no fine-tune for this task)",
        "batch_size": 1,
        "env_name": args.env_name,
        "instruction": instruction,
        "seed": args.seed,
        "action_bias": action_bias.tolist(),
        "action_bias_x": args.action_bias_x,
        "max_steps": max_steps,
        "trials": args.num_trials,
        "successes": successes,
        "success_rate": successes / args.num_trials,
        "episodes": episode_results,
    }
    result_path = args.output_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(evaluate(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
