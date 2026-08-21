"""Evaluate one task against an already-running frozen OpenPI policy server.

This is a single-task variant of OpenPI's ``examples/libero/main.py`` at
commit 15a9616a00943ada6c20a0f158e3adb39df2ccac. Run it inside the isolated
OpenPI LIBERO client image; it is not a dependency of residual_rl itself.
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
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
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
    parser.add_argument("--task-suite-name", choices=sorted(MAX_STEPS), default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--initial-state-offset", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/libero_smoke"))
    return parser.parse_args()


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / denominator


def make_environment(task: object, seed: int) -> OffScreenRenderEnv:
    task_bddl_file = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    env.seed(seed)
    return env


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    if args.num_trials <= 0:
        raise ValueError("num_trials must be positive")
    if args.replan_steps <= 0:
        raise ValueError("replan_steps must be positive")
    if args.initial_state_offset < 0:
        raise ValueError("initial_state_offset cannot be negative")

    np.random.seed(args.seed)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task_id must be in [0, {task_suite.n_tasks})")
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    initial_state_stop = args.initial_state_offset + args.num_trials
    if initial_state_stop > len(initial_states):
        raise ValueError(
            f"requested initial states [{args.initial_state_offset}, "
            f"{initial_state_stop}) but only {len(initial_states)} exist"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env = make_environment(task, args.seed)
    successes = 0
    episode_results = []
    try:
        for episode_index in range(args.num_trials):
            initial_state_id = args.initial_state_offset + episode_index
            env.reset()
            observation = env.set_init_state(initial_states[initial_state_id])
            action_plan = collections.deque()
            replay_images = []
            inference_requests = 0
            observation_digest = hashlib.sha256()
            action_digest = hashlib.sha256()
            first_action = None
            action_dtype = None
            done = False
            step = 0

            while step < MAX_STEPS[args.task_suite_name] + args.num_steps_wait:
                if step < args.num_steps_wait:
                    observation, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                    step += 1
                    continue

                image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
                wrist_image = np.ascontiguousarray(
                    observation["robot0_eye_in_hand_image"][::-1, ::-1]
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
                        "prompt": str(task.language),
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
                observation, _, done, _ = env.step(action.tolist())
                step += 1
                if done:
                    successes += 1
                    break

            suffix = "success" if done else "failure"
            video_name = (
                f"task_{args.task_id}_episode_{episode_index}_{suffix}.mp4"
            )
            video_path = args.output_dir / video_name
            imageio.mimwrite(video_path, replay_images, fps=10)
            episode_results.append(
                {
                    "episode": episode_index,
                    "initial_state_id": initial_state_id,
                    "success": bool(done),
                    "steps": step,
                    "inference_requests": inference_requests,
                    "observation_trace_sha256": observation_digest.hexdigest(),
                    "action_trace_sha256": action_digest.hexdigest(),
                    "action_dtype": action_dtype,
                    "first_action": first_action,
                    "video": str(video_path),
                }
            )
            logging.info(
                "episode=%d success=%s steps=%d requests=%d",
                episode_index,
                done,
                step,
                inference_requests,
            )
    finally:
        env.close()

    result = {
        "policy": "frozen pi05_libero",
        "batch_size": 1,
        "task_suite": args.task_suite_name,
        "task_id": args.task_id,
        "task_description": str(task.language),
        "seed": args.seed,
        "initial_state_offset": args.initial_state_offset,
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
