"""Screen frozen OpenPI pi0.5-LIBERO on native robosuite precision tasks.

This runner belongs in the isolated Python 3.8 LIBERO client image. It keeps
robosuite, MuJoCo, and the lightweight OpenPI websocket client out of the main
``residual_rl`` environment. The remote policy checkpoint remains frozen and
each request contains exactly one observation.
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
import robosuite as suite
from robosuite.controllers import load_controller_config
from openpi_client import image_tools
from openpi_client import websocket_client_policy


OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"
LIBERO_COMMIT = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
ACTION_DIM = 7
STATE_DIM = 8
ENV_RESOLUTION = 256
DUMMY_ACTION = [0.0] * 6 + [-1.0]

TASK_SPECS = {
    "NutAssemblySquare": {
        "prompt": "place the square nut on the square peg",
        "max_steps": 600,
        "stage_names": ("reached", "grasped", "lifted", "hovered", "success"),
    },
    "ToolHang": {
        "prompt": "assemble the tool stand and hang the tool on the stand",
        "max_steps": 1000,
        "stage_names": ("frame_assembled", "tool_on_frame", "success"),
    },
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


def sim_state_sha256(env: Any) -> str:
    state = env.sim.get_state()
    if hasattr(state, "flatten"):
        state = state.flatten()
    digest = hashlib.sha256()
    update_array_digest(digest, "sim_state", np.asarray(state))
    return digest.hexdigest()


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(denominator, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="host.docker.internal")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", choices=sorted(TASK_SPECS), default="NutAssemblySquare")
    parser.add_argument("--prompt", help="override the task's fixed language instruction")
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--render-gpu-device-id", type=int, default=0)
    parser.add_argument("--video-stride", type=int, default=2)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/data/robosuite_precision"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("num_trials", "resize_size", "replan_steps", "video_stride"):
        if int(getattr(args, name)) <= 0:
            raise ValueError("{} must be positive".format(name))
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if args.num_steps_wait < 0:
        raise ValueError("num_steps_wait cannot be negative")
    if args.render_gpu_device_id < 0:
        raise ValueError("render_gpu_device_id cannot be negative")


def make_environment(
    task_name: str,
    max_steps: int,
    num_steps_wait: int,
    render_gpu_device_id: int,
) -> Any:
    controller = load_controller_config(default_controller="OSC_POSE")
    return suite.make(
        env_name=task_name,
        robots="Panda",
        controller_configs=controller,
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        use_object_obs=True,
        reward_shaping=False,
        camera_names=["agentview", "robot0_eye_in_hand"],
        camera_heights=ENV_RESOLUTION,
        camera_widths=ENV_RESOLUTION,
        camera_depths=False,
        render_gpu_device_id=render_gpu_device_id,
        control_freq=20,
        horizon=max_steps + num_steps_wait,
        ignore_done=False,
        hard_reset=True,
    )


def model_observation(
    observation: dict,
    prompt: str,
    resize_size: int,
) -> dict:
    # Preserve the exact pi05_libero convention used by pinned OpenPI. The
    # 180-degree rotation is part of its training/evaluation preprocessing.
    image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
    wrist_image = np.ascontiguousarray(
        observation["robot0_eye_in_hand_image"][::-1, ::-1]
    )
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(image, resize_size, resize_size)
    )
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, resize_size, resize_size)
    )
    state = np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    )
    if state.shape != (STATE_DIM,):
        raise ValueError("OpenPI state must have shape ({},), got {}".format(STATE_DIM, state.shape))
    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": prompt,
    }


def _scalar_bool(value: Any) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("stage observation must be scalar, got shape {}".format(array.shape))
    return bool(array.reshape(-1)[0])


def task_stage_snapshot(
    env: Any,
    task_name: str,
    observation: dict,
    success: bool,
) -> dict[str, Any]:
    if task_name == "NutAssemblySquare":
        reach, grasp, lift, hover = (float(value) for value in env.staged_rewards())
        return {
            "reach_reward": reach,
            "grasp_reward": grasp,
            "lift_reward": lift,
            "hover_reward": hover,
            # These thresholds denote entry into each robosuite reward stage.
            "reached": reach >= 0.05,
            "grasped": grasp > 0.0,
            "lifted": lift > 0.35 + 1e-9,
            "hovered": hover > 0.50 + 1e-9,
            "success": bool(success),
        }
    if task_name == "ToolHang":
        return {
            "frame_assembled": _scalar_bool(observation["frame_is_assembled"]),
            "tool_on_frame": _scalar_bool(observation["tool_on_frame"]),
            "success": bool(success),
        }
    raise ValueError("unsupported precision task {!r}".format(task_name))


class StageTracker:
    """Track continuous maxima, transient stage completion, and final state."""

    def __init__(self) -> None:
        self.maxima: dict[str, float] = {}
        self.ever: dict[str, bool] = {}
        self.final: dict[str, Any] = {}

    def update(self, snapshot: dict[str, Any]) -> None:
        self.final = dict(snapshot)
        for key, value in snapshot.items():
            if isinstance(value, (bool, np.bool_)):
                self.ever[key] = self.ever.get(key, False) or bool(value)
            else:
                numeric = float(value)
                self.maxima[key] = max(self.maxima.get(key, -math.inf), numeric)

    def result(self) -> dict[str, Any]:
        return {
            "maxima": dict(sorted(self.maxima.items())),
            "ever": dict(sorted(self.ever.items())),
            "final": dict(sorted(self.final.items())),
        }


def _validated_action_chunk(response: Any, replan_steps: int) -> np.ndarray:
    if not isinstance(response, dict) or "actions" not in response:
        raise KeyError("OpenPI response is missing 'actions'")
    actions = np.asarray(response["actions"])
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(
            "OpenPI actions must have shape (chunk_length, {}), got {}".format(
                ACTION_DIM, actions.shape
            )
        )
    if len(actions) < replan_steps:
        raise ValueError(
            "OpenPI returned {} actions, fewer than replan_steps={}".format(
                len(actions), replan_steps
            )
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("OpenPI actions must contain only finite values")
    return actions


def aggregate_stage_counts(episodes: list[dict[str, Any]]) -> dict[str, int]:
    stage_names = TASK_SPECS[episodes[0]["task"]]["stage_names"] if episodes else ()
    return {
        stage: sum(bool(episode["stages"]["ever"].get(stage, False)) for episode in episodes)
        for stage in stage_names
    }


def build_result(
    args: argparse.Namespace,
    prompt: str,
    max_steps: int,
    action_low: np.ndarray,
    action_high: np.ndarray,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    successes = sum(bool(episode["success"]) for episode in episodes)
    return {
        "policy": "frozen pi05_libero",
        "checkpoint": CHECKPOINT,
        "openpi_commit": OPENPI_COMMIT,
        "libero_commit": LIBERO_COMMIT,
        "robosuite_version": str(getattr(suite, "__version__", "unknown")),
        "batch_size": 1,
        "task": args.task,
        "prompt": prompt,
        "robot": "Panda",
        "controller": "OSC_POSE",
        "action_dim": ACTION_DIM,
        "action_low": np.asarray(action_low).tolist(),
        "action_high": np.asarray(action_high).tolist(),
        "state_dim": STATE_DIM,
        "camera_names": ["agentview", "robot0_eye_in_hand"],
        "environment_resolution": ENV_RESOLUTION,
        "model_resize": args.resize_size,
        "replan_steps": args.replan_steps,
        "max_steps": max_steps,
        "num_steps_wait": args.num_steps_wait,
        "seed": args.seed,
        "trials": len(episodes),
        "successes": successes,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "stage_counts": aggregate_stage_counts(episodes),
        "episodes": episodes,
    }


def write_result(path: Path, result: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    task_spec = TASK_SPECS[args.task]
    prompt = str(args.prompt or task_spec["prompt"])
    max_steps = int(args.max_steps or task_spec["max_steps"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    env = make_environment(
        args.task,
        max_steps,
        args.num_steps_wait,
        args.render_gpu_device_id,
    )
    episodes = []
    try:
        action_low, action_high = env.action_spec
        action_low = np.asarray(action_low)
        action_high = np.asarray(action_high)
        if action_low.shape != (ACTION_DIM,) or action_high.shape != (ACTION_DIM,):
            raise ValueError("robosuite task must expose a 7-D action specification")
        for episode_index in range(args.num_trials):
            placement_seed = args.seed + episode_index
            np.random.seed(placement_seed)
            observation = env.reset()
            action_plan = collections.deque()
            replay_images = []
            replay_wrist_images = []
            observation_digest = hashlib.sha256()
            action_digest = hashlib.sha256()
            stage_tracker = StageTracker()
            inference_requests = 0
            first_action = None
            action_dtype = None
            success = bool(env._check_success())
            stage_tracker.update(
                task_stage_snapshot(env, args.task, observation, success)
            )
            initial_sim_state_sha256 = sim_state_sha256(env)

            for _ in range(args.num_steps_wait):
                observation, _, _, _ = env.step(DUMMY_ACTION)
                success = bool(env._check_success())
                stage_tracker.update(
                    task_stage_snapshot(env, args.task, observation, success)
                )
                if success:
                    break
            stabilized_sim_state_sha256 = sim_state_sha256(env)

            control_steps = 0
            while not success and control_steps < max_steps:
                prepared = model_observation(observation, prompt, args.resize_size)
                if not args.no_video and control_steps % args.video_stride == 0:
                    replay_images.append(prepared["observation/image"])
                    replay_wrist_images.append(prepared["observation/wrist_image"])
                if not action_plan:
                    update_observation_digest(observation_digest, prepared)
                    actions = _validated_action_chunk(
                        policy.infer(prepared), args.replan_steps
                    )
                    update_array_digest(action_digest, "actions", actions)
                    if first_action is None:
                        first_action = actions[0].tolist()
                        action_dtype = str(actions.dtype)
                    action_plan.extend(
                        action.copy() for action in actions[: args.replan_steps]
                    )
                    inference_requests += 1

                action = np.asarray(action_plan.popleft())
                observation, _, _, _ = env.step(action.tolist())
                control_steps += 1
                success = bool(env._check_success())
                stage_tracker.update(
                    task_stage_snapshot(env, args.task, observation, success)
                )

            suffix = "success" if success else "failure"
            videos = {"agentview": None, "wrist": None}
            if not args.no_video and replay_images:
                agentview_name = "{}_episode_{:03d}_{}_agentview.mp4".format(
                    args.task, episode_index, suffix
                )
                wrist_name = "{}_episode_{:03d}_{}_wrist.mp4".format(
                    args.task, episode_index, suffix
                )
                imageio.mimwrite(
                    args.output_dir / agentview_name,
                    [np.asarray(image) for image in replay_images],
                    fps=max(1, 20 // args.video_stride),
                )
                imageio.mimwrite(
                    args.output_dir / wrist_name,
                    [np.asarray(image) for image in replay_wrist_images],
                    fps=max(1, 20 // args.video_stride),
                )
                videos = {"agentview": agentview_name, "wrist": wrist_name}
            episode = {
                "episode": episode_index,
                "task": args.task,
                "placement_seed": placement_seed,
                "success": bool(success),
                "control_steps": control_steps,
                "inference_requests": inference_requests,
                "initial_sim_state_sha256": initial_sim_state_sha256,
                "stabilized_sim_state_sha256": stabilized_sim_state_sha256,
                "final_sim_state_sha256": sim_state_sha256(env),
                "observation_trace_sha256": observation_digest.hexdigest(),
                "action_trace_sha256": action_digest.hexdigest(),
                "action_dtype": action_dtype,
                "first_action": first_action,
                "stages": stage_tracker.result(),
                "videos": videos,
            }
            episodes.append(episode)
            result = build_result(
                args, prompt, max_steps, action_low, action_high, episodes
            )
            write_result(args.output_dir / "result.json", result)
            logging.info(
                "%s episode %d seed %d: success=%s stages=%s",
                args.task,
                episode_index,
                placement_seed,
                success,
                episode["stages"]["ever"],
            )
    finally:
        env.close()

    return build_result(args, prompt, max_steps, action_low, action_high, episodes)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    result = evaluate(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
