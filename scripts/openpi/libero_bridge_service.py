"""Serve one LIBERO task and frozen OpenPI actions over HTTP/JSON.

Run this service inside the isolated OpenPI LIBERO client image. The main
residual_rl environment receives only simulator state and a frozen base action;
camera observations and legacy LIBERO dependencies remain in this process.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import logging
import math
from pathlib import Path
import traceback
from typing import Any

import imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
LIBERO_ACTION_DIM = 7
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
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8765)
    parser.add_argument("--policy-host", default="host.docker.internal")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument(
        "--task-suite-name",
        choices=sorted(MAX_STEPS),
        default="libero_spatial",
    )
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/libero_bridge"))
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


class LiberoOpenPIBridge:
    """Own the simulator, frozen client, action cache, and episode artifacts."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.replan_steps <= 0:
            raise ValueError("replan_steps must be positive")
        if args.num_steps_wait < 0:
            raise ValueError("num_steps_wait cannot be negative")
        self.args = args
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        np.random.seed(args.seed)
        self.task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
        if not 0 <= args.task_id < self.task_suite.n_tasks:
            raise ValueError(f"task_id must be in [0, {self.task_suite.n_tasks})")
        self.task = self.task_suite.get_task(args.task_id)
        self.initial_states = self.task_suite.get_task_init_states(args.task_id)
        self.policy = websocket_client_policy.WebsocketClientPolicy(
            args.policy_host,
            args.policy_port,
        )
        self.env = make_environment(self.task, args.seed)
        self.action_plan = collections.deque()
        self.observation = None
        self.active = False
        self.closed = False
        self.episode_index = 0
        self.initial_state_id = 0
        self.control_steps = 0
        self.inference_requests = 0
        self.episode_return = 0.0
        self.replay_images = []
        self.episode_results = []

    def _state(self, observation: dict) -> np.ndarray:
        return np.concatenate(
            (
                observation["robot0_eef_pos"],
                quat_to_axis_angle(observation["robot0_eef_quat"]),
                observation["robot0_gripper_qpos"],
            )
        )

    def _model_observation(self, observation: dict) -> dict:
        image = np.ascontiguousarray(observation["agentview_image"][::-1, ::-1])
        wrist_image = np.ascontiguousarray(
            observation["robot0_eye_in_hand_image"][::-1, ::-1]
        )
        image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                image,
                self.args.resize_size,
                self.args.resize_size,
            )
        )
        wrist_image = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                wrist_image,
                self.args.resize_size,
                self.args.resize_size,
            )
        )
        self.replay_images.append(image)
        return {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": self._state(observation),
            "prompt": str(self.task.language),
        }

    def _next_base_action(self, model_observation: dict) -> np.ndarray:
        if not self.action_plan:
            update_observation_digest(self.observation_digest, model_observation)
            response = self.policy.infer(model_observation)
            self.inference_requests += 1
            if "actions" not in response:
                raise KeyError("OpenPI response is missing 'actions'")
            raw_actions = np.asarray(response["actions"])
            if raw_actions.ndim != 2 or raw_actions.shape[1] != LIBERO_ACTION_DIM:
                raise ValueError(
                    "OpenPI actions must have shape "
                    f"(chunk_length, {LIBERO_ACTION_DIM}), got {raw_actions.shape}"
                )
            update_array_digest(self.action_digest, "actions", raw_actions)
            if self.first_action is None:
                self.first_action = raw_actions[0].tolist()
                self.action_dtype = str(raw_actions.dtype)
            if len(raw_actions) < self.args.replan_steps:
                raise ValueError(
                    f"OpenPI returned {len(raw_actions)} actions, fewer than "
                    f"replan_steps={self.args.replan_steps}"
                )
            if not np.all(np.isfinite(raw_actions)):
                raise ValueError("OpenPI actions must contain only finite values")
            self.action_plan.extend(
                action.copy() for action in raw_actions[: self.args.replan_steps]
            )
        return self.action_plan.popleft()

    def _payload(self, observation: dict, *, terminal: bool = False) -> dict:
        if terminal:
            base_action = np.zeros(LIBERO_ACTION_DIM, dtype=np.float64)
        else:
            model_observation = self._model_observation(observation)
            base_action = self._next_base_action(model_observation)
        return {
            "observation/state": self._state(observation).tolist(),
            "base_action": base_action.tolist(),
        }

    def reset(self, payload: dict) -> dict:
        if self.closed:
            raise RuntimeError("bridge is closed")
        if self.active:
            self._finalize(success=False, reason="reset_before_terminal")
        seed = payload.get("seed")
        options = payload.get("options") or {}
        if not isinstance(options, dict):
            raise TypeError("reset options must be a mapping")
        if "initial_state_id" in options:
            initial_state_id = int(options["initial_state_id"])
        elif seed is None:
            initial_state_id = self.episode_index % len(self.initial_states)
        else:
            initial_state_id = int(seed) % len(self.initial_states)
        if not 0 <= initial_state_id < len(self.initial_states):
            raise ValueError(
                f"initial_state_id must be in [0, {len(self.initial_states)})"
            )

        self.env.reset()
        observation = self.env.set_init_state(self.initial_states[initial_state_id])
        for _ in range(self.args.num_steps_wait):
            observation, _, done, _ = self.env.step(LIBERO_DUMMY_ACTION)
            if done:
                raise RuntimeError("LIBERO task terminated during stabilization steps")

        self.observation = observation
        self.action_plan.clear()
        self.active = True
        self.initial_state_id = initial_state_id
        self.control_steps = 0
        self.inference_requests = 0
        self.episode_return = 0.0
        self.replay_images = []
        self.observation_digest = hashlib.sha256()
        self.action_digest = hashlib.sha256()
        self.first_action = None
        self.action_dtype = None
        return {
            "observation": self._payload(observation),
            "info": {
                "success": False,
                "task_suite": self.args.task_suite_name,
                "task_id": self.args.task_id,
                "task_description": str(self.task.language),
                "initial_state_id": initial_state_id,
                "stabilization_steps": self.args.num_steps_wait,
            },
        }

    def step(self, payload: dict) -> dict:
        if not self.active or self.observation is None:
            raise RuntimeError("bridge must be reset before stepping")
        action = np.asarray(payload.get("action"))
        if action.shape != (LIBERO_ACTION_DIM,):
            raise ValueError(
                f"action must have shape ({LIBERO_ACTION_DIM},), got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain only finite values")

        observation, reward, done, _ = self.env.step(action.tolist())
        self.observation = observation
        self.control_steps += 1
        self.episode_return += float(reward)
        terminated = bool(done)
        truncated = bool(
            self.control_steps >= MAX_STEPS[self.args.task_suite_name] and not terminated
        )
        response = {
            "observation": self._payload(
                observation,
                terminal=terminated or truncated,
            ),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": {
                "success": terminated,
                "initial_state_id": self.initial_state_id,
                "control_steps": self.control_steps,
                "total_steps": self.control_steps + self.args.num_steps_wait,
                "inference_requests": self.inference_requests,
            },
        }
        if terminated or truncated:
            self._finalize(
                success=terminated,
                reason="success" if terminated else "time_limit",
            )
        return response

    def _finalize(self, *, success: bool, reason: str) -> None:
        if not self.active:
            return
        suffix = "success" if success else "failure"
        video_name = (
            f"task_{self.args.task_id}_episode_{self.episode_index}_{suffix}.mp4"
        )
        video_path = self.args.output_dir / video_name
        if self.replay_images:
            imageio.mimwrite(video_path, self.replay_images, fps=10)
        result = {
            "episode": self.episode_index,
            "initial_state_id": self.initial_state_id,
            "success": bool(success),
            "reason": reason,
            "control_steps": self.control_steps,
            "total_steps": self.control_steps + self.args.num_steps_wait,
            "inference_requests": self.inference_requests,
            "return": self.episode_return,
            "observation_trace_sha256": self.observation_digest.hexdigest(),
            "action_trace_sha256": self.action_digest.hexdigest(),
            "action_dtype": self.action_dtype,
            "first_action": self.first_action,
            "video": str(video_path) if self.replay_images else None,
        }
        self.episode_results.append(result)
        self.episode_index += 1
        self.active = False
        self._write_results()
        logging.info("episode result: %s", result)

    def _write_results(self) -> None:
        successes = sum(int(result["success"]) for result in self.episode_results)
        result = {
            "policy": "frozen pi05_libero",
            "batch_size": 1,
            "representation": "LIBERO environment state",
            "task_suite": self.args.task_suite_name,
            "task_id": self.args.task_id,
            "task_description": str(self.task.language),
            "episodes": self.episode_results,
            "trials": len(self.episode_results),
            "successes": successes,
            "success_rate": successes / len(self.episode_results),
        }
        result_path = self.args.output_dir / "bridge_result.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def status(self) -> dict:
        return {
            "ready": not self.closed,
            "active": self.active,
            "task_suite": self.args.task_suite_name,
            "task_id": self.args.task_id,
            "task_description": str(self.task.language),
            "completed_episodes": len(self.episode_results),
        }

    def close(self) -> dict:
        if self.closed:
            return {"closed": True}
        if self.active:
            self._finalize(success=False, reason="client_closed")
        self.env.close()
        self.closed = True
        return {"closed": True}


class BridgeRequestHandler(BaseHTTPRequestHandler):
    bridge = None

    def _write_json(self, status: HTTPStatus, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        if not isinstance(payload, dict):
            raise TypeError("request body must be a JSON object")
        return payload

    def do_GET(self) -> None:
        if self.path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "unknown route"})
            return
        self._write_json(HTTPStatus.OK, self.bridge.status())

    def do_POST(self) -> None:
        routes = {
            "/reset": self.bridge.reset,
            "/step": self.bridge.step,
            "/close": lambda _: self.bridge.close(),
        }
        handler = routes.get(self.path)
        if handler is None:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "unknown route"})
            return
        try:
            payload = self._read_json()
            response = handler(payload)
            self._write_json(HTTPStatus.OK, response)
        except Exception as exc:
            logging.error("bridge request failed:\n%s", traceback.format_exc())
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": type(exc).__name__, "message": str(exc)},
            )

    def log_message(self, format: str, *args: object) -> None:
        logging.info("http: " + format, *args)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    bridge = LiberoOpenPIBridge(args)
    BridgeRequestHandler.bridge = bridge
    server = HTTPServer(
        (args.listen_host, args.listen_port),
        BridgeRequestHandler,
    )
    logging.info(
        "LIBERO bridge listening on %s:%d for %s task %d",
        args.listen_host,
        args.listen_port,
        args.task_suite_name,
        args.task_id,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        bridge.close()


if __name__ == "__main__":
    main()
