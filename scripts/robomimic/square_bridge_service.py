"""Serve NutAssemblySquare state and a frozen robomimic BC-RNN base action.

Run this service inside the isolated robomimic-train image (Python 3.8). The
main residual_rl environment receives only simulator state and a frozen base
action; robomimic/robosuite and the loaded checkpoint remain in this process.

Unlike the LIBERO bridge, this is single-task, low-dimensional-only (no
camera observations), and the base policy predicts one action per step
directly (no VLA-style action chunking), so the protocol is simpler: no
task-suite/task-id selection and no action-chunk replanning buffer.

Optionally computes a heuristic "critical phase" trigger from ground-truth
simulator state (gripper-to-peg distance and/or gripper speed) and reports it
as a 1-dim task_context, for use with last_millimeter's TRIGGERED control
mode. This hand-crafts the RL Token paper's critical-phase boundary instead
of asking a learned gate to discover it from a sparse episode-terminal
reward.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils

STATE_KEYS = ["object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
ACTION_DIM = 7


def update_array_digest(digest: Any, key: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(value)
    digest.update(key.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8766)
    parser.add_argument("--checkpoint", required=True, help="path to a robomimic .pth checkpoint")
    parser.add_argument("--horizon", type=int, default=None, help="override rollout horizon from config")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/square_bridge"))
    parser.add_argument(
        "--trigger",
        action="store_true",
        help="report a heuristic critical-phase trigger as a 1-dim task_context",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=0.14,
        help=(
            "gripper-to-peg distance (meters) below which the phase counts as "
            "'close'. Calibrated empirically against the frozen policy's own "
            "rollouts: distance is measured to the peg body's origin (its "
            "base, not the insertion point at its top), so it bottoms out "
            "around 0.11m even at a successful insertion rather than near 0."
        ),
    )
    parser.add_argument(
        "--speed-threshold",
        type=float,
        default=0.02,
        help="gripper displacement per control step (meters) below which the phase counts as 'slow'",
    )
    parser.add_argument(
        "--trigger-logic",
        choices=["and", "or"],
        default="and",
        help="combine the close/slow conditions with AND (both required) or OR (either)",
    )
    return parser.parse_args()


def flatten_state(obs: dict) -> np.ndarray:
    return np.concatenate([np.asarray(obs[key], dtype=np.float32).ravel() for key in STATE_KEYS])


class SquareBridge:
    """Own the robosuite env, the frozen BC-RNN policy, and episode bookkeeping."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        device = TorchUtils.get_torch_device(try_to_use_cuda=True)
        self.policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=str(args.checkpoint), device=device, verbose=True
        )
        if args.horizon is not None:
            self.horizon = int(args.horizon)
        else:
            config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)
            self.horizon = int(config.experiment.rollout.horizon)
        self.env, _ = FileUtils.env_from_checkpoint(
            ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=True
        )
        self._peg_body_id = None
        if args.trigger:
            raw_env = self.env.env
            self._peg_body_id = raw_env.sim.model.body_name2id("peg1")
            logging.info(
                "heuristic trigger enabled: distance<%.3fm %s speed<%.3fm/step",
                args.distance_threshold,
                args.trigger_logic.upper(),
                args.speed_threshold,
            )

        self.rng = np.random.default_rng(args.seed)
        self.active = False
        self.closed = False
        self.episode_index = 0
        self.control_steps = 0
        self.episode_return = 0.0
        self.observation_digest = None
        self.action_digest = None
        self.executed_action_digest = None
        self.episode_results: list[dict] = []
        self._prev_eef_pos: np.ndarray | None = None
        self._trigger_active_steps = 0

    def _base_action(self, obs: dict) -> np.ndarray:
        action = np.asarray(self.policy(ob=obs), dtype=np.float64)
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"policy action must have shape ({ACTION_DIM},), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("policy action must contain only finite values")
        return action

    def _compute_trigger(self, obs: dict) -> float:
        """Heuristic "critical phase" signal from ground-truth simulator state.

        Uses the raw peg body position (not the opaque low-dim "object"
        vector) so this is legible and independently checkable. Speed is a
        per-control-step displacement proxy (finite difference of eef_pos),
        not a true velocity -- adequate for a coarse "moving slowly/
        deliberately" signal without needing an explicit dt.
        """
        eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
        peg_pos = np.asarray(self.env.env.sim.data.body_xpos[self._peg_body_id], dtype=np.float64)
        distance = float(np.linalg.norm(eef_pos - peg_pos))
        if self._prev_eef_pos is None:
            speed = 0.0
        else:
            speed = float(np.linalg.norm(eef_pos - self._prev_eef_pos))
        self._prev_eef_pos = eef_pos.copy()

        close = distance < self.args.distance_threshold
        slow = speed < self.args.speed_threshold
        triggered = (close and slow) if self.args.trigger_logic == "and" else (close or slow)
        if triggered:
            self._trigger_active_steps += 1
        return 1.0 if triggered else 0.0

    def _payload(self, obs: dict) -> dict:
        # Always compute the real base action, even on a terminal/truncated
        # step: the main project's SAC update bootstraps through truncated
        # (non-terminated) transitions using this exact "next_base_action",
        # so a placeholder here would corrupt the critic target for every
        # episode that ends by timeout rather than success.
        state = flatten_state(obs)
        update_array_digest(self.observation_digest, "state", state)
        base_action = self._base_action(obs)
        update_array_digest(self.action_digest, "action", base_action)
        payload = {
            "observation/state": state.tolist(),
            "base_action": base_action.tolist(),
        }
        if self.args.trigger:
            payload["task_context"] = [self._compute_trigger(obs)]
        return payload

    def reset(self, payload: dict) -> dict:
        if self.closed:
            raise RuntimeError("bridge is closed")
        if self.active:
            self._finalize(success=False, reason="reset_before_terminal")
        seed = payload.get("seed")
        trial_seed = int(seed) if seed is not None else int(self.rng.integers(0, 2**31 - 1))
        np.random.seed(trial_seed)

        obs = self.env.reset()
        self.policy.start_episode()
        self.active = True
        self.control_steps = 0
        self.episode_return = 0.0
        self.observation_digest = hashlib.sha256()
        self.action_digest = hashlib.sha256()
        self.executed_action_digest = hashlib.sha256()
        self._prev_eef_pos = None
        self._trigger_active_steps = 0
        return {
            "observation": self._payload(obs),
            "info": {"success": False, "trial_seed": trial_seed},
        }

    def step(self, payload: dict) -> dict:
        if not self.active:
            raise RuntimeError("bridge must be reset before stepping")
        action = np.asarray(payload.get("action"))
        if action.shape != (ACTION_DIM,):
            raise ValueError(f"action must have shape ({ACTION_DIM},), got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain only finite values")
        update_array_digest(self.executed_action_digest, "executed_action", action)

        obs, reward, done, info = self.env.step(action.tolist())
        self.control_steps += 1
        self.episode_return += float(reward)
        success = bool(self.env.is_success().get("task", False))
        terminated = success
        truncated = bool(self.control_steps >= self.horizon and not terminated)
        response = {
            "observation": self._payload(obs),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": {
                "success": success,
                "control_steps": self.control_steps,
            },
        }
        if terminated or truncated:
            self._finalize(success=success, reason="success" if success else "time_limit")
        return response

    def _finalize(self, *, success: bool, reason: str) -> None:
        if not self.active:
            return
        result = {
            "episode": self.episode_index,
            "success": bool(success),
            "reason": reason,
            "control_steps": self.control_steps,
            "return": self.episode_return,
            "observation_trace_sha256": self.observation_digest.hexdigest(),
            "action_trace_sha256": self.action_digest.hexdigest(),
            "executed_action_trace_sha256": self.executed_action_digest.hexdigest(),
        }
        if self.args.trigger:
            result["trigger_active_steps"] = self._trigger_active_steps
            result["trigger_active_fraction"] = (
                self._trigger_active_steps / self.control_steps if self.control_steps else 0.0
            )
        self.episode_results.append(result)
        self.episode_index += 1
        self.active = False
        self._write_results()
        logging.info("episode result: %s", result)

    def _write_results(self) -> None:
        successes = sum(int(result["success"]) for result in self.episode_results)
        result = {
            "policy": "frozen robomimic BC-RNN (Square PH low-dim, epoch 800)",
            "env_name": "NutAssemblySquare",
            "checkpoint": str(self.args.checkpoint),
            "episodes": self.episode_results,
            "trials": len(self.episode_results),
            "successes": successes,
            "success_rate": successes / len(self.episode_results) if self.episode_results else None,
        }
        result_path = self.args.output_dir / "bridge_result.json"
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")

    def status(self) -> dict:
        return {
            "ready": not self.closed,
            "active": self.active,
            "completed_episodes": len(self.episode_results),
        }

    def close(self) -> dict:
        if self.closed:
            return {"closed": True}
        if self.active:
            self._finalize(success=False, reason="client_closed")
        try:
            self.env.close()
        except AttributeError:
            pass
        self.closed = True
        return {"closed": True}


class BridgeRequestHandler(BaseHTTPRequestHandler):
    bridge: SquareBridge | None = None

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
        except Exception as exc:  # noqa: BLE001
            logging.error("bridge request failed:\n%s", exc, exc_info=True)
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": type(exc).__name__, "message": str(exc)},
            )

    def log_message(self, format: str, *args: object) -> None:
        logging.info("http: " + format, *args)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    bridge = SquareBridge(args)
    BridgeRequestHandler.bridge = bridge
    server = HTTPServer((args.listen_host, args.listen_port), BridgeRequestHandler)
    logging.info("Square bridge listening on %s:%d", args.listen_host, args.listen_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        bridge.close()


if __name__ == "__main__":
    main()
