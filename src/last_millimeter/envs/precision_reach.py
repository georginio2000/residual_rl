"""Small continuous-control task for validating the residual-RL pipeline."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PrecisionReachEnv(gym.Env[np.ndarray, np.ndarray]):
    """Move a point to a randomly sampled target with high precision.

    Observations are ``[position_x, position_y, goal_x, goal_y]``. The frozen
    base controller reaches the correct neighborhood but has a systematic
    action bias that prevents reliable convergence inside a tight tolerance.
    This mirrors the intended research setting: useful global behavior with a
    correctable precision-stage error.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        *,
        max_episode_steps: int = 60,
        step_scale: float = 0.1,
        success_tolerance: float = 0.035,
        success_bonus: float = 5.0,
        action_penalty: float = 0.01,
        process_noise_std: float = 0.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("render_mode must be None or 'ansi'")

        self.max_episode_steps = int(max_episode_steps)
        self.step_scale = float(step_scale)
        self.success_tolerance = float(success_tolerance)
        self.success_bonus = float(success_bonus)
        self.action_penalty = float(action_penalty)
        self.process_noise_std = float(process_noise_std)
        self.render_mode = render_mode

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.position = np.zeros(2, dtype=np.float32)
        self.goal = np.zeros(2, dtype=np.float32)
        self.steps = 0

    def _observation(self) -> np.ndarray:
        return np.concatenate((self.position, self.goal)).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        if "position" in options and "goal" in options:
            position = np.asarray(options["position"], dtype=np.float32)
            goal = np.asarray(options["goal"], dtype=np.float32)
        else:
            position, goal = self._sample_problem()

        if position.shape != (2,) or goal.shape != (2,):
            raise ValueError("position and goal reset options must have shape (2,)")

        self.position = np.clip(position, -1.0, 1.0)
        self.goal = np.clip(goal, -1.0, 1.0)
        self.steps = 0
        distance = float(np.linalg.norm(self.goal - self.position))
        return self._observation(), {"distance": distance, "success": False}

    def _sample_problem(self) -> tuple[np.ndarray, np.ndarray]:
        for _ in range(100):
            position = self.np_random.uniform(-0.8, 0.8, size=2).astype(np.float32)
            goal = self.np_random.uniform(-0.8, 0.8, size=2).astype(np.float32)
            if np.linalg.norm(goal - position) >= 0.5:
                return position, goal
        raise RuntimeError("failed to sample sufficiently separated position and goal")

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != self.action_space.shape:
            raise ValueError(f"action must have shape {self.action_space.shape}, got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)

        noise = self.np_random.normal(0.0, self.process_noise_std, size=2)
        displacement = self.step_scale * action + noise
        self.position = np.clip(self.position + displacement, -1.0, 1.0).astype(np.float32)
        self.steps += 1

        distance = float(np.linalg.norm(self.goal - self.position))
        success = distance <= self.success_tolerance
        terminated = bool(success)
        truncated = bool(self.steps >= self.max_episode_steps and not terminated)
        reward = -distance - self.action_penalty * float(np.square(action).sum())
        if success:
            reward += self.success_bonus

        info = {
            "distance": distance,
            "success": success,
            "task_reward": reward,
        }
        return self._observation(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        if self.render_mode != "ansi":
            return None
        return (
            f"step={self.steps} position={self.position.round(3).tolist()} "
            f"goal={self.goal.round(3).tolist()} "
            f"distance={np.linalg.norm(self.goal - self.position):.4f}"
        )

