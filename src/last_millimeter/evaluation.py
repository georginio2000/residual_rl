"""Shared deterministic evaluation loop and intervention measurements."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from last_millimeter.config import ProjectConfig
from last_millimeter.factory import ExperimentComponents, encode_step, make_environment
from last_millimeter.policies import ControlMode


def evaluate_components(
    config: ProjectConfig,
    components: ExperimentComponents,
    *,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    env = make_environment(config, components.backends)
    episode_values: dict[str, list[float]] = defaultdict(list)

    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        components.base_policy.reset()
        state, base_action = encode_step(observation, components)
        task_return = 0.0
        regularized_return = 0.0
        correction_norms: list[float] = []
        gates: list[float] = []
        success = False
        final_distance = float("nan")
        episode_length = 0

        while True:
            if components.composer.mode is ControlMode.FROZEN:
                policy_output = None
            else:
                assert components.agent is not None
                policy_output = components.agent.select_policy_output(
                    state, base_action, deterministic=True
                )
            intervention = components.composer.compose(base_action, policy_output)
            next_observation, reward, terminated, truncated, info = env.step(
                intervention.executed_action
            )

            penalty = (
                config.training.lambda_delta * float(np.square(intervention.correction).sum())
                + config.training.lambda_gate * intervention.gate
            )
            task_return += reward
            regularized_return += reward - penalty
            correction_norms.append(float(np.linalg.norm(intervention.correction)))
            gates.append(intervention.gate)
            episode_length += 1
            success = bool(info["success"])
            final_distance = float(info["distance"])

            if terminated or truncated:
                break
            observation = next_observation
            state, base_action = encode_step(observation, components)

        episode_values["task_return"].append(task_return)
        episode_values["regularized_return"].append(regularized_return)
        episode_values["success"].append(float(success))
        episode_values["final_distance"].append(final_distance)
        episode_values["episode_length"].append(float(episode_length))
        episode_values["correction_norm"].append(float(np.mean(correction_norms)))
        episode_values["gate"].append(float(np.mean(gates)))

    env.close()
    components.base_policy.reset()
    return {
        "episodes": float(episodes),
        "success_rate": float(np.mean(episode_values["success"])),
        "mean_task_return": float(np.mean(episode_values["task_return"])),
        "mean_regularized_return": float(np.mean(episode_values["regularized_return"])),
        "mean_final_distance": float(np.mean(episode_values["final_distance"])),
        "mean_episode_length": float(np.mean(episode_values["episode_length"])),
        "mean_correction_norm": float(np.mean(episode_values["correction_norm"])),
        "mean_gate": float(np.mean(episode_values["gate"])),
    }
