"""Command-line training entry point."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from last_millimeter.config import ProjectConfig, load_config, save_config
from last_millimeter.evaluation import evaluate_components
from last_millimeter.factory import encode_step, extract_trigger, make_components, make_environment
from last_millimeter.metrics import MetricWriter, write_json
from last_millimeter.policies import ControlMode
from last_millimeter.rl import ReplayBuffer


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluation_row(step: int, metrics: dict[str, float]) -> dict[str, float | int | str]:
    return {
        "step": step,
        "phase": "evaluation",
        "episode": "",
        "success_rate": metrics["success_rate"],
        "task_return": metrics["mean_task_return"],
        "regularized_return": metrics["mean_regularized_return"],
        "final_distance": metrics["mean_final_distance"],
        "episode_length": metrics["mean_episode_length"],
        "correction_norm": metrics["mean_correction_norm"],
        "gate": metrics["mean_gate"],
    }


def run_training(config: ProjectConfig) -> dict[str, Any]:
    config.validate()
    seed_everything(config.experiment.seed)
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, output_dir / "config.yaml")

    components = make_components(config)
    metrics_path = output_dir / "metrics.csv"

    if config.mode is ControlMode.FROZEN:
        evaluation = evaluate_components(
            config,
            components,
            episodes=config.training.evaluation_episodes,
            seed=config.experiment.seed + 10_000,
        )
        with MetricWriter(metrics_path) as writer:
            writer.write(**_evaluation_row(0, evaluation))
        summary = {
            "experiment": config.experiment.name,
            "mode": config.mode.value,
            "steps": 0,
            "device": "none (frozen policy)",
            "final_evaluation": evaluation,
        }
        write_json(summary, output_dir / "summary.json")
        components.base_policy.close()
        return summary

    assert components.agent is not None
    env = make_environment(config, components.backends)
    rng = np.random.default_rng(config.experiment.seed)
    replay = ReplayBuffer(
        capacity=config.training.replay_capacity,
        state_dim=components.encoder.output_dim,
        action_dim=components.base_policy.action_dim,
        policy_output_dim=components.composer.policy_output_dim,
        seed=config.experiment.seed,
    )

    observation, _ = env.reset(seed=config.experiment.seed)
    components.base_policy.reset()
    state, base_action = encode_step(observation, components)
    trigger = extract_trigger(observation)
    episode = 0
    episode_task_return = 0.0
    episode_regularized_return = 0.0
    episode_corrections: list[float] = []
    episode_gates: list[float] = []
    episode_length = 0
    latest_losses: dict[str, float] = {}
    last_evaluation: dict[str, float] | None = None

    with MetricWriter(metrics_path) as writer:
        for step in range(1, config.training.total_steps + 1):
            if step <= config.training.start_steps:
                policy_output = components.composer.random_policy_output(rng)
            else:
                policy_output = components.agent.select_policy_output(
                    state, base_action, deterministic=False
                )
            intervention = components.composer.compose(base_action, policy_output, trigger=trigger)
            next_observation, task_reward, terminated, truncated, info = env.step(
                intervention.executed_action
            )
            next_state, next_base_action = encode_step(next_observation, components)
            next_trigger = extract_trigger(next_observation)

            intervention_penalty = (
                config.training.lambda_delta * float(np.square(intervention.correction).sum())
                + config.training.lambda_gate * intervention.gate
            )
            regularized_reward = task_reward - intervention_penalty
            replay.add(
                state,
                base_action,
                policy_output,
                regularized_reward,
                next_state,
                next_base_action,
                terminated,
            )

            episode_task_return += task_reward
            episode_regularized_return += regularized_reward
            episode_corrections.append(float(np.linalg.norm(intervention.correction)))
            episode_gates.append(intervention.gate)
            episode_length += 1

            if len(replay) >= config.training.batch_size:
                for _ in range(config.training.updates_per_step):
                    latest_losses = components.agent.update(
                        replay.sample(config.training.batch_size)
                    )

            if terminated or truncated:
                writer.write(
                    step=step,
                    phase="train_episode",
                    episode=episode,
                    success_rate=float(info.get("success", terminated)),
                    task_return=episode_task_return,
                    regularized_return=episode_regularized_return,
                    final_distance=float(info.get("distance", float("nan"))),
                    episode_length=episode_length,
                    correction_norm=float(np.mean(episode_corrections)),
                    gate=float(np.mean(episode_gates)),
                    **latest_losses,
                )
                episode += 1
                observation, _ = env.reset()
                components.base_policy.reset()
                state, base_action = encode_step(observation, components)
                trigger = extract_trigger(observation)
                episode_task_return = 0.0
                episode_regularized_return = 0.0
                episode_corrections.clear()
                episode_gates.clear()
                episode_length = 0
            else:
                observation = next_observation
                state = next_state
                base_action = next_base_action
                trigger = next_trigger

            if (
                config.training.evaluation_interval > 0
                and step % config.training.evaluation_interval == 0
            ):
                last_evaluation = evaluate_components(
                    config,
                    components,
                    episodes=config.training.evaluation_episodes,
                    seed=config.experiment.seed + 10_000,
                )
                writer.write(**_evaluation_row(step, last_evaluation))
                print(
                    f"step={step} success={last_evaluation['success_rate']:.3f} "
                    f"return={last_evaluation['mean_task_return']:.3f} "
                    f"correction={last_evaluation['mean_correction_norm']:.3f} "
                    f"gate={last_evaluation['mean_gate']:.3f}"
                )
                components.base_policy.reset()
                state, base_action = encode_step(observation, components)
                trigger = extract_trigger(observation)

            if (
                config.training.checkpoint_interval > 0
                and step % config.training.checkpoint_interval == 0
            ):
                components.agent.save(output_dir / "checkpoints" / f"step_{step}.pt")

        if last_evaluation is None or (
            config.training.total_steps % max(config.training.evaluation_interval, 1) != 0
        ):
            last_evaluation = evaluate_components(
                config,
                components,
                episodes=config.training.evaluation_episodes,
                seed=config.experiment.seed + 10_000,
            )
            writer.write(**_evaluation_row(config.training.total_steps, last_evaluation))

    env.close()
    components.base_policy.close()
    components.agent.save(output_dir / "checkpoints" / "final.pt")
    summary = {
        "experiment": config.experiment.name,
        "mode": config.mode.value,
        "steps": config.training.total_steps,
        "episodes": episode,
        "updates": components.agent.updates,
        "device": str(components.agent.device),
        "final_evaluation": last_evaluation,
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config")
    parser.add_argument("--output-dir", help="Override experiment.output_dir")
    parser.add_argument("--device", help="Override experiment.device (cpu, cuda, or auto)")
    parser.add_argument("--seed", type=int, help="Override experiment.seed")
    parser.add_argument("--steps", type=int, help="Override training.total_steps")
    return parser.parse_args()


def apply_overrides(config: ProjectConfig, args: argparse.Namespace) -> None:
    if args.output_dir is not None:
        config.experiment.output_dir = args.output_dir
    if args.device is not None:
        config.experiment.device = args.device
    if args.seed is not None:
        config.experiment.seed = args.seed
    if args.steps is not None:
        config.training.total_steps = args.steps
    config.validate()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)
    summary = run_training(config)
    evaluation = summary["final_evaluation"]
    print(
        f"finished {summary['experiment']} on {summary['device']}: "
        f"success={evaluation['success_rate']:.3f}, "
        f"return={evaluation['mean_task_return']:.3f}"
    )


if __name__ == "__main__":
    main()
