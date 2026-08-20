"""Construct experiment components from a project configuration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from last_millimeter.config import ProjectConfig
from last_millimeter.envs import PrecisionReachEnv
from last_millimeter.policies import ActionComposer, BasePolicy, ControlMode, ProportionalBasePolicy
from last_millimeter.representations import IdentityStateEncoder, RepresentationEncoder
from last_millimeter.rl import SACAgent


@dataclass(slots=True)
class ExperimentComponents:
    base_policy: BasePolicy
    encoder: RepresentationEncoder
    composer: ActionComposer
    agent: SACAgent | None


def make_environment(config: ProjectConfig) -> PrecisionReachEnv:
    return PrecisionReachEnv(
        max_episode_steps=config.environment.max_episode_steps,
        step_scale=config.environment.step_scale,
        success_tolerance=config.environment.success_tolerance,
        success_bonus=config.environment.success_bonus,
        action_penalty=config.environment.action_penalty,
        process_noise_std=config.environment.process_noise_std,
    )


def make_components(config: ProjectConfig) -> ExperimentComponents:
    base_policy = ProportionalBasePolicy(
        gain=config.base_policy.gain,
        bias=tuple(config.base_policy.bias),
    )
    encoder = IdentityStateEncoder(observation_dim=4)
    composer = ActionComposer(
        mode=config.mode,
        action_dim=base_policy.action_dim,
        residual_scale=config.agent.residual_scale,
    )

    agent = None
    if config.mode is not ControlMode.FROZEN:
        agent = SACAgent(
            state_dim=encoder.output_dim,
            action_dim=base_policy.action_dim,
            composer=composer,
            hidden_dims=config.agent.hidden_dims,
            gamma=config.agent.gamma,
            tau=config.agent.tau,
            learning_rate=config.agent.learning_rate,
            alpha=config.agent.alpha,
            automatic_entropy_tuning=config.agent.automatic_entropy_tuning,
            device=config.experiment.device,
        )

    return ExperimentComponents(base_policy, encoder, composer, agent)


def encode_step(
    observation: np.ndarray,
    components: ExperimentComponents,
) -> tuple[np.ndarray, np.ndarray]:
    state = components.encoder.encode(observation)
    base_action = components.base_policy.act(observation)
    return state, base_action

