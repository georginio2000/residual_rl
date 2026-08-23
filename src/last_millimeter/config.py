"""Typed configuration loading and validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from last_millimeter.policies.composition import ControlMode


@dataclass(slots=True)
class ExperimentConfig:
    name: str = "experiment"
    mode: str = ControlMode.RESIDUAL.value
    seed: int = 0
    device: str = "auto"
    output_dir: str = "runs/experiment"


@dataclass(slots=True)
class EnvironmentConfig:
    backend: str = "precision_reach"
    options: dict[str, Any] = field(default_factory=dict)
    max_episode_steps: int = 60
    step_scale: float = 0.1
    success_tolerance: float = 0.035
    success_bonus: float = 5.0
    action_penalty: float = 0.01
    process_noise_std: float = 0.0
    eval_endpoint: str | None = None
    """Optional separate remote-bridge endpoint used only for evaluation.

    Remote bridges (`remote_libero`/`remote_robomimic`) are stateful
    singletons: a second environment instance pointed at the *same* endpoint
    as an in-progress training loop will tear down that shared bridge state
    when it closes. Setting this lets mid-training evaluation run against an
    independent bridge process instead of being disabled entirely.
    """


@dataclass(slots=True)
class BasePolicyConfig:
    backend: str = "proportional"
    gain: float = 5.0
    bias: list[float] = field(default_factory=lambda: [0.18, -0.12])
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RepresentationConfig:
    backend: str = "identity"
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AgentConfig:
    hidden_dims: list[int] = field(default_factory=lambda: [128, 128])
    residual_scale: float = 0.35
    gamma: float = 0.99
    tau: float = 0.005
    learning_rate: float = 3e-4
    alpha: float = 0.2
    automatic_entropy_tuning: bool = True


@dataclass(slots=True)
class TrainingConfig:
    total_steps: int = 30_000
    replay_capacity: int = 100_000
    batch_size: int = 256
    start_steps: int = 1_000
    updates_per_step: int = 1
    evaluation_interval: int = 3_000
    evaluation_episodes: int = 50
    checkpoint_interval: int = 10_000
    lambda_delta: float = 0.02
    lambda_gate: float = 0.0


@dataclass(slots=True)
class ProjectConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    base_policy: BasePolicyConfig = field(default_factory=BasePolicyConfig)
    representation: RepresentationConfig = field(default_factory=RepresentationConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def mode(self) -> ControlMode:
        return ControlMode(self.experiment.mode)

    def validate(self) -> None:
        try:
            ControlMode(self.experiment.mode)
        except ValueError as exc:
            choices = ", ".join(mode.value for mode in ControlMode)
            raise ValueError(f"experiment.mode must be one of: {choices}") from exc

        if not self.environment.backend:
            raise ValueError("environment.backend cannot be empty")
        if not self.base_policy.backend:
            raise ValueError("base_policy.backend cannot be empty")
        if not self.representation.backend:
            raise ValueError("representation.backend cannot be empty")
        if not isinstance(self.base_policy.options, dict):
            raise TypeError("base_policy.options must be a mapping")
        if not isinstance(self.environment.options, dict):
            raise TypeError("environment.options must be a mapping")
        if not isinstance(self.representation.options, dict):
            raise TypeError("representation.options must be a mapping")
        if self.base_policy.backend == "proportional" and len(self.base_policy.bias) != 2:
            raise ValueError("base_policy.bias must contain exactly two values")
        if not self.agent.hidden_dims or any(width <= 0 for width in self.agent.hidden_dims):
            raise ValueError("agent.hidden_dims must contain positive widths")
        if self.environment.max_episode_steps <= 0:
            raise ValueError("environment.max_episode_steps must be positive")
        if self.environment.step_scale <= 0:
            raise ValueError("environment.step_scale must be positive")
        if self.environment.success_tolerance <= 0:
            raise ValueError("environment.success_tolerance must be positive")
        if self.agent.residual_scale <= 0:
            raise ValueError("agent.residual_scale must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("training.batch_size must be positive")
        if self.training.replay_capacity < self.training.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if self.training.total_steps < 0:
            raise ValueError("training.total_steps cannot be negative")
        if self.training.evaluation_episodes <= 0:
            raise ValueError("training.evaluation_episodes must be positive")
        if self.training.lambda_delta < 0 or self.training.lambda_gate < 0:
            raise ValueError("intervention regularization coefficients cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name, {})
    if not isinstance(section, dict):
        raise TypeError(f"configuration section {name!r} must be a mapping")
    return section


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise TypeError("top-level configuration must be a mapping")

    config = ProjectConfig(
        experiment=ExperimentConfig(**_section(raw, "experiment")),
        environment=EnvironmentConfig(**_section(raw, "environment")),
        base_policy=BasePolicyConfig(**_section(raw, "base_policy")),
        representation=RepresentationConfig(**_section(raw, "representation")),
        agent=AgentConfig(**_section(raw, "agent")),
        training=TrainingConfig(**_section(raw, "training")),
    )
    config.validate()
    return config


def save_config(config: ProjectConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
