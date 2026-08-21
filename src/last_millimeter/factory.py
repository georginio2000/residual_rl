"""Construct configurable experiment backends from project configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from last_millimeter.config import (
    BasePolicyConfig,
    EnvironmentConfig,
    ProjectConfig,
    RepresentationConfig,
)
from last_millimeter.envs import PrecisionReachEnv
from last_millimeter.policies import (
    ActionComposer,
    BasePolicy,
    ControlMode,
    OpenPIClientBasePolicy,
    ProportionalBasePolicy,
)
from last_millimeter.representations import (
    IdentityStateEncoder,
    ObservationKeyEncoder,
    RepresentationEncoder,
)
from last_millimeter.rl import SACAgent


Environment = gym.Env[Any, np.ndarray]


@dataclass(frozen=True, slots=True)
class BackendContext:
    """Environment dimensions exposed to policy and representation builders."""

    observation_space: gym.Space[Any]
    action_space: spaces.Box
    observation_dim: int
    action_dim: int
    action_low: float
    action_high: float


EnvironmentBuilder = Callable[[EnvironmentConfig], Environment]
BasePolicyBuilder = Callable[[BasePolicyConfig, BackendContext], BasePolicy]
RepresentationBuilder = Callable[[RepresentationConfig, BackendContext], RepresentationEncoder]


@dataclass(slots=True)
class BackendRegistry:
    """Named construction boundary for optional simulator and VLA integrations."""

    environments: dict[str, EnvironmentBuilder] = field(default_factory=dict)
    base_policies: dict[str, BasePolicyBuilder] = field(default_factory=dict)
    representations: dict[str, RepresentationBuilder] = field(default_factory=dict)

    def register_environment(self, name: str, builder: EnvironmentBuilder) -> None:
        self._register(self.environments, name, builder)

    def register_base_policy(self, name: str, builder: BasePolicyBuilder) -> None:
        self._register(self.base_policies, name, builder)

    def register_representation(self, name: str, builder: RepresentationBuilder) -> None:
        self._register(self.representations, name, builder)

    @staticmethod
    def _register(registry: dict[str, Any], name: str, builder: Any) -> None:
        if not name:
            raise ValueError("backend name cannot be empty")
        if name in registry:
            raise ValueError(f"backend {name!r} is already registered")
        registry[name] = builder

    @classmethod
    def with_defaults(cls) -> BackendRegistry:
        registry = cls()
        registry.register_environment("precision_reach", _make_precision_reach)
        registry.register_base_policy("proportional", _make_proportional_policy)
        registry.register_base_policy("openpi_websocket", _make_openpi_websocket_policy)
        registry.register_representation("identity", _make_identity_encoder)
        registry.register_representation("observation_key", _make_observation_key_encoder)
        return registry


@dataclass(slots=True)
class ExperimentComponents:
    base_policy: BasePolicy
    encoder: RepresentationEncoder
    composer: ActionComposer
    agent: SACAgent | None
    backends: BackendRegistry


def _unknown_backend(kind: str, name: str, available: dict[str, Any]) -> ValueError:
    choices = ", ".join(sorted(available)) or "<none>"
    return ValueError(f"unknown {kind} backend {name!r}; available backends: {choices}")


def _make_precision_reach(config: EnvironmentConfig) -> Environment:
    if config.options:
        raise ValueError(f"unknown precision_reach options: {sorted(config.options)}")
    return PrecisionReachEnv(
        max_episode_steps=config.max_episode_steps,
        step_scale=config.step_scale,
        success_tolerance=config.success_tolerance,
        success_bonus=config.success_bonus,
        action_penalty=config.action_penalty,
        process_noise_std=config.process_noise_std,
    )


def _make_proportional_policy(
    config: BasePolicyConfig, context: BackendContext
) -> BasePolicy:
    if context.observation_dim != 4 or context.action_dim != 2:
        raise ValueError("proportional backend requires observation_dim=4 and action_dim=2")
    return ProportionalBasePolicy(gain=config.gain, bias=tuple(config.bias))


def _make_openpi_websocket_policy(
    config: BasePolicyConfig, context: BackendContext
) -> BasePolicy:
    options = dict(config.options)
    host = str(options.pop("host", "127.0.0.1"))
    port = int(options.pop("port", 8000))
    replan_steps = int(options.pop("replan_steps", 5))
    if options:
        raise ValueError(f"unknown openpi_websocket options: {sorted(options)}")
    try:
        from openpi_client import websocket_client_policy
    except ImportError as exc:
        raise RuntimeError(
            "openpi_websocket requires the lightweight openpi-client package; "
            "install it only in the simulator/client environment"
        ) from exc
    client = websocket_client_policy.WebsocketClientPolicy(host, port)
    return OpenPIClientBasePolicy(
        client, action_dim=context.action_dim, replan_steps=replan_steps
    )


def _make_identity_encoder(
    config: RepresentationConfig, context: BackendContext
) -> RepresentationEncoder:
    if config.options:
        raise ValueError(f"unknown identity representation options: {sorted(config.options)}")
    return IdentityStateEncoder(context.observation_dim)


def _make_observation_key_encoder(
    config: RepresentationConfig, context: BackendContext
) -> RepresentationEncoder:
    del context
    options = dict(config.options)
    try:
        key = str(options.pop("key"))
        output_dim = int(options.pop("output_dim"))
    except KeyError as exc:
        raise ValueError("observation_key requires 'key' and 'output_dim' options") from exc
    if options:
        raise ValueError(f"unknown observation_key options: {sorted(options)}")
    return ObservationKeyEncoder(key, output_dim)


def _backend_context(env: Environment) -> BackendContext:
    action_space = env.action_space
    if not isinstance(action_space, spaces.Box) or len(action_space.shape) != 1:
        raise TypeError("continuous-control backends require a one-dimensional Box action space")
    low = np.asarray(action_space.low, dtype=np.float32)
    high = np.asarray(action_space.high, dtype=np.float32)
    if not np.all(low == low.flat[0]) or not np.all(high == high.flat[0]):
        raise ValueError("all action dimensions must currently use the same bounds")

    observation_space = env.observation_space
    try:
        observation_dim = int(spaces.utils.flatdim(observation_space))
    except (NotImplementedError, TypeError) as exc:
        raise TypeError("environment observation space must have a finite flat dimension") from exc
    return BackendContext(
        observation_space=observation_space,
        action_space=action_space,
        observation_dim=observation_dim,
        action_dim=int(action_space.shape[0]),
        action_low=float(low.flat[0]),
        action_high=float(high.flat[0]),
    )


def make_environment(
    config: ProjectConfig, backends: BackendRegistry | None = None
) -> Environment:
    backends = backends or BackendRegistry.with_defaults()
    try:
        builder = backends.environments[config.environment.backend]
    except KeyError as exc:
        raise _unknown_backend(
            "environment", config.environment.backend, backends.environments
        ) from exc
    return builder(config.environment)


def make_components(
    config: ProjectConfig, backends: BackendRegistry | None = None
) -> ExperimentComponents:
    backends = backends or BackendRegistry.with_defaults()
    probe_env = make_environment(config, backends)
    try:
        context = _backend_context(probe_env)
    finally:
        probe_env.close()

    try:
        policy_builder = backends.base_policies[config.base_policy.backend]
    except KeyError as exc:
        raise _unknown_backend(
            "base-policy", config.base_policy.backend, backends.base_policies
        ) from exc
    base_policy = policy_builder(config.base_policy, context)
    if base_policy.action_dim != context.action_dim:
        raise ValueError(
            f"base policy action_dim={base_policy.action_dim} does not match "
            f"environment action_dim={context.action_dim}"
        )

    try:
        representation_builder = backends.representations[config.representation.backend]
    except KeyError as exc:
        raise _unknown_backend(
            "representation", config.representation.backend, backends.representations
        ) from exc
    encoder = representation_builder(config.representation, context)
    composer = ActionComposer(
        mode=config.mode,
        action_dim=context.action_dim,
        residual_scale=config.agent.residual_scale,
        action_low=context.action_low,
        action_high=context.action_high,
    )

    agent = None
    if config.mode is not ControlMode.FROZEN:
        agent = SACAgent(
            state_dim=encoder.output_dim,
            action_dim=context.action_dim,
            composer=composer,
            hidden_dims=config.agent.hidden_dims,
            gamma=config.agent.gamma,
            tau=config.agent.tau,
            learning_rate=config.agent.learning_rate,
            alpha=config.agent.alpha,
            automatic_entropy_tuning=config.agent.automatic_entropy_tuning,
            device=config.experiment.device,
        )

    return ExperimentComponents(base_policy, encoder, composer, agent, backends)


def encode_step(
    observation: Any,
    components: ExperimentComponents,
) -> tuple[np.ndarray, np.ndarray]:
    state = components.encoder.encode(observation)
    base_action = components.base_policy.act(observation)
    return state, base_action
