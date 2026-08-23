"""Map an RL policy output and frozen reference action to an executed action."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from torch import Tensor


class ControlMode(str, Enum):
    FROZEN = "frozen"
    SCRATCH = "scratch"
    RESIDUAL = "residual"
    GATED = "gated"


@dataclass(frozen=True, slots=True)
class Intervention:
    executed_action: np.ndarray
    correction: np.ndarray
    gate: float


class ActionComposer:
    """Compose policy outputs using identical NumPy and differentiable Torch paths."""

    # Gate bias/scale for an untrained (near-zero pre-tanh output) GATED actor.
    # An untrained actor's affine output collapses toward its bias, so a 0.5
    # bias means the actor starts by intervening on ~half of every action by
    # default -- the opposite of the intended "trust the frozen policy unless
    # learned otherwise" prior. Biasing low instead means training starts from
    # near-baseline behavior. The scale is set to (1 - bias) so the gate can
    # still reach a full 1.0 at the actor's output extreme.
    GATE_INIT_BIAS = 0.1

    def __init__(
        self,
        mode: ControlMode,
        action_dim: int,
        residual_scale: float,
        action_low: float = -1.0,
        action_high: float = 1.0,
    ) -> None:
        self.mode = ControlMode(mode)
        self.action_dim = int(action_dim)
        self.residual_scale = float(residual_scale)
        self.action_low = float(action_low)
        self.action_high = float(action_high)

    @property
    def policy_output_dim(self) -> int:
        if self.mode is ControlMode.GATED:
            return self.action_dim + 1
        if self.mode is ControlMode.FROZEN:
            return 0
        return self.action_dim

    @property
    def actor_uses_base_action(self) -> bool:
        return self.mode in (ControlMode.RESIDUAL, ControlMode.GATED)

    def policy_output_scale(self) -> np.ndarray:
        """Affine scale applied after tanh by the stochastic actor."""
        if self.mode is ControlMode.GATED:
            return np.concatenate(
                (
                    np.full(self.action_dim, self.residual_scale, dtype=np.float32),
                    np.asarray([1.0 - self.GATE_INIT_BIAS], dtype=np.float32),
                )
            )
        if self.mode is ControlMode.RESIDUAL:
            return np.full(self.action_dim, self.residual_scale, dtype=np.float32)
        if self.mode is ControlMode.SCRATCH:
            return np.full(self.action_dim, (self.action_high - self.action_low) / 2, dtype=np.float32)
        return np.empty(0, dtype=np.float32)

    def policy_output_bias(self) -> np.ndarray:
        if self.mode is ControlMode.GATED:
            return np.concatenate(
                (
                    np.zeros(self.action_dim, dtype=np.float32),
                    np.asarray([self.GATE_INIT_BIAS], dtype=np.float32),
                )
            )
        if self.mode is ControlMode.SCRATCH:
            midpoint = (self.action_high + self.action_low) / 2
            return np.full(self.action_dim, midpoint, dtype=np.float32)
        return np.zeros(self.policy_output_dim, dtype=np.float32)

    def compose(self, base_action: np.ndarray, policy_output: np.ndarray | None) -> Intervention:
        base_action = np.asarray(base_action)
        if base_action.shape != (self.action_dim,):
            raise ValueError(f"base action must have shape ({self.action_dim},)")
        if not np.all(np.isfinite(base_action)):
            raise ValueError("base action must contain only finite values")

        if self.mode is ControlMode.FROZEN:
            executed = base_action
            correction = np.zeros(self.action_dim, dtype=np.float32)
            gate = 0.0
        else:
            base_action = base_action.astype(np.float32, copy=False)
            if policy_output is None:
                raise ValueError(f"policy output is required in {self.mode.value} mode")
            policy_output = np.asarray(policy_output, dtype=np.float32)
            expected = (self.policy_output_dim,)
            if policy_output.shape != expected:
                raise ValueError(f"policy output must have shape {expected}")

            if self.mode is ControlMode.SCRATCH:
                executed = policy_output
                correction = np.zeros_like(base_action)
                gate = 0.0
            elif self.mode is ControlMode.RESIDUAL:
                correction = policy_output
                gate = 1.0
                executed = base_action + correction
            else:
                correction = policy_output[: self.action_dim]
                gate = float(np.clip(policy_output[-1], 0.0, 1.0))
                executed = base_action + gate * correction

        executed = np.clip(executed, self.action_low, self.action_high)
        if self.mode is not ControlMode.FROZEN:
            executed = executed.astype(np.float32)
        return Intervention(executed, correction.astype(np.float32), gate)

    def compose_tensor(self, base_action: Tensor, policy_output: Tensor) -> Tensor:
        """Differentiable composition used by SAC actor and target calculations."""
        if self.mode is ControlMode.SCRATCH:
            executed = policy_output
        elif self.mode is ControlMode.RESIDUAL:
            executed = base_action + policy_output
        elif self.mode is ControlMode.GATED:
            correction = policy_output[..., : self.action_dim]
            gate = policy_output[..., self.action_dim : self.action_dim + 1]
            executed = base_action + gate * correction
        else:
            executed = base_action
        return torch.clamp(executed, self.action_low, self.action_high)

    def random_policy_output(self, rng: np.random.Generator) -> np.ndarray:
        if self.mode is ControlMode.SCRATCH:
            return rng.uniform(self.action_low, self.action_high, self.action_dim).astype(np.float32)
        if self.mode is ControlMode.RESIDUAL:
            return rng.uniform(-self.residual_scale, self.residual_scale, self.action_dim).astype(
                np.float32
            )
        if self.mode is ControlMode.GATED:
            correction = rng.uniform(
                -self.residual_scale, self.residual_scale, self.action_dim
            ).astype(np.float32)
            gate = np.asarray([rng.uniform(0.0, 1.0)], dtype=np.float32)
            return np.concatenate((correction, gate))
        return np.empty(0, dtype=np.float32)
