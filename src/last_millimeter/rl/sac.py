"""Soft actor-critic with differentiable residual action composition."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam

from last_millimeter.policies.composition import ActionComposer
from last_millimeter.rl.networks import SquashedGaussianActor, TwinQNetwork


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return resolved


class SACAgent:
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        composer: ActionComposer,
        hidden_dims: Sequence[int] = (256, 256),
        gamma: float = 0.99,
        tau: float = 0.005,
        learning_rate: float = 3e-4,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        device: str = "auto",
    ) -> None:
        if composer.policy_output_dim <= 0:
            raise ValueError("SACAgent cannot be constructed in frozen mode")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.composer = composer
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.device = resolve_device(device)
        self.automatic_entropy_tuning = bool(automatic_entropy_tuning)

        actor_input_dim = state_dim + (action_dim if composer.actor_uses_base_action else 0)
        output_scale = torch.as_tensor(composer.policy_output_scale(), dtype=torch.float32)
        output_bias = torch.as_tensor(composer.policy_output_bias(), dtype=torch.float32)
        self.actor = SquashedGaussianActor(
            actor_input_dim,
            composer.policy_output_dim,
            hidden_dims,
            output_scale,
            output_bias,
        ).to(self.device)
        critic_state_dim = actor_input_dim
        critic_action_dim = composer.policy_output_dim
        self.critics = TwinQNetwork(critic_state_dim, critic_action_dim, hidden_dims).to(self.device)
        self.target_critics = TwinQNetwork(
            critic_state_dim, critic_action_dim, hidden_dims
        ).to(self.device)
        self.target_critics.load_state_dict(self.critics.state_dict())
        self.target_critics.requires_grad_(False)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = Adam(self.critics.parameters(), lr=learning_rate)

        if self.automatic_entropy_tuning:
            self.target_entropy = -float(composer.policy_output_dim)
            self.log_alpha = torch.tensor(
                np.log(alpha), dtype=torch.float32, device=self.device, requires_grad=True
            )
            self.alpha_optimizer: Adam | None = Adam([self.log_alpha], lr=learning_rate)
            self._fixed_alpha = None
        else:
            self.target_entropy = 0.0
            self.log_alpha = None
            self.alpha_optimizer = None
            self._fixed_alpha = float(alpha)

        self.updates = 0

    @property
    def alpha(self) -> Tensor:
        if self.log_alpha is not None:
            return self.log_alpha.exp()
        return torch.tensor(self._fixed_alpha, dtype=torch.float32, device=self.device)

    def _actor_input(self, state: Tensor, base_action: Tensor) -> Tensor:
        if self.composer.actor_uses_base_action:
            return torch.cat((state, base_action), dim=-1)
        return state

    @torch.no_grad()
    def select_policy_output(
        self,
        state: np.ndarray,
        base_action: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        base_tensor = torch.as_tensor(
            base_action, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        actor_input = self._actor_input(state_tensor, base_tensor)
        if deterministic:
            output = self.actor.deterministic(actor_input)
        else:
            output, _, _ = self.actor.sample(actor_input)
        return output.squeeze(0).cpu().numpy().astype(np.float32)

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        tensors = {
            key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
            for key, value in batch.items()
        }
        state = tensors["state"]
        base_action = tensors["base_action"]
        replay_policy_output = tensors["policy_output"]
        reward = tensors["reward"]
        next_state = tensors["next_state"]
        next_base_action = tensors["next_base_action"]
        terminated = tensors["terminated"]

        with torch.no_grad():
            next_actor_input = self._actor_input(next_state, next_base_action)
            next_policy_output, next_log_prob, _ = self.actor.sample(next_actor_input)
            next_q1, next_q2 = self.target_critics(next_actor_input, next_policy_output)
            next_q = torch.minimum(next_q1, next_q2) - self.alpha.detach() * next_log_prob
            target_q = reward + (1.0 - terminated) * self.gamma * next_q

        actor_input = self._actor_input(state, base_action)
        q1, q2 = self.critics(actor_input, replay_policy_output)
        q1_loss = torch.nn.functional.mse_loss(q1, target_q)
        q2_loss = torch.nn.functional.mse_loss(q2, target_q)
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critics.requires_grad_(False)
        policy_output, log_prob, _ = self.actor.sample(actor_input)
        policy_q1, policy_q2 = self.critics(actor_input, policy_output)
        actor_loss = (self.alpha.detach() * log_prob - torch.minimum(policy_q1, policy_q2)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critics.requires_grad_(True)

        alpha_loss_value = 0.0
        if self.log_alpha is not None and self.alpha_optimizer is not None:
            alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss_value = float(alpha_loss.detach().cpu())

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critics.parameters(), self.critics.parameters(), strict=True
            ):
                target_parameter.mul_(1.0 - self.tau).add_(parameter, alpha=self.tau)

        self.updates += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.detach().cpu()),
            "mean_log_prob": float(log_prob.detach().mean().cpu()),
        }

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "actor": self.actor.state_dict(),
            "critics": self.critics.state_dict(),
            "target_critics": self.target_critics.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "updates": self.updates,
        }
        if self.log_alpha is not None and self.alpha_optimizer is not None:
            payload["log_alpha"] = self.log_alpha.detach().cpu()
            payload["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        return payload

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint(), path)

    def load(self, path: str | Path, *, load_optimizers: bool = True) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critics.load_state_dict(checkpoint["critics"])
        self.target_critics.load_state_dict(checkpoint["target_critics"])
        self.updates = int(checkpoint.get("updates", 0))
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        if self.log_alpha is not None and "log_alpha" in checkpoint:
            with torch.no_grad():
                self.log_alpha.copy_(checkpoint["log_alpha"].to(self.device))
            if load_optimizers and self.alpha_optimizer is not None:
                self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
