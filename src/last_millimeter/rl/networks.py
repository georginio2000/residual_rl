"""Neural networks used by soft actor-critic."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.distributions import Normal


def build_mlp(input_dim: int, hidden_dims: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Gaussian policy transformed to bounded, mode-specific policy outputs."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        output_scale: Tensor,
        output_bias: Tensor,
    ) -> None:
        super().__init__()
        if output_scale.shape != (output_dim,) or output_bias.shape != (output_dim,):
            raise ValueError("actor output scale and bias must match output_dim")
        if torch.any(output_scale <= 0):
            raise ValueError("all actor output scales must be positive")

        self.backbone = build_mlp(input_dim, hidden_dims, 2 * output_dim)
        self.output_dim = output_dim
        self.register_buffer("output_scale", output_scale.clone().detach())
        self.register_buffer("output_bias", output_bias.clone().detach())

    def _distribution(self, state: Tensor) -> Normal:
        mean, log_std = self.backbone(state).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return Normal(mean, log_std.exp())

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self._distribution(state)
        pre_tanh = distribution.rsample()
        squashed = torch.tanh(pre_tanh)
        output = squashed * self.output_scale + self.output_bias

        transform_jacobian = self.output_scale * (1.0 - squashed.pow(2))
        log_prob = distribution.log_prob(pre_tanh) - torch.log(transform_jacobian + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_output = torch.tanh(distribution.mean) * self.output_scale + self.output_bias
        return output, log_prob, mean_output

    def deterministic(self, state: Tensor) -> Tensor:
        distribution = self._distribution(state)
        return torch.tanh(distribution.mean) * self.output_scale + self.output_bias


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.network = build_mlp(state_dim + action_dim, hidden_dims, 1)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.network(torch.cat((state, action), dim=-1))


class TwinQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int]) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)

    def forward(self, state: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        return self.q1(state, action), self.q2(state, action)

