from pathlib import Path

import numpy as np
import pytest
import torch

from last_millimeter.policies import ActionComposer, ControlMode
from last_millimeter.rl import SACAgent


def make_batch(policy_output_dim: int, batch_size: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(1)
    return {
        "state": rng.normal(size=(batch_size, 4)).astype(np.float32),
        "base_action": rng.uniform(-1, 1, size=(batch_size, 2)).astype(np.float32),
        "policy_output": rng.uniform(
            -0.3, 0.3, size=(batch_size, policy_output_dim)
        ).astype(np.float32),
        "reward": rng.normal(size=(batch_size, 1)).astype(np.float32),
        "next_state": rng.normal(size=(batch_size, 4)).astype(np.float32),
        "next_base_action": rng.uniform(-1, 1, size=(batch_size, 2)).astype(np.float32),
        "terminated": rng.integers(0, 2, size=(batch_size, 1)).astype(np.float32),
        "trigger": np.ones((batch_size, 1), dtype=np.float32),
        "next_trigger": np.ones((batch_size, 1), dtype=np.float32),
    }


@pytest.mark.parametrize(
    "mode",
    [ControlMode.SCRATCH, ControlMode.RESIDUAL, ControlMode.GATED, ControlMode.TRIGGERED],
)
def test_sac_update_is_finite(mode: ControlMode) -> None:
    composer = ActionComposer(mode, action_dim=2, residual_scale=0.3)
    agent = SACAgent(
        state_dim=4,
        action_dim=2,
        composer=composer,
        hidden_dims=(16, 16),
        device="cpu",
    )

    metrics = agent.update(make_batch(composer.policy_output_dim))

    assert agent.updates == 1
    assert all(np.isfinite(value) for value in metrics.values())
    output = agent.select_policy_output(
        np.zeros(4, dtype=np.float32),
        np.zeros(2, dtype=np.float32),
        deterministic=True,
    )
    assert output.shape == (composer.policy_output_dim,)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    composer = ActionComposer(ControlMode.RESIDUAL, action_dim=2, residual_scale=0.3)
    agent = SACAgent(
        state_dim=4,
        action_dim=2,
        composer=composer,
        hidden_dims=(16, 16),
        device="cpu",
    )
    state = np.zeros(4, dtype=np.float32)
    base_action = np.ones(2, dtype=np.float32)
    expected = agent.select_policy_output(state, base_action, deterministic=True)
    checkpoint = tmp_path / "agent.pt"
    agent.save(checkpoint)

    restored = SACAgent(
        state_dim=4,
        action_dim=2,
        composer=composer,
        hidden_dims=(16, 16),
        device="cpu",
    )
    restored.load(checkpoint, load_optimizers=False)
    actual = restored.select_policy_output(state, base_action, deterministic=True)

    torch.testing.assert_close(torch.from_numpy(actual), torch.from_numpy(expected))
