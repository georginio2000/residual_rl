import numpy as np
import pytest

from last_millimeter.rl import ReplayBuffer


def test_replay_buffer_wraps_and_samples_expected_shapes() -> None:
    replay = ReplayBuffer(capacity=3, state_dim=4, action_dim=2, seed=0)
    for value in range(5):
        state = np.full(4, value, dtype=np.float32)
        action = np.full(2, value, dtype=np.float32)
        replay.add(state, action, action, float(value), state + 1, action + 1, False)

    assert len(replay) == 3
    batch = replay.sample(2)
    assert batch["state"].shape == (2, 4)
    assert batch["policy_output"].shape == (2, 2)
    assert batch["reward"].shape == (2, 1)


def test_replay_buffer_rejects_oversized_sample() -> None:
    replay = ReplayBuffer(capacity=3, state_dim=4, action_dim=2)
    with pytest.raises(ValueError, match="cannot sample"):
        replay.sample(1)
