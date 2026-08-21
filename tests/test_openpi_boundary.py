from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from last_millimeter.policies import OpenPIClientBasePolicy


class MockOpenPIClient:
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        assert observation["prompt"] == "pick up the bowl"
        self.calls += 1
        offset = 10 * self.calls
        return {
            "actions": np.asarray(
                [[offset + index, -(offset + index)] for index in range(4)],
                dtype=np.float32,
            )
        }


def test_openpi_boundary_replans_one_observation_at_a_time() -> None:
    client = MockOpenPIClient()
    policy = OpenPIClientBasePolicy(client, action_dim=2, replan_steps=3)
    observation = {"prompt": "pick up the bowl"}

    np.testing.assert_array_equal(policy.act(observation), [10.0, -10.0])
    np.testing.assert_array_equal(policy.act(observation), [11.0, -11.0])
    np.testing.assert_array_equal(policy.act(observation), [12.0, -12.0])
    assert client.calls == 1

    np.testing.assert_array_equal(policy.act(observation), [20.0, -20.0])
    assert client.calls == 2

    policy.reset()
    np.testing.assert_array_equal(policy.act(observation), [30.0, -30.0])
    assert client.calls == 3


def test_openpi_boundary_validates_action_chunk_shape() -> None:
    class BadClient:
        def infer(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
            del observation
            return {"actions": np.zeros((2, 3), dtype=np.float32)}

    policy = OpenPIClientBasePolicy(BadClient(), action_dim=2, replan_steps=1)

    with pytest.raises(ValueError, match="chunk_length, 2"):
        policy.act({"prompt": "test"})
