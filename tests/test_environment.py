import numpy as np

from last_millimeter.envs import PrecisionReachEnv
from last_millimeter.policies import ProportionalBasePolicy


def test_precision_environment_is_seed_reproducible() -> None:
    env = PrecisionReachEnv()
    first_observation, first_info = env.reset(seed=7)
    second_observation, second_info = env.reset(seed=7)

    np.testing.assert_allclose(first_observation, second_observation)
    assert first_info == second_info


def test_successful_step_terminates() -> None:
    env = PrecisionReachEnv(step_scale=0.1, success_tolerance=0.02)
    observation, _ = env.reset(
        seed=0,
        options={"position": [0.0, 0.0], "goal": [0.1, 0.0]},
    )
    assert observation.shape == (4,)

    _, reward, terminated, truncated, info = env.step(np.asarray([1.0, 0.0]))

    assert terminated
    assert not truncated
    assert info["success"]
    assert reward > 0


def test_base_policy_moves_toward_goal_but_has_precision_bias() -> None:
    env = PrecisionReachEnv(success_tolerance=0.035)
    policy = ProportionalBasePolicy(gain=5.0, bias=(0.18, -0.12))
    observation, info = env.reset(
        seed=0,
        options={"position": [-0.6, -0.5], "goal": [0.5, 0.5]},
    )
    initial_distance = info["distance"]

    for _ in range(env.max_episode_steps):
        observation, _, terminated, truncated, info = env.step(policy.act(observation))
        if terminated or truncated:
            break

    assert info["distance"] < 0.1 * initial_distance
    assert info["distance"] > env.success_tolerance

