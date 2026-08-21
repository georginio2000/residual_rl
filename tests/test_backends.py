import numpy as np
import pytest

from last_millimeter.config import (
    BasePolicyConfig,
    EnvironmentConfig,
    ExperimentConfig,
    ProjectConfig,
    RepresentationConfig,
)
from last_millimeter.envs import PrecisionReachEnv
from last_millimeter.factory import BackendRegistry, make_components, make_environment
from last_millimeter.policies import BasePolicy
from last_millimeter.representations import ObservationKeyEncoder, RepresentationEncoder


class MockBasePolicy(BasePolicy):
    @property
    def action_dim(self) -> int:
        return 2

    def act(self, observation: object) -> np.ndarray:
        del observation
        return np.asarray([0.25, -0.25], dtype=np.float32)


class MockEncoder(RepresentationEncoder):
    @property
    def output_dim(self) -> int:
        return 1

    def encode(self, observation: object) -> np.ndarray:
        values = np.asarray(observation, dtype=np.float32)
        return np.asarray([values.sum()], dtype=np.float32)


def test_named_mock_backends_are_selected_without_optional_dependencies() -> None:
    registry = BackendRegistry()
    registry.register_environment("mock_env", lambda _: PrecisionReachEnv())
    registry.register_base_policy("mock_policy", lambda _config, _context: MockBasePolicy())
    registry.register_representation("mock_rep", lambda _config, _context: MockEncoder())
    config = ProjectConfig(
        experiment=ExperimentConfig(mode="frozen"),
        environment=EnvironmentConfig(backend="mock_env"),
        base_policy=BasePolicyConfig(backend="mock_policy"),
        representation=RepresentationConfig(backend="mock_rep"),
    )

    components = make_components(config, registry)

    assert isinstance(components.base_policy, MockBasePolicy)
    assert isinstance(components.encoder, MockEncoder)
    assert components.composer.action_dim == 2
    assert components.agent is None


def test_unknown_environment_backend_lists_available_choices() -> None:
    config = ProjectConfig(environment=EnvironmentConfig(backend="missing"))

    with pytest.raises(ValueError, match="precision_reach"):
        make_environment(config)


def test_observation_key_encoder_copies_flat_state() -> None:
    encoder = ObservationKeyEncoder("observation/state", output_dim=3)
    source = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)

    encoded = encoder.encode({"observation/state": source})
    source[0] = 99.0

    np.testing.assert_array_equal(encoded, [1.0, 2.0, 3.0])
