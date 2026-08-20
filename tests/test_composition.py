import numpy as np
import pytest
import torch

from last_millimeter.policies import ActionComposer, ControlMode


@pytest.mark.parametrize("mode", [ControlMode.SCRATCH, ControlMode.RESIDUAL, ControlMode.GATED])
def test_numpy_and_torch_composition_match(mode: ControlMode) -> None:
    composer = ActionComposer(mode, action_dim=2, residual_scale=0.3)
    base = np.asarray([0.8, -0.9], dtype=np.float32)
    outputs = {
        ControlMode.SCRATCH: np.asarray([-0.4, 0.2], dtype=np.float32),
        ControlMode.RESIDUAL: np.asarray([0.3, -0.3], dtype=np.float32),
        ControlMode.GATED: np.asarray([0.3, -0.3, 0.25], dtype=np.float32),
    }
    output = outputs[mode]

    numpy_result = composer.compose(base, output).executed_action
    torch_result = composer.compose_tensor(
        torch.from_numpy(base).unsqueeze(0), torch.from_numpy(output).unsqueeze(0)
    )

    np.testing.assert_allclose(numpy_result, torch_result.squeeze(0).numpy())


def test_gated_composition_reports_intervention() -> None:
    composer = ActionComposer(ControlMode.GATED, action_dim=2, residual_scale=0.4)
    result = composer.compose(
        np.asarray([0.1, 0.2], dtype=np.float32),
        np.asarray([0.4, -0.2, 0.5], dtype=np.float32),
    )

    np.testing.assert_allclose(result.executed_action, [0.3, 0.1])
    np.testing.assert_allclose(result.correction, [0.4, -0.2])
    assert result.gate == 0.5


def test_frozen_composition_ignores_policy_output() -> None:
    composer = ActionComposer(ControlMode.FROZEN, action_dim=2, residual_scale=0.3)
    base = np.asarray([0.25, -0.5], dtype=np.float32)
    result = composer.compose(base, None)

    np.testing.assert_array_equal(result.executed_action, base)
    np.testing.assert_array_equal(result.correction, np.zeros(2, dtype=np.float32))
    assert result.gate == 0.0

