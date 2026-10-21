"""Unit checks for stance dynamics, gait events, swing cost and the objective."""

import numpy as np
import pytest

from locomotor_learning_model.initializing.load_initial_body_state import load_initial_body_state
from locomotor_learning_model.initializing.load_learnable_parameters_initial import (
    load_learnable_parameters_initial,
)
from locomotor_learning_model.learning.detect_endstance import detect_endstance
from locomotor_learning_model.learning.detect_midstance import detect_midstance
from locomotor_learning_model.learning.f_objective_asymmetric_nominal import (
    f_objective_asymmetric_nominal,
)
from locomotor_learning_model.learning.single_pendulum_ode import single_pendulum_ode
from locomotor_learning_model.learning.swing_cost_doke import swing_cost_doke


@pytest.mark.parametrize(
    "state",
    [
        [0.0, 0.0, 0.0],
        [0.1, 0.1, 0.0],
        [0.3, 0.2, 0.1],
        [-0.1, -0.05, -0.02],
    ],
)
def test_pendulum_ode(model_parameters, state):
    fixed, controller = model_parameters
    derivative = single_pendulum_ode(0.0, np.array(state), fixed, controller)
    expected = [
        state[1],
        fixed["gravg"] / fixed["leglength"] * np.sin(state[0]),
        fixed["imposedFootSpeeds"]["footSpeed1List"][0],
    ]
    assert derivative.shape == (3,)
    assert np.all(np.isfinite(derivative))
    np.testing.assert_allclose(derivative, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("angle", [0.0, 0.1, 0.3, 0.5, -0.1, -0.3])
def test_detection_functions(model_parameters, angle):
    fixed, controller = model_parameters
    controller["theta_end_thisStep"] = 0.3
    state = np.array([angle, 0.1, 0.0])
    endstance = detect_endstance(0.0, state, fixed, controller)
    midstance = detect_midstance(0.0, state, fixed, controller)
    assert np.isfinite(endstance)
    assert np.isfinite(midstance)
    assert endstance == pytest.approx(angle - 0.3, abs=1e-12)
    assert midstance == pytest.approx(
        fixed["leglength"] * np.sin(angle + fixed["angleSlope"]), abs=1e-12
    )


@pytest.mark.parametrize(
    "duration,initial_speed,final_speed,body_speed",
    [
        (0.5, 0.0, 0.1, 0.0),
        (0.5, 0.1, 0.0, 0.0),
        (1.0, 0.2, 0.3, 0.1),
        (0.3, -0.1, 0.1, 0.0),
        (0.8, 0.5, 0.5, 0.2),
    ],
)
def test_swing_cost_function(
    model_parameters, duration, initial_speed, final_speed, body_speed
):
    fixed, _ = model_parameters
    cost = swing_cost_doke(duration, initial_speed, final_speed, body_speed, fixed)
    assert np.isfinite(cost)
    assert cost >= 0


@pytest.mark.parametrize("perturbation", [0.0, 0.01, 0.05, 0.1, -0.01, -0.05])
def test_objective_function_robustness(model_parameters, perturbation):
    fixed, controller = model_parameters
    parameters = load_learnable_parameters_initial(fixed)
    initial_state = load_initial_body_state(parameters)
    objective, final_state, end_time, energy, constraint = f_objective_asymmetric_nominal(
        parameters + perturbation, initial_state, controller, fixed, 0.0
    )
    assert np.all(np.isfinite([objective, end_time, energy, constraint]))
    assert np.all(np.isfinite(final_state))
    assert objective > 0
    assert end_time > 0


@pytest.mark.parametrize("collect_diagnostics", [False, True])
def test_omitted_fall_penalty_mode_records_com_margin(model_parameters, collect_diagnostics):
    fixed, controller = model_parameters
    fixed.pop("fallPenaltyMode", None)
    fixed["fallPenaltyWeight"] = 0.0
    fixed["collectFallDiagnostics"] = collect_diagnostics
    parameters = load_learnable_parameters_initial(fixed)
    state = load_initial_body_state(parameters)
    f_objective_asymmetric_nominal(parameters, state, controller, fixed, 0.0)
    assert fixed["_lastFallPenaltyMode"] == "com_margin"
    if collect_diagnostics:
        assert np.isfinite(fixed["_lastFallPenalty"])
    else:
        assert np.isnan(fixed["_lastFallPenalty"])
