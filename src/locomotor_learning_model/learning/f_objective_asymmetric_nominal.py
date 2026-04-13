from __future__ import annotations

from typing import Tuple

import numpy as np

from locomotor_learning_model.learning.simulate_many_steps_asymmetric_control import (
    simulate_many_steps_asymmetric_control,
)


def f_objective_asymmetric_nominal(
    p_input_controller_nominal: np.ndarray,
    state_var0_model: np.ndarray,
    param_controller,
    param_fixed,
    t_start: float,
) -> Tuple[float, np.ndarray, float, float, float]:
    """Python port of ``fObjective_AsymmetricNominal.m``."""

    # unwrap controller parameters for odd and even steps
    param_controller['Odd']['theta_end_nominal'] = float(p_input_controller_nominal[0])
    param_controller['Odd']['ydot_at_midstance_nominal_beltframe'] = float(p_input_controller_nominal[1])
    param_controller['Odd']['pushoff_impulse_magnitude_nominal'] = float(p_input_controller_nominal[2])
    param_controller['Odd']['y_at_midstance_nominal_slopeframe'] = float(p_input_controller_nominal[3])
    param_controller['Odd']['sumy_at_midstance_nominal_slopeframe'] = float(p_input_controller_nominal[4])

    param_controller['Even']['theta_end_nominal'] = float(p_input_controller_nominal[5])
    param_controller['Even']['ydot_at_midstance_nominal_beltframe'] = float(p_input_controller_nominal[6])
    param_controller['Even']['pushoff_impulse_magnitude_nominal'] = float(p_input_controller_nominal[7])
    param_controller['Even']['y_at_midstance_nominal_slopeframe'] = float(p_input_controller_nominal[8])
    param_controller['Even']['sumy_at_midstance_nominal_slopeframe'] = float(p_input_controller_nominal[9])

    # simulate a handful of steps to compute steady-state energy
    param_fixed['num_steps'] = param_fixed['Learner']['numStepsPerIteration']
    (
        state_var0,
        t_store,
        state_store,
        emet_total_store,
        emet_per_time_store,
        t_step_store,
        ework_pushoff_store,
        ework_heelstrike_store,
    ) = simulate_many_steps_asymmetric_control(
        state_var0_model, param_controller, param_fixed, t_start
    )

    f_energy = np.sum(emet_total_store) / np.sum(t_step_store)
    t_end = float(t_store[-1][-1])

    # periodicity constraint on theta_dot and foot position
    f_constraint = float(
        np.linalg.norm(state_var0_model[1:3] - state_var0[1:3]) ** 2
    )

    # trade off energy and periodicity
    lambda_energy_vs_periodicity = param_fixed['lambdaEnergyVsPeriodicity']
    f_objective = (
        lambda_energy_vs_periodicity * f_energy
        + (1 - lambda_energy_vs_periodicity) * f_constraint
    )

    # symmetry constraint from the first two simulated steps
    angle_theta_end_1 = state_store[0][-1, 0]
    angle_theta_end_2 = state_store[1][-1, 0]
    f_symmetry = float((angle_theta_end_1 - angle_theta_end_2) ** 2)

    f_objective = (
        param_fixed['lambdaEnergyVsSymmetry'] * f_objective
        + (1 - param_fixed['lambdaEnergyVsSymmetry'])
        * f_symmetry
        * param_fixed['symmetryMultiplier']
    )

    return f_objective, state_var0, t_end, f_energy, f_constraint