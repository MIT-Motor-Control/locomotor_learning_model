from __future__ import annotations

import numpy as np


def compute_steady_energy_gradient_v2_no_singular(
    adynamics_model: np.ndarray,
    adynamics_energy: np.ndarray,
    p_input_now: np.ndarray,
    num_learning_dimensions: int,
    num_state_dimensions: int,
    include_internal_model: bool,
) -> np.ndarray:
    """Compute the steady energy gradient without singularities.

    This is a direct line-by-line translation of the MATLAB function
    ``computeSteadyEnergyGradientV2_NoSingular.m``.  It returns the gradient of
    the steady-state energy with respect to the learnable control parameters.
    ``p_input_now`` is unused but retained for API compatibility.
    """

    # because we removed one state to avoid singular matrices
    num_state_dimensions = num_state_dimensions - 1

    inow = np.eye(num_state_dimensions)

    # state dynamics: x_{i+1} = A x_i + B p_i + C
    a = adynamics_model[0:num_state_dimensions, 0:num_state_dimensions]
    b = adynamics_model[
        0:num_state_dimensions,
        num_state_dimensions : num_state_dimensions + num_learning_dimensions,
    ]

    # energy dynamics: E_i = G x_i + H p_i + K
    g = adynamics_energy[0, 0:num_state_dimensions]
    h = adynamics_energy[
        0,
        num_state_dimensions : num_state_dimensions + num_learning_dimensions,
    ]

    g_energy = (
        g @ np.linalg.solve(inow - a, b) * float(include_internal_model) + h
    ).reshape(-1, 1)

    return g_energy