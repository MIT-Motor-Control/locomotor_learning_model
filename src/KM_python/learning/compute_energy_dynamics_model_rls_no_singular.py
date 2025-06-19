from __future__ import annotations

from typing import Tuple
import numpy as np

from learning.rlsupdate import rlsupdate


def compute_energy_dynamics_model_rls_no_singular(
    adynamics_now: np.ndarray,
    state_var_now_store: np.ndarray,
    edot_store: np.ndarray,
    p_input_controller_now_store: np.ndarray,
    num_steps_to_use: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update the linear energy dynamics model using least squares.

    This is a line-by-line Python translation of the MATLAB function
    ``computeEnergyDynamicsModelRLS_NoSingular.m``.
    """

    beta = 0.5  # Unused but kept for API compatibility

    aold = adynamics_now

    mu_measurement = 0.0  # Dummy variable, not used

    input_now_store = np.vstack((state_var_now_store, p_input_controller_now_store))
    output_next_store = edot_store

    # Remove the first row of zeros to avoid singularity
    input_now_store = input_now_store[1:, :]

    adynamics_new, error_old_model, error_new_model = rlsupdate(
        aold,
        beta,
        input_now_store,
        output_next_store,
        mu_measurement,
        num_steps_to_use,
    )

    return adynamics_new, error_old_model, error_new_model