from __future__ import annotations

from typing import Tuple
import numpy as np

from locomotor_learning_model.learning.rlsupdate import rlsupdate


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

    # Ensure arrays have the same number of columns by taking the minimum
    min_cols = min(state_var_now_store.shape[1], p_input_controller_now_store.shape[1])
    
    state_var_trimmed = state_var_now_store[:, :min_cols]
    p_input_trimmed = p_input_controller_now_store[:, :min_cols]
    
    input_now_store = np.vstack((state_var_trimmed, p_input_trimmed))
    
    # Ensure output_next_store is 2D and has the right number of columns
    output_next_store = np.atleast_2d(edot_store)
    if output_next_store.shape[0] == 1:
        # If it's a row vector, we want it as a row for compatibility with column indexing
        pass
    else:
        # If it's a column vector, transpose to row
        output_next_store = output_next_store.T
    
    # Trim to match the minimum columns
    output_next_store = output_next_store[:, :min_cols]

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