from __future__ import annotations

from typing import Tuple

import numpy as np

from learning.rlsupdate import rlsupdate


def compute_model_dynamics_model_rls_no_singular(
    adynamics_now: np.ndarray,
    state_var_now_store: np.ndarray,
    state_var_next_store: np.ndarray,
    p_input_controller_now_store: np.ndarray,
    num_steps_to_use: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update the internal linear dynamics model using least squares.

    This is a direct line-by-line translation of the MATLAB function
    ``computeModelDynamicsModelRLS_NoSingular.m``.
    """

    beta = 0.5  # Unused but kept for API compatibility

    aold = adynamics_now

    mu_measurement = 0.0  # Dummy variable, not used

    # Ensure both arrays have the same number of columns by taking the minimum
    min_cols = min(state_var_now_store.shape[1], p_input_controller_now_store.shape[1], 
                   state_var_next_store.shape[1])
    
    state_var_trimmed = state_var_now_store[:, :min_cols]
    p_input_trimmed = p_input_controller_now_store[:, :min_cols]
    state_next_trimmed = state_var_next_store[:, :min_cols]
    
    input_now_store = np.vstack((state_var_trimmed, p_input_trimmed))
    output_next_store = state_next_trimmed

    # Remove the first row of zeros to avoid singularity
    input_now_store = input_now_store[1:, :]
    output_next_store = output_next_store[1:, :]

    adynamics_new, error_old_model, error_new_model = rlsupdate(
        aold,
        beta,
        input_now_store,
        output_next_store,
        mu_measurement,
        num_steps_to_use,
    )

    return adynamics_new, error_old_model, error_new_model