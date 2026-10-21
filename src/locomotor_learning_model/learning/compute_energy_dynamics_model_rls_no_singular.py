from __future__ import annotations

import numpy as np

from locomotor_learning_model.learning.rlsupdate import rlsupdate


def compute_energy_dynamics_model_rls_no_singular(
    adynamics_now: np.ndarray,
    state_var_now_store: np.ndarray,
    edot_store: np.ndarray,
    p_input_controller_now_store: np.ndarray,
    num_steps_to_use: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update the linear energy dynamics model using recent stride data."""

    beta = 0.5

    aold = adynamics_now

    mu_measurement = 0.0

    min_cols = min(state_var_now_store.shape[1], p_input_controller_now_store.shape[1])
    
    state_var_trimmed = state_var_now_store[:, :min_cols]
    p_input_trimmed = p_input_controller_now_store[:, :min_cols]
    
    input_now_store = np.vstack((state_var_trimmed, p_input_trimmed))
    
    # Store energy observations as a row vector, with one column per stride.
    output_next_store = np.atleast_2d(edot_store)
    if output_next_store.shape[0] != 1:
        output_next_store = output_next_store.T
    
    output_next_store = output_next_store[:, :min_cols]

    # Midstance fixes the first state component at zero.
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
