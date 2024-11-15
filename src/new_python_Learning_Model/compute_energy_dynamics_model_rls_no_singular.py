import numpy as np
from rls_update import rls_update

def compute_energy_dynamics_model_rls_no_singular(Adynamics_now, state_var_now_store, edot_store, p_input_controller_now_store, num_steps_to_use):
    """
    Computes a linear model that predicts energy for the stride from input state and control parameters.

    Args:
        Adynamics_now (ndarray): Current dynamics matrix.
        state_var_now_store (ndarray): Stored current state variables.
        edot_store (ndarray): Stored energy rate data.
        p_input_controller_now_store (ndarray): Stored controller input variables.
        num_steps_to_use (int): Number of steps to use for the computation.

    Returns:
        tuple: Updated dynamics matrix, error of the old model, and error of the new model.
    """
    # Parameters
    beta = 0.5  # Not used, can be anything
    Aold = Adynamics_now
    mu_measurement = 0  # Dummy variable, not used

    # Prepare input and output data
    input_now_store = np.vstack((state_var_now_store, p_input_controller_now_store))
    output_next_store = edot_store

    # Remove the first row to avoid singularity
    input_now_store = input_now_store[1:, :]

    # Update dynamics matrix using RLS update
    Adynamics_new, error_old_model, error_new_model = rls_update(Aold, beta, input_now_store, output_next_store, mu_measurement, num_steps_to_use)

    return Adynamics_new, error_old_model, error_new_model

