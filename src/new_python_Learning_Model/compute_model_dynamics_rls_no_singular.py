import numpy as np

from rls_update import rls_update

def compute_model_dynamics_rls_no_singular(Adynamics_now, state_var_now_store, state_var_next_store, p_input_controller_now_store, num_steps_to_use):
    """
    Computes a linear dynamic model by computing the gradient of state with respect to state and action (internal model).

    Args:
        Adynamics_now (ndarray): Current dynamics matrix.
        state_var_now_store (ndarray): Stored current state variables.
        state_var_next_store (ndarray): Stored next state variables.
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
    output_next_store = state_var_next_store

    # Remove the first row to avoid singularity
    input_now_store = input_now_store[1:, :]
    output_next_store = output_next_store[1:, :]

    # Update dynamics matrix using RLS update
    Adynamics_new, error_old_model, error_new_model = rls_update(Aold, beta, input_now_store, output_next_store, mu_measurement, num_steps_to_use)

    return Adynamics_new, error_old_model, error_new_model


