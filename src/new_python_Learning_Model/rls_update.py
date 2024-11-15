import numpy as np

def rls_update(Aold, beta, input_now_store, output_next_store, mu_measurement, num_steps_to_use):
    """
    Performs 'rolling least squares' to build a linear model between inputs and outputs.

    Args:
        Aold (ndarray): Previous dynamics matrix.
        beta (float): Forgetting factor (not used in this version).
        input_now_store (ndarray): Stored input data.
        output_next_store (ndarray): Stored output data.
        mu_measurement (float): Measurement noise (not used in this version).
        num_steps_to_use (int): Number of steps to use for the regression.

    Returns:
        tuple: Updated dynamics matrix, error of the old model, and error of the new model.
    """
    # Dimensions
    num_input_dimensions = input_now_store.shape[0]
    num_output_dimensions = output_next_store.shape[0]

    # Pick only a certain number of steps: each column is a step
    input_now_store = input_now_store[:, -num_steps_to_use:]
    output_next_store = output_next_store[:, -num_steps_to_use:]

    # Transpose so that each row is a step
    input_now_store = input_now_store.T
    output_next_store = output_next_store.T

    # Add a column of ones for the constant term
    input_now_store = np.hstack((input_now_store, np.ones((input_now_store.shape[0], 1))))

    # Perform least squares regression
    Anew = np.linalg.lstsq(input_now_store, output_next_store, rcond=None)[0]

    # Find the error on all steps and then the most recent step using the pre-change
    Aold = Aold.T
    error_old_linear_all_steps = output_next_store - input_now_store @ Aold
    error_old_linear_most_recent_step = error_old_linear_all_steps[-1, :].reshape(-1, 1)

    error_new_linear_all_steps = output_next_store - input_now_store @ Anew
    error_new_linear_most_recent_step = error_new_linear_all_steps[-1, :].reshape(-1, 1)

    # Transpose Anew so steps are columns instead of rows
    Anew = Anew.T

    return Anew, error_old_linear_most_recent_step, error_new_linear_most_recent_step

