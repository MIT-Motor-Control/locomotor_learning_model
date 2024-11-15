import numpy as np
from error_from_memory_compute import error_from_memory_compute


def gradient_of_error_from_memory_compute(slope_j_controller_vs_context, p_input_now_considered_good, param_fixed, context_gait_now):
    """
    Computes the gradient of the memory update based on the prediction error between
    the memory function approximation and the current state/sensing.

    Args:
        slope_j_controller_vs_context (ndarray): Slope of the controller versus context.
        p_input_now_considered_good (ndarray): Current input parameters considered good.
        param_fixed (dict): Fixed parameters for the simulation.
        context_gait_now (ndarray): Current gait context.

    Returns:
        ndarray: Gradient of the error with respect to memory parameters.
    """
    # Initial error computation
    f0 = error_from_memory_compute(slope_j_controller_vs_context, p_input_now_considered_good, param_fixed, context_gait_now)

    # Small step size for numerical gradient computation
    h = 1e-4

    # Initialize gradient matrix
    g = np.zeros_like(slope_j_controller_vs_context)

    # Compute gradient using finite differences
    for i_count in range(slope_j_controller_vs_context.shape[0]):
        for j_count in range(slope_j_controller_vs_context.shape[1]):
            slope_j_controller_vs_context_now = slope_j_controller_vs_context.copy()
            slope_j_controller_vs_context_now[i_count, j_count] += h

            f = error_from_memory_compute(slope_j_controller_vs_context_now, p_input_now_considered_good, param_fixed, context_gait_now)

            g[i_count, j_count] = (f - f0) / h

    return g


