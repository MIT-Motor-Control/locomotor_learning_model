import numpy as np

def error_from_memory_compute(slope_j_controller_vs_context, p_input_now_considered_good, param_fixed, context_gait_now):
    """
    Computes the prediction error between the memory function approximation and the current state/sensing.

    Args:
        slope_j_controller_vs_context (ndarray): Slope of the controller versus context.
        p_input_now_considered_good (ndarray): Current input parameters considered good.
        param_fixed (dict): Fixed parameters for the simulation.
        context_gait_now (ndarray): Current gait context.

    Returns:
        float: The prediction error.
    """
    # Compute current memory prediction
    p_input_memory_now = param_fixed['storedmemory']['nominalControl'] + \
        slope_j_controller_vs_context @ (context_gait_now - param_fixed['storedmemory']['nominalContext'])

    # Compute prediction error
    f = (p_input_now_considered_good - p_input_memory_now).T @ (p_input_now_considered_good - p_input_memory_now)

    return f
