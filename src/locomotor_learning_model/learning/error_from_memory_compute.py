from __future__ import annotations

import numpy as np


def error_from_memory_compute(
    slope_j_controller_vs_context: np.ndarray,
    p_input_now_considered_good: np.ndarray,
    param_fixed,
    context_gait_now: np.ndarray,
) -> float:
    """Compute the prediction error between memory and controller.

    This is a line-by-line translation of the MATLAB function
    ``errorFromMemoryCompute.m``.  The routine evaluates the discrepancy between
    the controller parameters that were deemed good and the controller values
    predicted by the stored memory at the current context.
    """

    p_input_memory_now = (
        param_fixed['storedmemory']['nominalControl']
        + slope_j_controller_vs_context
        @ (context_gait_now - param_fixed['storedmemory']['nominalContext'])
    )

    diff = p_input_now_considered_good - p_input_memory_now
    
    # Ensure diff is 1D for proper dot product
    diff_flat = np.asarray(diff).flatten()
    
    # Compute scalar dot product (sum of squared differences)
    f = float(np.sum(diff_flat * diff_flat))

    return f