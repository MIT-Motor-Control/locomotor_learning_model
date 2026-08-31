from __future__ import annotations

import numpy as np


def error_from_memory_compute(
    slope_j_controller_vs_context: np.ndarray,
    p_input_now_considered_good: np.ndarray,
    param_fixed,
    context_gait_now: np.ndarray,
) -> float:
    """Compute squared prediction error between memory and controller state.

    The memory model predicts controller parameters from the current gait
    context. This returns the squared distance between that prediction and the
    controller parameters currently considered successful.
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
