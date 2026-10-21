from __future__ import annotations

import numpy as np


def error_from_memory_compute(
    slope_j_controller_vs_context: np.ndarray,
    p_input_now_considered_good: np.ndarray,
    param_fixed,
    context_gait_now: np.ndarray,
) -> float:
    """Return squared error between context-predicted and successful controller parameters."""

    p_input_memory_now = (
        param_fixed['storedmemory']['nominalControl']
        + slope_j_controller_vs_context
        @ (context_gait_now - param_fixed['storedmemory']['nominalContext'])
    )

    diff = p_input_now_considered_good - p_input_memory_now
    
    diff_flat = np.asarray(diff).flatten()
    
    f = float(np.sum(diff_flat * diff_flat))

    return f
