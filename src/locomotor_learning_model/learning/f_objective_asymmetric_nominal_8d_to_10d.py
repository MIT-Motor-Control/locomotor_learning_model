from __future__ import annotations

from typing import Tuple
import numpy as np

from locomotor_learning_model.learning.f_objective_asymmetric_nominal import f_objective_asymmetric_nominal


def f_objective_asymmetric_nominal_8d_to_10d(
    p_input_controller_nominal: np.ndarray,
    state_var0_model: np.ndarray,
    param_controller,
    param_fixed,
    t_start: float,
) -> Tuple[float, np.ndarray, float, float, float]:
    """Expand an 8D learned controller vector into the 10D controller state."""
    # Ensure input is 1D (flatten if needed)
    p_input_flat = np.asarray(p_input_controller_nominal).flatten()
    
    temp = np.zeros(10)
    temp[0:3] = p_input_flat[0:3]
    temp[5:10] = p_input_flat[3:8]
    p_input_controller_nominal_full = temp
    return f_objective_asymmetric_nominal(
        p_input_controller_nominal_full,
        state_var0_model,
        param_controller,
        param_fixed,
        t_start,
    )
