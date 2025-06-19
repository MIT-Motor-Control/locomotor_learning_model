from __future__ import annotations

import numpy as np


def detect_endstance(
    t: float, state_var: np.ndarray, param_fixed, param_controller
) -> float:
    """Event function for endstance detection.

    Returns the quantity that becomes zero at endstance. Used with
    ``solve_ivp`` to stop integration when the stance leg reaches the
    desired angle.
    """

    angle_theta = state_var[0]
    value = angle_theta - param_controller.theta_end_thisStep
    return float(value)