from __future__ import annotations

import numpy as np


def detect_midstance(
    t: float, state_var: np.ndarray, param_fixed, param_controller
) -> float:
    """Event function for midstance detection.

    This is a direct line-by-line translation of ``DetectMidstance.m`` and
    returns the quantity that becomes zero at midstance. It is used with
    ``solve_ivp`` to stop integration when the body passes over the stance foot.
    """

    angle_theta = state_var[0]

    y_now = param_fixed['leglength'] * np.sin(angle_theta + param_fixed['angleSlope'])
    value = y_now

    return float(value)