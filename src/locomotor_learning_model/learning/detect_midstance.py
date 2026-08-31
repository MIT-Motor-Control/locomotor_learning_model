from __future__ import annotations

import numpy as np


def detect_midstance(
    t: float, state_var: np.ndarray, param_fixed, param_controller
) -> float:
    """Return the event value used to detect midstance.

    The value crosses zero when the body passes over the stance foot, allowing
    ``solve_ivp`` to stop integration at the midstance event.
    """

    angle_theta = state_var[0]

    y_now = param_fixed['leglength'] * np.sin(angle_theta + param_fixed['angleSlope'])
    value = y_now

    return float(value)
