from __future__ import annotations

import numpy as np


def swing_cost_doke(
    t_duration_of_change: float,
    v_initial: float,
    v_final: float,
    v_body: float,
    param_fixed,
) -> float:
    """Approximate leg swing cost following Doke & Kuo 2005.

    This is a direct translation of ``swingCostDoke.m`` which assumes the cost
    is related to the force rate needed to change leg swing velocity.
    """

    v_initial_relative = v_initial - v_body
    v_final_relative = v_final - v_body

    delta_v = v_final_relative - v_initial_relative
    force = param_fixed['mFoot'] * delta_v / t_duration_of_change
    force_rate = force / t_duration_of_change

    epsilon = 0.01

    c = (
        param_fixed['swingCost']['Coeff']
        * np.sqrt(force_rate**2 + epsilon**2) ** param_fixed['swingCost']['alpha']
    )

    return float(c)