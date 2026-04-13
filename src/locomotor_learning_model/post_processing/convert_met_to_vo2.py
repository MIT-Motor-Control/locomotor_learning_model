"""Convert stride-wise metabolic rate into an indirect-calorimetry-like signal."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def convert_met_to_vo2(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Mirror MATLAB's first-order VO2 smoothing model."""
    time_constant = 44.0
    grav = 9.81
    leg_length = 0.95
    time_scaling = np.sqrt(leg_length / grav)
    time_constant = time_constant / time_scaling
    gain = 1.0 / time_constant

    t_span = np.asarray(params["tList"], dtype=float)
    emet_rate_list = np.asarray(params["EmetRateList"], dtype=float)
    initial_window = min(30, len(emet_rate_list))
    emet_vo2_initial = float(np.mean(emet_rate_list[:initial_window]))

    def ode_met_to_vo2(t: float, emet_vo2: np.ndarray) -> np.ndarray:
        emet_now = np.interp(t, t_span, emet_rate_list)
        return np.array([gain * (emet_now - emet_vo2[0])])

    solution = solve_ivp(
        ode_met_to_vo2,
        (float(t_span[0]), float(t_span[-1])),
        np.array([emet_vo2_initial]),
        t_eval=t_span,
        rtol=1e-8,
        atol=1e-8,
    )
    return solution.t, solution.y[0]
