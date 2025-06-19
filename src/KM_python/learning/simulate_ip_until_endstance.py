from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp

from learning.single_pendulum_ode import single_pendulum_ode
from learning.detect_endstance import detect_endstance


def simulate_ip_until_endstance(
    state_var0: np.ndarray,
    tspan: np.ndarray,
    param_fixed,
    param_controller,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the inverted pendulum dynamics until endstance."""

    def ode(t, y):
        return single_pendulum_ode(t, y, param_fixed, param_controller)

    def event(t, y):
        return detect_endstance(t, y, param_fixed, param_controller)

    event.terminal = True
    event.direction = 1.0

    sol = solve_ivp(
        ode,
        (float(tspan[0]), float(tspan[-1])),
        state_var0,
        events=event,
        rtol=1e-10,
        atol=1e-10,
        dense_output=False,
    )

    return sol.t, sol.y.T