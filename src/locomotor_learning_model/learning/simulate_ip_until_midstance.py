from __future__ import annotations

from typing import Tuple
import numpy as np
from scipy.integrate import solve_ivp

from locomotor_learning_model.learning.single_pendulum_ode import single_pendulum_ode
from locomotor_learning_model.learning.detect_midstance import detect_midstance


def simulate_ip_until_midstance(
    state_var0: np.ndarray,
    tspan: np.ndarray,
    param_fixed,
    param_controller,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integrate the inverted pendulum dynamics until midstance."""

    def ode(t, y):
        return single_pendulum_ode(t, y, param_fixed, param_controller)

    def event(t, y):
        return detect_midstance(t, y, param_fixed, param_controller)

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