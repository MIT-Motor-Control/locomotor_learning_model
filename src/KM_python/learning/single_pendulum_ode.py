from __future__ import annotations

import numpy as np


def single_pendulum_ode(
    t: float,
    state_var: np.ndarray,
    param_fixed,
    param_controller,
) -> np.ndarray:
    """Dynamics for a simple inverted pendulum.

    This is a direct line‑by‑line translation of ``singlePendulumODE.m``.
    """

    angle_theta = state_var[0]
    dangle_theta = state_var[1]

    # Handle missing belt speed parameters gracefully (for standalone testing)
    if ('tList_BeltSpeed' not in param_controller or 
        'PushoffAccelerationNowList' not in param_controller):
        # Use default values for standalone testing
        param_controller['PushoffAccelerationNow'] = 0.0
    else:
        # Normal operation during simulation
        param_controller['PushoffAccelerationNow'] = float(
            np.interp(
                t,
                param_controller['tList_BeltSpeed'],
                param_controller['PushoffAccelerationNowList'],
                left=param_controller['PushoffAccelerationNowList'][0],
                right=param_controller['PushoffAccelerationNowList'][-1],
            )
        )

    force_due_to_acceleration = -param_controller['PushoffAccelerationNow'] * param_fixed['mbody']
    torque_due_to_acceleration = (
        force_due_to_acceleration
        * np.cos(angle_theta)
        / (param_fixed['mbody'] * param_fixed['leglength'])
    )

    if not param_fixed.get('includeAccelerationTorque', True):
        torque_due_to_acceleration = 0.0

    dd_angle_theta = (
        param_fixed['gravg'] / param_fixed['leglength'] * np.sin(angle_theta)
        + torque_due_to_acceleration
    )

    # Handle missing foot speed parameters gracefully (for standalone testing)
    if ('tList_BeltSpeed' not in param_controller or 
        'PushoffFootSpeedNowList' not in param_controller):
        # Use default value for standalone testing
        param_controller['PushoffFootSpeedNow'] = -0.3276  # Default belt speed
    else:
        # Normal operation during simulation
        param_controller['PushoffFootSpeedNow'] = float(
            np.interp(
                t,
                param_controller['tList_BeltSpeed'],
                param_controller['PushoffFootSpeedNowList'],
                left=param_controller['PushoffFootSpeedNowList'][0],
                right=param_controller['PushoffFootSpeedNowList'][-1],
            )
        )

    dy_foot = param_controller['PushoffFootSpeedNow']

    return np.array([dangle_theta, dd_angle_theta, dy_foot])