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

    # Inertial force for acceleration
    param_controller.PushoffAccelerationNow = float(
        np.interp(
            t,
            param_controller.tList_BeltSpeed,
            param_controller.PushoffAccelerationNowList,
            left=param_controller.PushoffAccelerationNowList[0],
            right=param_controller.PushoffAccelerationNowList[-1],
        )
    )

    force_due_to_acceleration = -param_controller.PushoffAccelerationNow * param_fixed.mbody
    torque_due_to_acceleration = (
        force_due_to_acceleration
        * np.cos(angle_theta)
        / (param_fixed.mbody * param_fixed.leglength)
    )

    if not getattr(param_fixed, "include_acceleration_torque", True):
        torque_due_to_acceleration = 0.0

    dd_angle_theta = (
        param_fixed.gravg / param_fixed.leglength * np.sin(angle_theta)
        + torque_due_to_acceleration
    )

    param_controller.PushoffFootSpeedNow = float(
        np.interp(
            t,
            param_controller.tList_BeltSpeed,
            param_controller.PushoffFootSpeedNowList,
            left=param_controller.PushoffFootSpeedNowList[0],
            right=param_controller.PushoffFootSpeedNowList[-1],
        )
    )

    dy_foot = param_controller.PushoffFootSpeedNow

    return np.array([dangle_theta, dd_angle_theta, dy_foot])