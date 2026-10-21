from __future__ import annotations

import numpy as np


def single_pendulum_ode(
    t: float,
    state_var: np.ndarray,
    param_fixed,
    param_controller,
) -> np.ndarray:
    """Evaluate simple inverted-pendulum stance dynamics."""

    angle_theta = state_var[0]
    dangle_theta = state_var[1]

    if ('tList_BeltSpeed' not in param_controller or 
        'PushoffAccelerationNowList' not in param_controller):
        param_controller['PushoffAccelerationNow'] = 0.0
    else:
        # Hold acceleration constant within each belt-schedule segment;
        # linear interpolation would introduce torque during constant-speed phases.
        _t_knots = param_controller['tList_BeltSpeed']
        _acc_list = param_controller['PushoffAccelerationNowList']
        _idx = int(np.searchsorted(_t_knots, t, side='right')) - 1
        _idx = min(max(_idx, 0), len(_acc_list) - 1)
        param_controller['PushoffAccelerationNow'] = float(_acc_list[_idx])

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

    if ('tList_BeltSpeed' not in param_controller or 
        'PushoffFootSpeedNowList' not in param_controller):
        param_controller['PushoffFootSpeedNow'] = -0.3276
    else:
        # Belt velocity varies linearly during each transition ramp.
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
