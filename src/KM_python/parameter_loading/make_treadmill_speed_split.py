"""Simple split-belt treadmill speed profile."""
from __future__ import annotations

import numpy as np


def make_treadmill_speed_split(param_fixed: dict) -> dict:
    """Return imposed belt speeds for a classic split-belt protocol."""
    L = 0.95
    g = 9.81
    time_scaling = np.sqrt(L / g)

    delta = 0.0328
    v_normal = -0.3276
    v_fast = v_normal - 5 * delta
    v_slow = v_normal + 5 * delta

    t_duration_transition = param_fixed.get('transitionTime', 15) / time_scaling

    speed_protocol = param_fixed.get('speedProtocol', 'classic split belt')
    if speed_protocol != 'classic split belt':
        raise NotImplementedError("Only 'classic split belt' protocol implemented")

    t_duration1 = 1 * 60 / time_scaling
    t_duration2 = 5 * 60 / time_scaling
    t_duration3 = 45 * 60 / time_scaling
    t_duration4 = 5 * 60 / time_scaling

    foot_speed1_store = np.array([
        v_normal,
        v_normal,
        v_normal,
        v_fast,
        v_normal,
    ])
    foot_speed2_store = np.array([
        v_normal,
        v_normal,
        v_normal,
        v_slow,
        v_normal,
    ])

    t_store = np.array([0, t_duration1, t_duration2, t_duration3, t_duration4])
    t_store = np.cumsum(t_store)

    t_list = [0.0]
    fs1 = [foot_speed1_store[0]]
    fs2 = [foot_speed2_store[0]]

    for i in range(1, len(t_store)):
        if i < len(t_store) - 1:
            t_list.extend([t_store[i], t_store[i] + t_duration_transition])
            fs1.extend([foot_speed1_store[i], foot_speed1_store[i + 1]])
            fs2.extend([foot_speed2_store[i], foot_speed2_store[i + 1]])
        else:
            t_list.append(t_store[i])
            fs1.append(foot_speed1_store[i])
            fs2.append(foot_speed2_store[i])

    t_arr = np.array(t_list)
    fs1_arr = np.array(fs1)
    fs2_arr = np.array(fs2)

    a1_list = np.diff(fs1_arr) / np.diff(t_arr)
    a2_list = np.diff(fs2_arr) / np.diff(t_arr)
    a1_arr = np.concatenate([a1_list, [0]])
    a2_arr = np.concatenate([a2_list, [0]])

    return {
        'tList': t_arr,
        'footSpeed1List': fs1_arr,
        'footSpeed2List': fs2_arr,
        'footAcc1List': a1_arr,
        'footAcc2List': a2_arr,
    }