"""Interpolation of imposed treadmill belt speeds."""
from __future__ import annotations

import numpy as np
from typing import Mapping, Sequence, Tuple


def _interp_extrap(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear interpolation with extrapolation."""
    result = np.empty_like(t, dtype=float)
    idxs = np.searchsorted(x, t, side="left")
    for i, (ti, idx) in enumerate(zip(t, idxs)):
        if idx == 0:
            x1, x2 = x[0], x[1]
            y1, y2 = y[0], y[1]
        elif idx >= len(x):
            x1, x2 = x[-2], x[-1]
            y1, y2 = y[-2], y[-1]
        else:
            x1, x2 = x[idx - 1], x[idx]
            y1, y2 = y[idx - 1], y[idx]
        result[i] = y1 + (y2 - y1) * (ti - x1) / (x2 - x1)
    return result


def get_treadmill_speed(
    t: Sequence[float] | float,
    belt_speeds_imposed: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the belt speeds corresponding to time ``t``.

    Mirrors ``getTreadmillSpeed.m`` from MATLAB.
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    foot_speed1 = _interp_extrap(
        t_arr,
        np.asarray(belt_speeds_imposed["tList"], dtype=float),
        np.asarray(belt_speeds_imposed["footSpeed1List"], dtype=float),
    )
    foot_speed2 = _interp_extrap(
        t_arr,
        np.asarray(belt_speeds_imposed["tList"], dtype=float),
        np.asarray(belt_speeds_imposed["footSpeed2List"], dtype=float),
    )
    if foot_speed1.size == 1:
        return foot_speed1.item(), foot_speed2.item()
    return foot_speed1, foot_speed2