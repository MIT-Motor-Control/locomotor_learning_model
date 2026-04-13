"""Initial mid-stance body state loader."""
from __future__ import annotations

import numpy as np
from typing import Sequence


def load_initial_body_state(p_input_controller_asymmetric_nominal: Sequence[float]) -> np.ndarray:
    """Return the initial mid-stance state of the model.

    Mirrors ``loadInitialBodyState.m`` from MATLAB.
    """
    v_swing_initial = 0.35
    state_var0_model = np.array([
        0.0,  # angleTheta0 - stance leg angle
        p_input_controller_asymmetric_nominal[1],  # dAngleTheta0 - stance leg angular rate
        0.0,  # yFoot0 in lab frame
        0.0,  # sum of yFoot in lab frame
        v_swing_initial,
    ])
    return state_var0_model