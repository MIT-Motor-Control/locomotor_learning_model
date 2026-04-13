"""Return initial learnable controller parameters."""
from __future__ import annotations

import numpy as np


def load_learnable_parameters_initial(param_fixed: dict) -> np.ndarray:
    """Return baseline learnable controller parameters.

    Mirrors ``loadLearnableParametersInitial.m`` from MATLAB.
    """
    if param_fixed.get('swingCost', {}).get('Coeff') == 0.9:
        p_input_controller_asymmetric_nominal = np.array([
            0.328221262798818,
            0.310751796902254,
            0.153556843539029,
            0.0,
            0.0,
            0.328221491356562,
            0.310751694570805,
            0.153557221281688,
            -0.000000038953735,
            -0.000000038953735,
        ])
    else:
        raise NotImplementedError(
            "Only swingCost.Coeff=0.9 configuration supported"
        )
    return p_input_controller_asymmetric_nominal