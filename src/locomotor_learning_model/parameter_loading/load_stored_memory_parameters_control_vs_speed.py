"""Initial stored controller memory."""
from __future__ import annotations

import numpy as np


def load_stored_memory_parameters_control_vs_speed(param_fixed: dict) -> dict:
    """Return dictionary with stored memory settings."""
    stored = {}
    if param_fixed.get('swingCost', {}).get('Coeff') == 0.9:
        stored['nominalControl'] = np.array([
            0.328221262798818,
            0.310751796902254,
            0.153556843539029,
            0.328221491356562,
            0.310751694570805,
            0.153557221281688,
            -0.000000038953735,
            -0.000000038953735,
        ])
    stored['nominalContext'] = np.array([-0.35, -0.35])
    stored['controlSlopeVsContext'] = np.zeros((8, 2))
    param_fixed['storedmemory'] = stored
    return param_fixed