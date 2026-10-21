"""Utilities for configuring sensory noise parameters."""
from __future__ import annotations


def load_sensory_noise_parameters(param_fixed: dict | None = None) -> dict:
    """Return sensory noise settings."""
    if param_fixed is None:
        param_fixed = {}

    # multiplicative noise for energy measurements
    param_fixed['noiseEnergySensory'] = 0.0001

    # additive noise on velocity measurements
    param_fixed['velocitySensoryNoise'] = 0

    return param_fixed
