"""Utilities for configuring sensory noise parameters."""
from __future__ import annotations


def load_sensory_noise_parameters(param_fixed: dict | None = None) -> dict:
    """Return dictionary with sensory noise settings.

    Mirrors ``loadSensoryNoiseParameters.m`` from the MATLAB code.
    """
    if param_fixed is None:
        param_fixed = {}

    # multiplicative noise for energy measurements
    param_fixed['noiseEnergySensory'] = 0.0001  # this is multiplicative noise
    # param_fixed['noiseEnergySensory'] = 0.01  # this is multiplicative noise
    # param_fixed['noiseEnergySensory'] = 0.00  # this is multiplicative noise

    # additive noise on velocity measurements
    param_fixed['velocitySensoryNoise'] = 0
    # this is additive noise

    return param_fixed