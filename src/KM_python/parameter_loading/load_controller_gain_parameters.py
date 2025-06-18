"""Feedback controller gain loader."""
from __future__ import annotations


def load_controller_gain_parameters(param_fixed: dict) -> dict:
    """Return feedback controller gains.

    Mirrors ``loadControllerGainParameters.m`` from the MATLAB code.
    """
    param_controller = {}

    swing_coeff = param_fixed.get('swingCost', {}).get('Coeff')
    if swing_coeff == 0.9:
        param_controller['pushoff_gain_ydot'] = -0.521088310893437
        param_controller['legAngle_gain_ydot'] = 0.279308152405233
        param_controller['pushoff_gain_y'] = -0.085939152240409
        param_controller['legAngle_gain_y'] = -0.028999999999996
        param_controller['pushoff_gain_SUMy'] = -0.007302228127922
        param_controller['legAngle_gain_SUMy'] = 0.002019507100848
        param_controller['legAngle_gain_BeltSpeed'] = -0.075044622818653
        param_controller['pushoff_gain_BeltSpeed'] = -1.055930819923483

    param_controller['ControllerBeltFrameVsLabFrame'] = 1

    return param_controller