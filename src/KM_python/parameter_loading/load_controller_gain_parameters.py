"""Feedback controller gain loader."""
from __future__ import annotations


def load_controller_gain_parameters(param_fixed: dict) -> dict:
    """Return feedback controller gains.

    Mirrors ``loadControllerGainParameters.m`` from the MATLAB code.
    """
    param_controller = {}

    swing_coeff = param_fixed.get('swingCost', {}).get('Coeff')
    if swing_coeff == 0.9:
        gains = {
            'pushoff_gain_ydot': -0.521088310893437,
            'legAngle_gain_ydot': 0.279308152405233,
            'pushoff_gain_y': -0.085939152240409,
            'legAngle_gain_y': -0.028999999999996,
            'pushoff_gain_SUMy': -0.007302228127922,
            'legAngle_gain_SUMy': 0.002019507100848,
            'legAngle_gain_BeltSpeed': -0.075044622818653,
            'pushoff_gain_BeltSpeed': -1.055930819923483,
        }
        # Odd and Even start with the same values, will be overwritten in f_objective_asymmetric_nominal
        param_controller['Odd'] = {
            'theta_end_nominal': 0.0,
            'ydot_at_midstance_nominal_beltframe': 0.0,
            'pushoff_impulse_magnitude_nominal': 0.0,
            'y_at_midstance_nominal_slopeframe': 0.0,
            'sumy_at_midstance_nominal_slopeframe': 0.0,
        }
        param_controller['Even'] = {
            'theta_end_nominal': 0.0,
            'ydot_at_midstance_nominal_beltframe': 0.0,
            'pushoff_impulse_magnitude_nominal': 0.0,
            'y_at_midstance_nominal_slopeframe': 0.0,
            'sumy_at_midstance_nominal_slopeframe': 0.0,
        }
        param_controller.update(gains)
    param_controller['ControllerBeltFrameVsLabFrame'] = 1

    return param_controller