from __future__ import annotations


def load_biped_model_parameters(param_fixed: dict | None = None) -> dict:
    """Return fixed biped model parameters."""
    if param_fixed is None:
        param_fixed = {}

    # Whether to include treadmill acceleration torque in the dynamics
    param_fixed['includeAccelerationTorque'] = 1

    # Body parameters (nondimensionalized)
    param_fixed['mbody'] = 1
    param_fixed['leglength'] = 1
    param_fixed['gravg'] = 1

    # Efficiency of positive and negative work
    param_fixed['efficiency_neg'] = 1.2
    param_fixed['efficiency_pos'] = 0.25
    param_fixed['bPos'] = 1 / param_fixed['efficiency_pos']
    param_fixed['bNeg'] = 1 / param_fixed['efficiency_neg']

    # Swing leg energy cost parameters
    param_fixed['mFoot'] = 0.05
    param_fixed['swingCost'] = {
        'Coeff': 0.9,
        'alpha': 1.0,
    }

    # Weighting between energy and periodicity objectives
    param_fixed['lambdaEnergyVsPeriodicity'] = 1

    # Weighting between energy and symmetry objectives
    param_fixed['lambdaEnergyVsSymmetry'] = 0.75
    param_fixed['symmetryMultiplier'] = 10

    # Optional explicit fall-risk reward term. A zero weight
    # preserves the original energy/symmetry objective exactly. See
    # locomotor_learning_model.learning.fall_margin_penalty for the definitions.
    param_fixed['fallPenaltyWeight'] = 0.0            # w; 0 => baseline model
    param_fixed['fallPenaltyMode'] = 'com_margin'     # primary local penalty
    param_fixed['fallPenaltyPower'] = 2.0
    param_fixed['fallPenaltyEpsilon'] = 1e-6
    # COM-margin normalization, pre-registered from the baseline per-step min-COM
    # range (~0.90-0.95 across conditions). The margin is 0 above fallSafeComHeight
    # and saturates at 1 by fallGroundComHeight. w is a free scale, so results are
    # invariant to this choice up to a rescaling of w.
    param_fixed['fallSafeComHeight'] = 0.95           # margin starts below this COM height
    param_fixed['fallGroundComHeight'] = 0.90         # margin saturates (=1) at/below this
    # capture_point (robustness alternative) parameters
    param_fixed['fallCapturePointLimit'] = 1.0
    param_fixed['fallCapturePointScale'] = 1.0
    param_fixed['fallPenaltyScaleToEnergy'] = False

    return param_fixed
