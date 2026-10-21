from __future__ import annotations


def load_biped_model_parameters(param_fixed: dict | None = None) -> dict:
    """Return fixed biped model parameters."""
    if param_fixed is None:
        param_fixed = {}

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

    # Optional local stability penalty; disabled in the baseline objective.
    param_fixed['fallPenaltyWeight'] = 0.0
    param_fixed['fallPenaltyMode'] = 'com_margin'
    param_fixed['fallPenaltyPower'] = 2.0
    param_fixed['fallPenaltyEpsilon'] = 1e-6
    # COM-height thresholds are dimensionless; the lower bound saturates the penalty.
    param_fixed['fallSafeComHeight'] = 0.95
    param_fixed['fallGroundComHeight'] = 0.90
    param_fixed['fallCapturePointLimit'] = 1.0
    param_fixed['fallCapturePointScale'] = 1.0
    param_fixed['fallPenaltyScaleToEnergy'] = False

    return param_fixed
