"""Adaptation protocol settings."""
from __future__ import annotations

import numpy as np

from parameter_loading.make_treadmill_speed_split import make_treadmill_speed_split


def load_protocol_parameters(param_fixed: dict) -> dict:
    """Return protocol parameters for the simulation."""
    param_fixed['SplitOrTied'] = 'split'
    param_fixed['speedProtocol'] = 'classic split belt'
    param_fixed['transitionTime'] = 15

    param_fixed['imposedFootSpeeds'] = make_treadmill_speed_split(param_fixed)

    param_fixed['angleSlope'] = 0

    nominal_step_time = 1.7
    t_list = param_fixed['imposedFootSpeeds']['tList']
    param_fixed['numStepsToLearn'] = t_list[-1] / nominal_step_time
    param_fixed['numStepsToLearn'] = round(param_fixed['numStepsToLearn'] / 100) * 100

    learner = param_fixed.get('Learner', {})
    param_fixed['numIterations'] = int(np.floor(param_fixed['numStepsToLearn'] / learner.get('numStepsPerIteration', 1)))

    if learner.get('numStepsPerIteration', 1) % 2 != 0:
        learner['numStepsPerIteration'] = learner.get('numStepsPerIteration', 1) + 1
        param_fixed['Learner'] = learner

    return param_fixed