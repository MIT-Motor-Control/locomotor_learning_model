"""Optional override for simulation length."""

from __future__ import annotations

import numpy as np


def load_how_long_parameters(param_fixed: dict) -> dict:
    """Override the default simulation duration."""
    param_fixed["numStepsToLearn"] = 2000

    learner = param_fixed["Learner"]
    num_iterations = int(
        np.floor(param_fixed["numStepsToLearn"] / learner["numStepsPerIteration"])
    )
    param_fixed["num_iterations"] = num_iterations
    param_fixed["numIterations"] = num_iterations

    if learner["numStepsPerIteration"] % 2 != 0:
        learner["numStepsPerIteration"] = learner["numStepsPerIteration"] + 1

    return param_fixed
