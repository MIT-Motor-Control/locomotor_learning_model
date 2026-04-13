"""Adaptation protocol settings."""
from __future__ import annotations

import numpy as np

from locomotor_learning_model.parameter_loading.make_treadmill_speed_tied import (
    make_treadmill_speed_tied,
)
from locomotor_learning_model.parameter_loading.make_treadmill_speed_split import make_treadmill_speed_split


def load_protocol_parameters(
    param_fixed: dict,
    split_or_tied: str | None = None,
    speed_protocol: str | None = None,
    transition_time: float | None = None,
) -> dict:
    """Return protocol parameters for the simulation."""
    split_or_tied = split_or_tied or param_fixed.get("SplitOrTied", "split")

    if split_or_tied == "split":
        param_fixed["SplitOrTied"] = "split"
        param_fixed["speedProtocol"] = (
            speed_protocol
            or param_fixed.get("speedProtocol")
            or "classic split belt"
        )
        param_fixed["transitionTime"] = (
            transition_time
            if transition_time is not None
            else param_fixed.get("transitionTime", 15)
        )
        param_fixed["imposedFootSpeeds"] = make_treadmill_speed_split(param_fixed)
    elif split_or_tied == "tied":
        param_fixed["SplitOrTied"] = "tied"
        param_fixed["speedProtocol"] = (
            speed_protocol
            or param_fixed.get("speedProtocol")
            or "four speed changes"
        )
        param_fixed["transitionTime"] = (
            transition_time
            if transition_time is not None
            else param_fixed.get("transitionTime", 3)
        )
        param_fixed["imposedFootSpeeds"] = make_treadmill_speed_tied(param_fixed)
    else:
        raise ValueError(f"Unknown treadmill configuration: {split_or_tied}")

    param_fixed["angleSlope"] = 0

    nominal_step_time = 1.7
    t_list = param_fixed["imposedFootSpeeds"]["tList"]
    param_fixed["numStepsToLearn"] = t_list[-1] / nominal_step_time
    param_fixed["numStepsToLearn"] = round(param_fixed["numStepsToLearn"] / 100) * 100

    learner = param_fixed.get("Learner", {})
    num_iterations = int(
        np.floor(param_fixed["numStepsToLearn"] / learner.get("numStepsPerIteration", 1))
    )
    param_fixed["num_iterations"] = num_iterations
    param_fixed["numIterations"] = num_iterations

    if learner.get("numStepsPerIteration", 1) % 2 != 0:
        learner["numStepsPerIteration"] = learner.get("numStepsPerIteration", 1) + 1
        param_fixed["Learner"] = learner

    return param_fixed
