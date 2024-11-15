def DetectEndstance(t, stateVar, paramFixed, paramController):
    """
    Event function to detect endstance.

    Returns:
    - value: float, zero-crossing value for the event.
    """
    angleTheta = stateVar[0]
    theta_end_thisStep = paramController['theta_end_thisStep']

    value = angleTheta - theta_end_thisStep
    isterminal = 1
    direction = +1

    return value, isterminal, direction
