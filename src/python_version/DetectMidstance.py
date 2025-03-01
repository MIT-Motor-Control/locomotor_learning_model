import numpy as np

def DetectMidstance(t, stateVar, vBelt, paramController, paramFixed):
    """
    Detect midstance (when leg is vertical)
    
    This function is used as an event function in the ODE solver.
    The event occurs when the value crosses zero.
    """
    # Unpack state variables
    theta = stateVar[0]
    
    # Midstance is when leg angle is zero (vertical position)
    # Returns negative before vertical, positive after
    eventValue = theta
    
    return eventValue