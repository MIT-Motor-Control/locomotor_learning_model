import numpy as np

def DetectEndstance(t, stateVar, vBelt, paramController, paramFixed):
    """
    Detect end of stance phase when leg angle reaches the target angle
    
    This function is used as an event function in the ODE solver.
    The event occurs when the value crosses zero.
    """
    # Unpack state variables and parameters
    theta = stateVar[0]
    theta_end_nominal = paramController['theta_end_nominal']
    
    # Event occurs when leg angle reaches target angle
    # Returns negative before the event and positive after
    eventValue = theta - theta_end_nominal
    
    return eventValue