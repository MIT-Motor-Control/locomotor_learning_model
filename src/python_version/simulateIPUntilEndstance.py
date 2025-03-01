import numpy as np
from scipy.integrate import solve_ivp
from singlePendulumODE import singlePendulumODE
from DetectEndstance import DetectEndstance

def simulateIPUntilEndstance(stateVar0, tStart, vBelt, paramController, paramFixed):
    """
    Simulate inverted pendulum from midstance until endstance
    
    Integrates the inverted pendulum dynamics from midstance until the 
    leg angle reaches the target angle for pushoff.
    """
    # Set up event detection for end stance
    def event_fn(t, y):
        return DetectEndstance(t, y, vBelt, paramController, paramFixed)
    
    event_fn.terminal = True    # Stop integration when event occurs
    event_fn.direction = 1      # Only detect when crossing from negative to positive
    
    # Maximum integration time (in case event isn't triggered)
    t_max = tStart + 5.0
    
    # Define the ODE function for scipy's solver
    def ode_fn(t, y):
        return singlePendulumODE(t, y, vBelt, paramController, paramFixed)
    
    # Solve the ODE system
    solution = solve_ivp(
        ode_fn,
        [tStart, t_max],
        stateVar0,
        method='RK45',
        events=event_fn,
        rtol=1e-6,
        atol=1e-6,
        max_step=0.05
    )
    
    # Extract solution
    tList = solution.t
    stateList = solution.y.T
    
    return tList, stateList