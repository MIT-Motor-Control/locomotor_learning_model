import numpy as np
from scipy.integrate import solve_ivp
from singlePendulumODE import singlePendulumODE
from DetectMidstance import DetectMidstance

def simulateIPUntilMidstance(stateVar0, tStart, vBelt, paramController, paramFixed):
    """
    Simulate inverted pendulum from endstance until midstance
    
    Integrates the inverted pendulum dynamics from end stance until the 
    leg angle reaches vertical (midstance), applying pushoff and heelstrike impulses.
    """
    # Apply pushoff impulse at the beginning
    stateVar_after_pushoff = stateVar0.copy()
    
    # Get pushoff impulse magnitude
    pushoff_impulse = paramController['PushoffImpulseMagnitude_nominal']
    
    # Apply pushoff impulse to angular velocity and foot position
    theta = stateVar0[0]
    pushoff_angular_velocity = pushoff_impulse * np.cos(theta)
    
    stateVar_after_pushoff[1] += pushoff_angular_velocity
    
    # Heelstrike: switching stance leg
    stateVar_after_heelstrike = stateVar_after_pushoff.copy()
    
    # Get desired midstance velocity (from controller)
    ydot_nominal = paramController['ydot_atMidstance_nominal_beltframe']
    
    # Calculate new foot position after heelstrike
    step_length = 2 * paramFixed['leglength'] * np.sin(theta)
    new_foot_position = stateVar_after_pushoff[2] + step_length
    
    # Update state after heelstrike
    stateVar_after_heelstrike[0] = -theta  # Flipped leg angle
    stateVar_after_heelstrike[1] *= -0.8   # Reduced and reversed angular velocity
    stateVar_after_heelstrike[2] = new_foot_position
    stateVar_after_heelstrike[3] += new_foot_position  # Update sum of foot positions
    stateVar_after_heelstrike[4] = 0.35  # Reset swing leg velocity
    
    # Set up event detection for midstance
    def event_fn(t, y):
        return DetectMidstance(t, y, vBelt, paramController, paramFixed)
    
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
        stateVar_after_heelstrike,
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