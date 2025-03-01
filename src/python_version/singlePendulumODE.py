import numpy as np

def singlePendulumODE(t, stateVar, vBelt, paramController, paramFixed):
    """
    ODE function for the single inverted pendulum model
    
    State variables:
    stateVar[0] = theta (leg angle)
    stateVar[1] = theta_dot (leg angular velocity)
    stateVar[2] = y_foot (foot position in lab frame)
    stateVar[3] = sum_y_foot (sum of foot positions for integral control)
    stateVar[4] = v_swing (swing leg velocity)
    """
    # Unpack state variables
    theta = stateVar[0]
    theta_dot = stateVar[1]
    y_foot = stateVar[2]
    sum_y_foot = stateVar[3]
    v_swing = stateVar[4]
    
    # Physical parameters
    g = paramFixed['gravg']        # Gravitational acceleration
    L = paramFixed['leglength']    # Leg length
    
    # Control parameters 
    # For more complex controlled simulations we would use these
    theta_end_nominal = paramController['theta_end_nominal']
    ydot_nominal = paramController['ydot_atMidstance_nominal_beltframe']
    pushoff_nominal = paramController['PushoffImpulseMagnitude_nominal']
    
    # Calculate gravitational force effect
    theta_ddot = g/L * np.sin(theta)
    
    # Add belt acceleration effect if configured
    if paramFixed['includeAccelerationTorque'] == 1:
        # Get belt acceleration at current time
        tList = paramFixed['imposedFootSpeeds']['tList']
        accList = paramFixed['imposedFootSpeeds']['footAcc1List'] 
        
        # Find belt acceleration through linear interpolation
        if t <= tList[0]:
            acc = accList[0]
        elif t >= tList[-1]:
            acc = accList[-1]
        else:
            # Find index of first time point >= current time
            idx = np.searchsorted(tList, t)
            # Linear interpolation between adjacent points
            t0, t1 = tList[idx-1], tList[idx]
            a0, a1 = accList[idx-1], accList[idx]
            acc = a0 + (a1 - a0) * (t - t0) / (t1 - t0)
        
        # Add acceleration torque effect
        theta_ddot = theta_ddot - acc * np.cos(theta) / g
    
    # State derivatives
    dstatedt = np.zeros(5)
    dstatedt[0] = theta_dot                  # d(theta)/dt = theta_dot
    dstatedt[1] = theta_ddot                 # d(theta_dot)/dt = theta_ddot
    dstatedt[2] = 0                          # y_foot is constant during stance
    dstatedt[3] = 0                          # sum_y_foot is constant during stance
    dstatedt[4] = 0                          # v_swing is changed at discrete events
    
    return dstatedt