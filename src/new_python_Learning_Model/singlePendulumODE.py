import numpy as np
from scipy.interpolate import interp1d

def singlePendulumODE(t, stateVar, paramFixed, paramController):
    """
    Compute the derivatives for the single pendulum ODE.
    """
    

    # Unwrap state variables
    angleTheta = stateVar[0]
    dangleTheta = stateVar[1]
    yFoot = stateVar[2]

    # Interpolate to get the right acceleration
    tList_BeltSpeed = paramController['tList_BeltSpeed']
    PushoffAccelerationNowList = paramController['PushoffAccelerationNowList']
    # Create interpolation function with kind='previous'
    f_accel = interp1d(tList_BeltSpeed, PushoffAccelerationNowList, kind='previous', fill_value='extrapolate', bounds_error=False)
    PushoffAccelerationNow = f_accel(t)
    paramController['PushoffAccelerationNow'] = PushoffAccelerationNow

    # Forward 'inertial force' due to backward belt acceleration
    mbody = paramFixed['mbody']
    leglength = paramFixed['leglength']
    gravg = paramFixed['gravg']
    includeAccelerationTorque = paramFixed['includeAccelerationTorque']

    forceDueToAcceleration = -PushoffAccelerationNow * mbody
    torqueDueToAcceleration = (forceDueToAcceleration * np.cos(angleTheta)) / (mbody * leglength)

    if not includeAccelerationTorque:
        torqueDueToAcceleration = 0

    # Angular acceleration of the stance leg
    ddAngleTheta = (gravg / leglength) * np.sin(angleTheta) + torqueDueToAcceleration

    # Interpolate to get the right speed
    PushoffFootSpeedNowList = paramController['PushoffFootSpeedNowList']
    f_speed = interp1d(tList_BeltSpeed, PushoffFootSpeedNowList, kind='linear', fill_value='extrapolate', bounds_error=False)
    PushoffFootSpeedNow = f_speed(t)
    paramController['PushoffFootSpeedNow'] = PushoffFootSpeedNow

    # Integrating yFoot position so that it is easily accessible
    dyFoot = PushoffFootSpeedNow  # Treadmill speed

    # Prepare derivatives
    dStateVar = [dangleTheta, ddAngleTheta, dyFoot]

    return dStateVar
