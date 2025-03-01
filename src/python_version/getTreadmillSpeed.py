import numpy as np
from scipy.interpolate import interp1d

def getTreadmillSpeed(t, beltSpeedsImposed):
    """
    Get treadmill speed at a specific time t
    """
    
    # Create interpolation functions
    interp_func1 = interp1d(beltSpeedsImposed['tList'], 
                          beltSpeedsImposed['footSpeed1List'],
                          kind='linear', 
                          fill_value='extrapolate')
    
    interp_func2 = interp1d(beltSpeedsImposed['tList'], 
                          beltSpeedsImposed['footSpeed2List'],
                          kind='linear', 
                          fill_value='extrapolate')
    
    # Handle both scalar and array inputs
    if np.isscalar(t):
        footSpeed1 = float(interp_func1(t))
        footSpeed2 = float(interp_func2(t))
    else:
        footSpeed1 = interp_func1(t)
        footSpeed2 = interp_func2(t)
    
    return footSpeed1, footSpeed2 
    
    # checked and essential. Used in the main program and in a 
    # post-processing plotting program could be wrapped into some other
    # function.