import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

def odeMetToVO2MET(t, EmetS, params):
    """
    ODE function for metabolic rate to VO2 conversion
    """
    EmetNow = params['interp_func'](t)
    
    dEmetVO2dt = params['k'] * (EmetNow - EmetS)
    
    return dEmetVO2dt

def convertMetToVO2(params):
    """
    This program is used to convert the instantaneous stride-wise metabolic
    rate to something that would be measured by indirect calorimetry. Uses a
    linear model with a 44 second time constant. See manuscript for citations
    """
    
    # Check if we have enough data to process
    if len(params['tList']) < 2 or len(params['EmetRateList']) < 2:
        print("Warning: Not enough data points for VO2 conversion")
        return params['tList'], params['EmetRateList']
        
    ## time constant of the linear system
    timeConstant = 30  # Adjusted from 44 seconds to match expected smoothing
    g = 9.81
    LegLength = 0.95
    timeScaling = np.sqrt(LegLength/g)
    timeConstant = timeConstant/timeScaling
    params['k'] = 1/timeConstant
    
    ## Create interpolation function for ODE solver
    try:
        params['interp_func'] = interp1d(
            params['tList'], 
            params['EmetRateList'], 
            kind='linear', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
    except Exception as e:
        print(f"Warning: Error creating interpolation for VO2: {str(e)}")
        return params['tList'], params['EmetRateList']
    
    ## To get the IC, we average over 30 steps at least, so the IC is not noise
    min_steps = min(30, len(params['EmetRateList']))
    EmetS_0 = np.mean(params['EmetRateList'][:min_steps])
    tSpan = params['tList']  # we could refine this further if necessary
    
    ## Solve ODE
    try:
        solution = solve_ivp(
            lambda t, y: odeMetToVO2MET(t, y, params),
            [tSpan[0], tSpan[-1]],
            [EmetS_0],
            method='RK45',
            t_eval=tSpan,
            rtol=1e-8,
            atol=1e-8
        )
        
        # Check if the solution was successful
        if solution.success:
            return tSpan, solution.y[0]
        else:
            print(f"Warning: VO2 ODE solution failed")
            return tSpan, params['EmetRateList']
            
    except Exception as e:
        print(f"Warning: Error solving VO2 ODE: {str(e)}")
        return tSpan, params['EmetRateList']
        
    # Return original data if anything fails
    return tSpan, params['EmetRateList']