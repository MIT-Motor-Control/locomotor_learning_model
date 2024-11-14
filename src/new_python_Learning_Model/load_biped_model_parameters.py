def load_biped_model_parameters(param_fixed):
    """
    Load parameters related to the biped dynamics, such as biped segment masses,
    lengths, gravity, and performance cost parameters.
    
    Args:
        param_fixed (dict): Dictionary containing existing fixed parameters.
        
    Returns:
        dict: Updated dictionary with biped model parameters.
    """
    # Treadmill acceleration include
    param_fixed['includeAccelerationTorque'] = 1
    
    # Body parameters, FIXED
    param_fixed['mbody'] = 1
    param_fixed['leglength'] = 1
    param_fixed['gravg'] = 1
    
    # Efficiency of positive and negative work
    param_fixed['efficiency_neg'] = 1.2
    param_fixed['efficiency_pos'] = 0.25
    param_fixed['bPos'] = 1 / param_fixed['efficiency_pos']
    param_fixed['bNeg'] = 1 / param_fixed['efficiency_neg']
    
    # Swing leg energy cost parameters
    param_fixed['mFoot'] = 0.05
    param_fixed['swingCost'] = {
        'Coeff': 0.9,
        'alpha': 1.0
    }
    
    # Energy vs periodicity variance reduction
    param_fixed['lambdaEnergyVsPeriodicity'] = 1  # 1 = energy, 0 = periodicity
    
    # Energy vs symmetry (commented out in MATLAB)
    # param_fixed['lambdaEnergyVsSymmetry'] = 1  # 1 = energy, 0 = symmetry
    
    return param_fixed