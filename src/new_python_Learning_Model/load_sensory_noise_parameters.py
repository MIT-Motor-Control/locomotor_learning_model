def load_sensory_noise_parameters(param_fixed):
    """
    Load parameters related to sensory noise, which affects learning and stability.
    
    Args:
        param_fixed (dict): Dictionary containing existing fixed parameters.
        
    Returns:
        dict: Updated dictionary with sensory noise parameters.
    """
    # Multiplicative noise for energy measurements
    param_fixed['noiseEnergySensory'] = 0.0001  # This is multiplicative noise
    
    # Noise terms
    param_fixed['velocitySensoryNoise'] = 0  # This is additive noise
    
    return param_fixed