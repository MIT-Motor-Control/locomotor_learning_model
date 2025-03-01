def loadSensoryNoiseParameters(paramFixed):
    """
    Sensory noise just makes learning challenging. Setting these to zero
    just lowers the noise in the behavior without affecting the qualitative
    trends, but increasing these eventually kills learning or makes the biped
    unstable and fall
    """
    
    ## multiplicative noise for energy measurements
    paramFixed['noiseEnergySensory'] = 0.0001  # this is multiplicative noise
    # paramFixed['noiseEnergySensory'] = 0.01  # this is multiplicative noise
    # paramFixed['noiseEnergySensory'] = 0.00  # this is multiplicative noise
    
    ## noise terms
    paramFixed['velocitySensoryNoise'] = 0
    # this is additive noise
    
    return paramFixed  # code checked