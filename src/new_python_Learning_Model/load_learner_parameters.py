def load_learner_parameters(param_fixed):
    """
    Load parameters related to the learner, including learning rate and noise.
    
    Args:
        param_fixed (dict): Dictionary containing existing fixed parameters.
        
    Returns:
        dict: Updated dictionary with learner parameters.
    """
    # Parameters of the learner
    param_fixed['Learner'] = {
        'LearningRate': 0.00012,  # Default learning rate for split belt walking
        'noiseSTDExploratory': 0.002,  # Higher for tied walking
        'includeInternalModel': 1,  # Takes values 0 or 1
        'numStepsToUseForEstimator': 30,
        'LearningRateTowardMemory': 0.01  # Default progress toward stored memory
    }
    
    return param_fixed
