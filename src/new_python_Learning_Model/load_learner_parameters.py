def load_learner_parameters(param_fixed):
    """
    Load parameters of the learner into the param_fixed dictionary.

    Args:
        param_fixed (dict): Dictionary containing fixed parameters for the simulation.

    Returns:
        dict: Updated param_fixed dictionary with learner parameters.
    """

    # Initialize 'Learner' key if it does not exist
    if 'Learner' not in param_fixed:
        param_fixed['Learner'] = {}
        
    # Parameters of the learner
    # Learning rate for split belt walking
    param_fixed['Learner']['LearningRate'] = 0.00012

    # Noise standard deviation for exploratory action
    param_fixed['Learner']['noiseSTDExploratory'] = 0.002  # Higher for tied walking

    # Include internal model (1: yes, 0: no)
    param_fixed['Learner']['includeInternalModel'] = 1

    # Number of steps to use for estimator
    param_fixed['Learner']['numStepsToUseForEstimator'] = 30

    # Moving toward memory
    param_fixed['Learner']['LearningRateTowardMemory'] = 0.01  # Some progress toward stored memory

    # Truncated cosine tuning
    param_fixed['Learner']['powerToTheMoveToMemory'] = 10
    param_fixed['Learner']['powerToTheMemoryFormation'] = 10

    # Making new memories
    param_fixed['Learner']['LearningRateForMemoryFormationUpdates'] = 0.03

    # Frequency of controller parameter updates
    param_fixed['Learner']['numStepsPerIteration'] = 2

    # Trust region settings
    param_fixed['Learner']['trustRegionSize'] = 0.25 * param_fixed['Learner']['noiseSTDExploratory']
    param_fixed['Learner']['shouldWeUseTrustRegion'] = 1

    # Prediction error threshold
    param_fixed['Learner']['predictionErrorThreshold'] = 2 * param_fixed['noiseEnergySensory']
    param_fixed['Learner']['shouldWeThresholdPredictionError'] = 0  # Set 0 if not interested in thresholding

    # Exponential forgetting for gradient estimator (not used)
    param_fixed['Learner']['alphaForgettingForEstimator'] = 0

    return param_fixed

