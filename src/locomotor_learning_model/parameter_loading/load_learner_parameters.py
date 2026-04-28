"""Parameters governing the learner."""
from __future__ import annotations


def load_learner_parameters(param_fixed: dict | None = None) -> dict:
    """Return learner settings for the adaptation simulation."""
    if param_fixed is None:
        param_fixed = {}

    learner = param_fixed.get('Learner', {})

    learner['LearningRate'] = 0.00012
    learner['noiseSTDExploratory'] = 0.002
    learner['includeInternalModel'] = 1
    learner['numStepsToUseForEstimator'] = 30

    learner['LearningRateTowardMemory'] = 0.01

    learner['powerToTheMoveToMemory'] = 10
    learner['powerToTheMemoryFormation'] = 10

    learner['LearningRateForMemoryFormationUpdates'] = 0.03

    learner['numStepsPerIteration'] = 2

    learner['trustRegionSize'] = 0.25 * learner['noiseSTDExploratory']
    learner['shouldWeUseTrustRegion'] = 1

    learner['predictionErrorThreshold'] = 2 * param_fixed.get('noiseEnergySensory', 0)
    learner['shouldWeThresholdPredictionError'] = 0

    learner['alphaForgettingForEstimator'] = 0

    param_fixed['Learner'] = learner
    return param_fixed
