function paramFixed = loadLearnerParameters(paramFixed)
% Set learning rates, exploration noise and estimator settings.

paramFixed.Learner.LearningRate = 0.00012;

paramFixed.Learner.noiseSTDExploratory = 0.002;

paramFixed.Learner.includeInternalModel = 1;

paramFixed.Learner.numStepsToUseForEstimator = 30;

paramFixed.Learner.LearningRateTowardMemory = 0.01;

paramFixed.Learner.powerToTheMoveToMemory = 10;
paramFixed.Learner.powerToTheMemoryFormation = 10;

paramFixed.Learner.LearningRateForMemoryFormationUpdates = 0.03;

paramFixed.Learner.numStepsPerIteration = 2;

paramFixed.Learner.trustRegionSize = 0.25*paramFixed.Learner.noiseSTDExploratory;
paramFixed.Learner.shouldWeUseTrustRegion = 1;

paramFixed.Learner.predictionErrorThreshold = 2*paramFixed.noiseEnergySensory;
paramFixed.Learner.shouldWeThresholdPredictionError = 0;

paramFixed.Learner.alphaForgettingForEstimator = 0;

end
