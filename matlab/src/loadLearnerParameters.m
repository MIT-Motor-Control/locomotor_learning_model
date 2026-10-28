function paramFixed = loadLearnerParameters(paramFixed)
% Add learning rates, exploration noise and estimator settings to paramFixed.
% Requires paramFixed.noiseEnergySensory (loadSensoryNoiseParameters).

paramFixed.Learner.LearningRate = 0.00012;

paramFixed.Learner.noiseSTDExploratory = 0.002;

paramFixed.Learner.includeInternalModel = 1;

% Estimator window, in learning iterations.
paramFixed.Learner.numStepsToUseForEstimator = 30;

paramFixed.Learner.LearningRateTowardMemory = 0.01;

% Exponents of the direction-cosine weights in simulateLearningStepByStep.
paramFixed.Learner.powerToTheMoveToMemory = 10;
paramFixed.Learner.powerToTheMemoryFormation = 10;

paramFixed.Learner.LearningRateForMemoryFormationUpdates = 0.03;

% Physical steps per learning iteration (one stride).
paramFixed.Learner.numStepsPerIteration = 2;

% Bound on the root-mean-square gradient step per controller parameter.
paramFixed.Learner.trustRegionSize = 0.25*paramFixed.Learner.noiseSTDExploratory;
paramFixed.Learner.shouldWeUseTrustRegion = 1;

% Optional suppression of the gradient step after large prediction errors (off).
paramFixed.Learner.predictionErrorThreshold = 2*paramFixed.noiseEnergySensory;
paramFixed.Learner.shouldWeThresholdPredictionError = 0;

% Not read by the simulation; RLSupdate applies no forgetting.
paramFixed.Learner.alphaForgettingForEstimator = 0;

end
