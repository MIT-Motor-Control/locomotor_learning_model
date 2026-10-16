function paramFixed = loadLearnerParameters(paramFixed)

%% Parameters of the learner
paramFixed.Learner.LearningRate = 0.00012; % baseline learning rate

paramFixed.Learner.noiseSTDExploratory = 0.002;

paramFixed.Learner.includeInternalModel = 1; % takes values zero or one

paramFixed.Learner.numStepsToUseForEstimator = 30;

%% moving toward memory
paramFixed.Learner.LearningRateTowardMemory = 0.01;

%% truncated cosine tuning
paramFixed.Learner.powerToTheMoveToMemory = 10;
paramFixed.Learner.powerToTheMemoryFormation = 10;

%% making new memories
paramFixed.Learner.LearningRateForMemoryFormationUpdates = 0.0;
paramFixed.Learner.LearningRateForMemoryFormationUpdates = 0.03;

%% how frequently do we update the controller parameter
paramFixed.Learner.numStepsPerIteration = 2;

%% trust region
paramFixed.Learner.trustRegionSize = 0.25*paramFixed.Learner.noiseSTDExploratory;
paramFixed.Learner.shouldWeUseTrustRegion = 1;

%% prediction error threshold
paramFixed.Learner.predictionErrorThreshold = 2*paramFixed.noiseEnergySensory; 
paramFixed.Learner.shouldWeThresholdPredictionError = 0;
% Enable thresholding to suppress gradient steps when model errors are large.

%% compatibility setting; the estimator uses a finite window without forgetting
paramFixed.Learner.alphaForgettingForEstimator = 0;

end
