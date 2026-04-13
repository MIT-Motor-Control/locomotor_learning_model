function export_reference_run(outputPath, numIterations)
%EXPORT_REFERENCE_RUN Save a deterministic MATLAB reference run for parity checks.

if nargin < 1 || isempty(outputPath)
    outputPath = fullfile(tempdir, 'locomotor_learning_model_reference.mat');
end

if nargin < 2 || isempty(numIterations)
    numIterations = 10;
end

thisDir = fileparts(mfilename('fullpath'));
matlabDir = fileparts(thisDir);
addpath(fullfile(matlabDir, 'src'));
set(0, 'DefaultFigureVisible', 'off');

rng(42, 'twister');
paramFixed = [];
paramFixed = loadBipedModelParameters(paramFixed);
paramFixed = loadSensoryNoiseParameters(paramFixed);
paramControllerGains = loadControllerGainParameters(paramFixed);
paramFixed = loadLearnerParameters(paramFixed);
paramFixed = loadProtocolParameters(paramFixed);
paramFixed = loadStoredMemoryParameters_ControlVsSpeed(paramFixed);

paramFixed.Learner.noiseSTDExploratory = 0;
paramFixed.noiseEnergySensory = 0;
paramFixed.numIterations = numIterations;

pInput = loadLearnableParametersInitial(paramFixed);
stateVar0 = loadInitialBodyState(pInput);
[vA, vB] = getTreadmillSpeed(0, paramFixed.imposedFootSpeeds);
contextNow = [vA; vB];
objectiveValue = fObjective_AsymmetricNominal(pInput, stateVar0, paramControllerGains, paramFixed, 0);
controllerStore8D = simulateLearningStepByStep(paramFixed, pInput, stateVar0, contextNow, paramControllerGains);

save(outputPath, 'objectiveValue', 'controllerStore8D', 'pInput', 'stateVar0', 'contextNow');
disp(['Saved MATLAB reference to ' outputPath]);
end
