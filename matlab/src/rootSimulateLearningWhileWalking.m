% Run the baseline learning simulation and plot its outputs.
clear; close all; clc;

paramFixed = [];

rng('shuffle','twister');

paramFixed = loadBipedModelParameters(paramFixed);

paramFixed = loadSensoryNoiseParameters(paramFixed);

paramControllerGains = loadControllerGainParameters(paramFixed);

paramFixed = loadLearnerParameters(paramFixed);

paramFixed = loadProtocolParameters(paramFixed);

paramFixed = loadStoredMemoryParameters_ControlVsSpeed(paramFixed);

pInputControllerAsymmetricNominal = loadLearnableParametersInitial(paramFixed);

stateVar0_Model = loadInitialBodyState(pInputControllerAsymmetricNominal);
tStart = 0;

stateVar0_Model_BeforeLearning = stateVar0_Model;

[vA,vB] = getTreadmillSpeed(0,paramFixed.imposedFootSpeeds);
contextNow = [vA; vB];
contextLength = length(contextNow);

[pInputControllerStore_OnesTried] = ...
    simulateLearningStepByStep(paramFixed,pInputControllerAsymmetricNominal, ...
    stateVar0_Model,contextNow,paramControllerGains);

pInputControllerStore_8D = pInputControllerStore_OnesTried;
pInputControllerStore_10D = [pInputControllerStore_8D(1:3,:);
    zeros(2,size(pInputControllerStore_8D,2));
    pInputControllerStore_8D(4:8,:)];

doAnimate = 0;
postProcessAfterLearning(pInputControllerStore_10D, ...
    stateVar0_Model_BeforeLearning, ...
    paramControllerGains,paramFixed,doAnimate);
