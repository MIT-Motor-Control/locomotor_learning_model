function pInputControllerStore_OnesTried = simulateLearningStepByStep(paramFixed, ...
    pInputControllerAsymmetricNominal,stateVar0_Model,contextNow, ...
    paramControllerGains)
% Update the controller from noisy energy estimates and learned dynamics.

disp(['Simulating Learning While Walking ... (' num2str(paramFixed.numIterations) ' strides)']);

noiseSTD = paramFixed.Learner.noiseSTDExploratory;
numIterations = paramFixed.numIterations;
includeInternalModelOrNot = paramFixed.Learner.includeInternalModel;

% Fix the two odd-leg position coordinates to define the spatial origin.
pInputControllerAsymmetricNominal = pInputControllerAsymmetricNominal([1:3 6:10]);
numLearningDimensions = length(pInputControllerAsymmetricNominal);
numStateDimensions = length(stateVar0_Model);

alphaEnergyLearningRate = paramFixed.Learner.LearningRate;

pInputNow = pInputControllerAsymmetricNominal;

tStart = 0;

AdynamicsModelNow = ...
    zeros(numStateDimensions-1,numLearningDimensions+numStateDimensions);
AdynamicsEnergyNow = ...
    zeros(1,numLearningDimensions+numStateDimensions);

gEnergyGradientNow = zeros(size(pInputControllerAsymmetricNominal));

pInputControllerStore_OnesTried = [];
stateVarNowStore = [];
stateVarNextStore = [];
EdotStore = [];
pInputControllerStore_OnesConsideredGood = [];
gradientEnergyEstimateStore = cell(1,1);
AdynamicsModelStore = cell(1,1);
AdynamicsEnergyStore = cell(1,1);

errorOldDynamicsModelStore = [];
errorNewDynamicsModelStore = [];
errorOldEnergyModelStore = [];
errorNewEnergyModelStore = [];
learningOnOrNotStore = [];
memoryToGradientDirectionCosineStore = [];

gGradientForMemoryUpdateStore = cell(1,1);
errorFromMemoryStore = [];
pInputMemoryNowStore = [];

numStridesToUse = paramFixed.Learner.numStepsToUseForEstimator;
stateVar0_ModelNow = stateVar0_Model;
pInputNow_ConsideredGood = pInputNow;

for iStride = 1:numIterations

    if mod(iStride,50)==0
        disp(['iStride = ' num2str(iStride)]);
    end

    delta_pInputNow1_Noise = noiseSTD*randn(numLearningDimensions,1);

    delta_pInputNow2_Gradient = alphaEnergyLearningRate*(-gEnergyGradientNow);

    if paramFixed.Learner.shouldWeUseTrustRegion
        if norm(delta_pInputNow2_Gradient)>paramFixed.Learner.trustRegionSize*sqrt(numLearningDimensions)
            delta_pInputNow2_Gradient = delta_pInputNow2_Gradient/norm(delta_pInputNow2_Gradient);
            delta_pInputNow2_Gradient = delta_pInputNow2_Gradient*paramFixed.Learner.trustRegionSize*sqrt(numLearningDimensions);
        end
    end

    pInputMemoryNow = paramFixed.storedmemory.nominalControl + ...
        paramFixed.storedmemory.controlSlopeVsContext*(contextNow-paramFixed.storedmemory.nominalContext);
    pInputMemoryNowStore = [pInputMemoryNowStore pInputMemoryNow];

    dirTowardMemory = (pInputMemoryNow-pInputNow_ConsideredGood);

    % Memory attraction depends on alignment with the energy gradient.
    temp = ...
        dot(dirTowardMemory,-gEnergyGradientNow)/norm(dirTowardMemory,2)/norm(gEnergyGradientNow,2);
    memoryToGradientDirectionCosineStore(iStride) = temp;

    powerMovetoMemory = paramFixed.Learner.powerToTheMoveToMemory;

    % A zero-length direction gives an undefined direction cosine.
    if isnan(temp)
        temp = 1*paramFixed.Learner.LearningRateTowardMemory;
    else
        temp = (1+temp)/2;
        temp = temp^powerMovetoMemory;
        temp = temp*paramFixed.Learner.LearningRateTowardMemory;
    end

    delta_pInputNow3_TowardMemory = temp*dirTowardMemory;

    pInputNow_ConsideredGood = pInputNow_ConsideredGood + delta_pInputNow3_TowardMemory;

    pInputNow_ConsideredGood = pInputNow_ConsideredGood + delta_pInputNow2_Gradient;
    pInputNow_ToTry = pInputNow_ConsideredGood + delta_pInputNow1_Noise;

    [EdotNow,stateVar0_ModelNext,tEnd] = ...
        fObjective_AsymmetricNominal_8DTo10D(pInputNow_ToTry,stateVar0_ModelNow, ...
        paramControllerGains,paramFixed,tStart);

    EdotNow = EdotNow*(1+randn*paramFixed.noiseEnergySensory);

    stateVarNowStore =  [stateVarNowStore  stateVar0_ModelNow];
    stateVarNextStore = [stateVarNextStore stateVar0_ModelNext];
    pInputControllerStore_OnesTried = [pInputControllerStore_OnesTried pInputNow_ToTry];
    EdotStore = [EdotStore EdotNow];
    pInputControllerStore_OnesConsideredGood = ...
        [pInputControllerStore_OnesConsideredGood pInputNow_ConsideredGood];

    if iStride>numStridesToUse
        [AdynamicsModelNow,errorOldDynamicsModel,errorNewDynamicsModel] = ...
            computeModelDynamicsModelRLS_NoSingular(AdynamicsModelNow,stateVarNowStore,...
            stateVarNextStore,pInputControllerStore_OnesTried,numStridesToUse);
    end

    if iStride>numStridesToUse
        [AdynamicsEnergyNow,errorOldEnergyModel,errorNewEnergyModel] = computeEnergyDynamicsModelRLS_NoSingular(AdynamicsEnergyNow, ...
            stateVarNowStore,EdotStore,pInputControllerStore_OnesTried,numStridesToUse);
    end

    if iStride>numStridesToUse
        errorOldDynamicsModelStore = [errorOldDynamicsModelStore errorOldDynamicsModel];
        errorNewDynamicsModelStore = [errorNewDynamicsModelStore errorNewDynamicsModel];

        errorOldEnergyModelStore = [errorOldEnergyModelStore errorOldEnergyModel];
        errorNewEnergyModelStore = [errorNewEnergyModelStore errorNewEnergyModel];
    end

    gEnergyGradientNow = computeSteadyEnergyGradientV2_NoSingular(AdynamicsModelNow, ...
        AdynamicsEnergyNow,pInputNow_ToTry, ...
        numLearningDimensions,numStateDimensions,includeInternalModelOrNot);

    gradientEnergyEstimateStore{iStride} = gEnergyGradientNow;
    AdynamicsModelStore{iStride} = AdynamicsModelNow;
    AdynamicsEnergyStore{iStride} = AdynamicsEnergyNow;

    if paramFixed.Learner.shouldWeThresholdPredictionError
        if iStride>numStridesToUse
            predictionErrorListNow = ...
                [errorOldEnergyModel; errorNewEnergyModel];
            if any(abs(predictionErrorListNow)>paramFixed.Learner.predictionErrorThreshold)
                gEnergyGradientNow = 0*gEnergyGradientNow;
                learningOnOrNotStore = [learningOnOrNotStore; 0];
            else
                learningOnOrNotStore = [learningOnOrNotStore; 1];
            end
        end
    end

    tStart = tEnd;
    stateVar0_ModelNow = stateVar0_ModelNext;

    drawnow;

    gGradientForMemoryUpdate = ...
        gradientOfErrorFromMemoryCompute(paramFixed.storedmemory.controlSlopeVsContext, ...
        pInputNow_ConsideredGood,paramFixed,contextNow);

    n_memorytocurrentcontroller = -dirTowardMemory;
    n_memorytocurrentcontroller = n_memorytocurrentcontroller/norm(n_memorytocurrentcontroller,2);

    n_vController = delta_pInputNow2_Gradient + delta_pInputNow3_TowardMemory;
    n_vController = n_vController/norm(n_vController);

    dot_MemoryToCurrent_vCurrent = dot(n_memorytocurrentcontroller,n_vController);

    temp = dot_MemoryToCurrent_vCurrent;
    powerForMemoryFormation = paramFixed.Learner.powerToTheMemoryFormation;
    % A zero-length direction gives an undefined direction cosine.
    if isnan(temp)
        temp = 0;
    else
        temp = (1+temp)/2;
        temp = temp^powerForMemoryFormation;
        temp = temp*paramFixed.Learner.LearningRateForMemoryFormationUpdates;
    end

    paramFixed.storedmemory.controlSlopeVsContext = ...
        paramFixed.storedmemory.controlSlopeVsContext + ...
        temp*(-gGradientForMemoryUpdate);

    gGradientForMemoryUpdateStore{iStride} = gGradientForMemoryUpdate;
    errorFromMemoryStore = [errorFromMemoryStore errorFromMemoryCompute(paramFixed.storedmemory.controlSlopeVsContext,pInputNow_ConsideredGood,paramFixed,contextNow)];

    [vA,vB] = getTreadmillSpeed(tStart,paramFixed.imposedFootSpeeds);
    contextNow = [vA; vB];

    contextNow = contextNow + paramFixed.noiseEnergySensory*randn;

end

end
