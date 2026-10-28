function pInputControllerStore_OnesTried = simulateLearningStepByStep(paramFixed, ...
    pInputControllerAsymmetricNominal,stateVar0_Model,contextNow, ...
    paramControllerGains)
% Simulate learning of the 8-D nominal controller over paramFixed.numIterations iterations.
% Each iteration moves the controller along a model-based estimate of the
% objective gradient and toward the context-dependent memory, adds exploration
% noise, walks one iteration with it, refits the linear state and objective
% models, and updates the memory.
% Inputs:  paramFixed                        - model, learner and protocol parameters
%          pInputControllerAsymmetricNominal - initial 10-D nominal controller
%          stateVar0_Model                   - initial midstance state
%          contextNow                        - initial belt speeds [belt 1; belt 2]
%          paramControllerGains              - feedback gains
% Outputs: pInputControllerStore_OnesTried   - 8-D controllers walked, one column per iteration

disp(['Simulating learning while walking (' num2str(paramFixed.numIterations) ' learning iterations)']);

noiseSTD = paramFixed.Learner.noiseSTDExploratory;
numIterations = paramFixed.numIterations;
includeInternalModelOrNot = paramFixed.Learner.includeInternalModel;

% Learn 8 of the 10 parameters; the odd-step position targets stay at zero
% and define the spatial origin.
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

    % Step toward the memory, weighted by its alignment with the descent direction.
    temp = ...
        dot(dirTowardMemory,-gEnergyGradientNow)/norm(dirTowardMemory,2)/norm(gEnergyGradientNow,2);
    memoryToGradientDirectionCosineStore(iStride) = temp;

    powerMovetoMemory = paramFixed.Learner.powerToTheMoveToMemory;

    % The cosine is NaN when either vector is zero, e.g. before the first model fit.
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

    % EdotNow is the weighted objective of fObjective_AsymmetricNominal; the
    % learner's energy variables (Edot*, AdynamicsEnergy*, gEnergy*) refer to it.
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

    % The models are first fit at iteration numStridesToUse+1; until then they
    % and the gradient estimate stay zero.
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

    % Memory formation: update the slopes more when the controller moves away
    % from the memory, and not at all when the cosine is undefined.
    n_memorytocurrentcontroller = -dirTowardMemory;
    n_memorytocurrentcontroller = n_memorytocurrentcontroller/norm(n_memorytocurrentcontroller,2);

    n_vController = delta_pInputNow2_Gradient + delta_pInputNow3_TowardMemory;
    n_vController = n_vController/norm(n_vController);

    dot_MemoryToCurrent_vCurrent = dot(n_memorytocurrentcontroller,n_vController);

    temp = dot_MemoryToCurrent_vCurrent;
    powerForMemoryFormation = paramFixed.Learner.powerToTheMemoryFormation;
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

    % One context-noise draw is shared by both belts.
    contextNow = contextNow + paramFixed.noiseEnergySensory*randn;

end

end
