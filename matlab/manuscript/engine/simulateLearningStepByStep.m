function [pInputControllerStore_OnesTried, fall_flag_vec,min_vert_pos] = simulateLearningStepByStep(paramFixed, ...
    pInputControllerAsymmetricNominal,stateVar0_Model,contextNow, ...
    paramControllerGains,SwingCostCoeff,speedChangeIdx)
% Update the controller from noisy energy estimates and learned dynamics.

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

fall_flag_vec = [0 0];
min_vert_pos = [];
numStridesToUse = paramFixed.Learner.numStepsToUseForEstimator;
stateVar0_ModelNow = stateVar0_Model;
pInputNow_ConsideredGood = pInputNow;

for iStride = 1:numIterations
    if mod(iStride, 200) == 0
        disp(['iStride = ' num2str(iStride)]);
    end

    delta_pInputNow1_Noise = noiseSTD*randn(numLearningDimensions,1);

    if length(alphaEnergyLearningRate) == 1
        delta_pInputNow2_Gradient = alphaEnergyLearningRate*(-gEnergyGradientNow);

    elseif length(alphaEnergyLearningRate) == 2
        alphaStart = alphaEnergyLearningRate(1); learningDecayRate = alphaEnergyLearningRate(2);
        if iStride > speedChangeIdx
            currentStride = iStride - speedChangeIdx;
            updatedLearningRate = alphaStart - learningDecayRate * (currentStride / (numIterations - speedChangeIdx)) * alphaStart;
            if updatedLearningRate < 0
                updatedLearningRate = 0;
            end
           delta_pInputNow2_Gradient = updatedLearningRate*(-gEnergyGradientNow);

        else
           delta_pInputNow2_Gradient = alphaStart*(-gEnergyGradientNow);
        end
    end

    if paramFixed.Learner.shouldWeUseTrustRegion
        if norm(delta_pInputNow2_Gradient)>paramFixed.Learner.trustRegionSize*sqrt(numLearningDimensions)
            delta_pInputNow2_Gradient = delta_pInputNow2_Gradient/norm(delta_pInputNow2_Gradient);
            delta_pInputNow2_Gradient = delta_pInputNow2_Gradient*paramFixed.Learner.trustRegionSize*sqrt(numLearningDimensions);
        end
    end

    pInputNow_ConsideredGood = pInputNow_ConsideredGood + delta_pInputNow2_Gradient;
    pInputNow_ToTry = pInputNow_ConsideredGood + delta_pInputNow1_Noise;

    [EdotNow,stateVar0_ModelNext,tEnd,f_energy,f_constraint,fall_info,y_pos_single] = ...
        fObjective_AsymmetricNominal_8DTo10D(pInputNow_ToTry,stateVar0_ModelNow, ...
        paramControllerGains,paramFixed,tStart,SwingCostCoeff,iStride);

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

    if fall_info(1) == 1
        fall_flag_vec = fall_info;
        break;
    end

    min_vert_pos = [min_vert_pos; y_pos_single'];
end

end
