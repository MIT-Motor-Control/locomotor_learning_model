function [pInputControllerStore_OnesTried, fall_flag_vec,min_vert_pos] = simulateLearningStepByStep(paramFixed, ...
    pInputControllerAsymmetricNominal,stateVar0_Model,contextNow, ...
    paramControllerGains,SwingCostCoeff,speedChangeIdx)

%% key paramaters
noiseSTD = paramFixed.Learner.noiseSTDExploratory;
numIterations = paramFixed.numIterations;
includeInternalModelOrNot = paramFixed.Learner.includeInternalModel;

%%
pInputControllerAsymmetricNominal = pInputControllerAsymmetricNominal([1:3 6:10]);
% because the last two variablees of the odd step are zero without loss of
% generality (origin setting)
numLearningDimensions = length(pInputControllerAsymmetricNominal);
numStateDimensions = length(stateVar0_Model);

%% learning rates
alphaEnergyLearningRate = paramFixed.Learner.LearningRate;

%% do the gradient descent

% learning parameter input
pInputNow = pInputControllerAsymmetricNominal;

% state initial condition
% stateVar0_Model = stateVar0_Model;

% initial time
tStart = 0; % re-setting this doesn't matter here

%% initial internal model
AdynamicsModelNow = ...
    zeros(numStateDimensions-1,numLearningDimensions+numStateDimensions); % no +1 for the constant term and -1 to avoid singularity?
AdynamicsEnergyNow = ...
    zeros(1,numLearningDimensions+numStateDimensions); % just on output

%% initializing gradient
gEnergyGradientNow = zeros(size(pInputControllerAsymmetricNominal)); % initial zero

%% initializing various empty arrays
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
%% copying some variables
numStridesToUse = paramFixed.Learner.numStepsToUseForEstimator; 
stateVar0_ModelNow = stateVar0_Model;
pInputNow_ConsideredGood = pInputNow;

%% looping over all strides
for iStride = 1:numIterations % 1 stride = 1 iteration
    if mod(iStride, 200) == 0
        disp(['iStride = ' num2str(iStride)]); % display stride
    end

    %% exploratory noise
    delta_pInputNow1_Noise = noiseSTD*randn(numLearningDimensions,1);
    
    %% compute the gradient step
    if length(alphaEnergyLearningRate) == 1
        delta_pInputNow2_Gradient = alphaEnergyLearningRate*(-gEnergyGradientNow); % static learning rate

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
           delta_pInputNow2_Gradient = alphaStart*(-gEnergyGradientNow); % first tied-belt static learning
        end 
    end

    %% restriction gradient step size if using trust region
    if paramFixed.Learner.shouldWeUseTrustRegion 
        if norm(delta_pInputNow2_Gradient)>paramFixed.Learner.trustRegionSize*sqrt(numLearningDimensions)
            delta_pInputNow2_Gradient = delta_pInputNow2_Gradient/norm(delta_pInputNow2_Gradient);
            delta_pInputNow2_Gradient = delta_pInputNow2_Gradient*paramFixed.Learner.trustRegionSize*sqrt(numLearningDimensions);
        end
    end
       
    %% add gradient and noise steps
    pInputNow_ConsideredGood = pInputNow_ConsideredGood + delta_pInputNow2_Gradient;
    pInputNow_ToTry = pInputNow_ConsideredGood + delta_pInputNow1_Noise;
    
    %% simulate walking
    [EdotNow,stateVar0_ModelNext,tEnd,f_energy,f_constraint,fall_info,y_pos_single] = ...
        fObjective_AsymmetricNominal_8DTo10D(pInputNow_ToTry,stateVar0_ModelNow, ...
        paramControllerGains,paramFixed,tStart,SwingCostCoeff,iStride);
    
    %% multiplicative measurement noise in energy estimates / measurements
    EdotNow = EdotNow*(1+randn*paramFixed.noiseEnergySensory);
    
    %% store all the data so far
    stateVarNowStore =  [stateVarNowStore  stateVar0_ModelNow];
    stateVarNextStore = [stateVarNextStore stateVar0_ModelNext];
    pInputControllerStore_OnesTried = [pInputControllerStore_OnesTried pInputNow_ToTry];
    EdotStore = [EdotStore EdotNow];
    pInputControllerStore_OnesConsideredGood = ...
        [pInputControllerStore_OnesConsideredGood pInputNow_ConsideredGood];

    %% update internal model dynamics (linear)
    if iStride>numStridesToUse
        [AdynamicsModelNow,errorOldDynamicsModel,errorNewDynamicsModel] = ... % find the gradient
            computeModelDynamicsModelRLS_NoSingular(AdynamicsModelNow,stateVarNowStore,...
            stateVarNextStore,pInputControllerStore_OnesTried,numStridesToUse);
    end
    
    %% update energy model dynamics
    if iStride>numStridesToUse
        [AdynamicsEnergyNow,errorOldEnergyModel,errorNewEnergyModel] = computeEnergyDynamicsModelRLS_NoSingular(AdynamicsEnergyNow, ...
            stateVarNowStore,EdotStore,pInputControllerStore_OnesTried,numStridesToUse);
    end
    
    %% compute the error in the linear models
    if iStride>numStridesToUse
        % error in the Dynamics model
        errorOldDynamicsModelStore = [errorOldDynamicsModelStore errorOldDynamicsModel];
        errorNewDynamicsModelStore = [errorNewDynamicsModelStore errorNewDynamicsModel];
        
        % error in the Energy model
        errorOldEnergyModelStore = [errorOldEnergyModelStore errorOldEnergyModel];
        errorNewEnergyModelStore = [errorNewEnergyModelStore errorNewEnergyModel];
    end
    
    %% get the gradient of the steady state energy    
    gEnergyGradientNow = computeSteadyEnergyGradientV2_NoSingular(AdynamicsModelNow, ...
        AdynamicsEnergyNow,pInputNow_ToTry, ...
        numLearningDimensions,numStateDimensions,includeInternalModelOrNot);
    
    gradientEnergyEstimateStore{iStride} = gEnergyGradientNow;
    AdynamicsModelStore{iStride} = AdynamicsModelNow;
    AdynamicsEnergyStore{iStride} = AdynamicsEnergyNow;
    
    %% implements prediction error thresholding. 
    if paramFixed.Learner.shouldWeThresholdPredictionError
        if iStride>numStridesToUse
            %         predictionErrorListNow = ...
            %             [errorOldDynamicsModel; errorNewDynamicsModel; ...
            %             errorOldEnergyModel; errorNewEnergyModel]; % error control on both old and new
            predictionErrorListNow = ... % just with the new model
                [errorOldEnergyModel; errorNewEnergyModel];
            if any(abs(predictionErrorListNow)>paramFixed.Learner.predictionErrorThreshold)
                gEnergyGradientNow = 0*gEnergyGradientNow;
                learningOnOrNotStore = [learningOnOrNotStore; 0];
                % if the gradient has high error, do not take the gradient
                % step 
            else
                learningOnOrNotStore = [learningOnOrNotStore; 1];
            end
        end
    end
    
    %% reset for the next step
    tStart = tEnd;
    stateVar0_ModelNow = stateVar0_ModelNext;
    
    %%
    drawnow;
    
    %% Model falling down check flag
    if fall_info(1) == 1
        fall_flag_vec = fall_info;
        break;
    end
    
    min_vert_pos = [min_vert_pos; y_pos_single'];
end
% fall_flag_vec = [0 numIterations]; % fall didn't happen

end
