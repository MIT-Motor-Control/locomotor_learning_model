function [AdynamicsNew,errorOldModel,errorNewModel] = ...
    computeModelDynamicsModelRLS_NoSingular(AdynamicsNow, ...
    stateVarNowStore,stateVarNextStore,pInputControllerNowStore,numStepsToUse)
% computes a linear dynamic model by computing the gradient of state with
% respect to state and action (internal model)

beta = 0.5; % unused compatibility argument

Aold = AdynamicsNow; 

muMeasurement = 0; % unused compatibility argument

InputNowStore = [stateVarNowStore; pInputControllerNowStore];
OutputNextStore = stateVarNextStore;

% remove 1st row of zeros to avoid singularity
InputNowStore = InputNowStore(2:end,:);
OutputNextStore = OutputNextStore(2:end,:);

[AdynamicsNew,errorOldModel,errorNewModel] = ...
    RLSupdate(Aold,beta,InputNowStore,...
    OutputNextStore,muMeasurement,numStepsToUse);


end
