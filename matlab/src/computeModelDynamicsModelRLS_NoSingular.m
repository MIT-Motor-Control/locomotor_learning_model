function [AdynamicsNew,errorOldModel,errorNewModel] = ...
    computeModelDynamicsModelRLS_NoSingular(AdynamicsNow, ...
    stateVarNowStore,stateVarNextStore,pInputControllerNowStore,numStepsToUse)
% Fit the next-state model from recent states and controller parameters.

beta = 0.5;

Aold = AdynamicsNow;

muMeasurement = 0;

InputNowStore = [stateVarNowStore; pInputControllerNowStore];
OutputNextStore = stateVarNextStore;

% Midstance fixes the first state coordinate at zero; omit it from the fit.
InputNowStore = InputNowStore(2:end,:);
OutputNextStore = OutputNextStore(2:end,:);

[AdynamicsNew,errorOldModel,errorNewModel] = ...
    RLSupdate(Aold,beta,InputNowStore,...
    OutputNextStore,muMeasurement,numStepsToUse);

end
