function [AdynamicsNew,errorOldModel,errorNewModel] = ...
    computeEnergyDynamicsModelRLS_NoSingular(AdynamicsNow, ...
    stateVarNowStore,EdotStore,pInputControllerNowStore,numStepsToUse)
% Fit stride energy from recent states and controller parameters.

beta = 0.5;

Aold = AdynamicsNow;

muMeasurement = 0;

InputNowStore = [stateVarNowStore; pInputControllerNowStore];
OutputNextStore = EdotStore;

% Midstance fixes the first state coordinate at zero; omit it from the fit.
InputNowStore = InputNowStore(2:end,:);

[AdynamicsNew,errorOldModel,errorNewModel] = ...
    RLSupdate(Aold,beta,InputNowStore,...
    OutputNextStore,muMeasurement,numStepsToUse);

end
