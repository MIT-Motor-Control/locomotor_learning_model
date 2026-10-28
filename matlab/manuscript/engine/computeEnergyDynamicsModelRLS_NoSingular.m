function [AdynamicsNew,errorOldModel,errorNewModel] = ...
    computeEnergyDynamicsModelRLS_NoSingular(AdynamicsNow, ...
    stateVarNowStore,EdotStore,pInputControllerNowStore,numStepsToUse)
% Fit the measured objective as a linear function of midstance state and controller.
% Inputs:  AdynamicsNow - current model [G H K]
%          stateVarNowStore - midstance states, one iteration per column
%          EdotStore - measured objective values, one per iteration
%          pInputControllerNowStore - controllers tried, one iteration per column
%          numStepsToUse - number of most recent iterations in the fit
% Outputs: AdynamicsNew - updated model; errorOldModel, errorNewModel - most recent residuals

beta = 0.5;

Aold = AdynamicsNow;

muMeasurement = 0;

InputNowStore = [stateVarNowStore; pInputControllerNowStore];
OutputNextStore = EdotStore;

% Omit the leg angle, as in computeModelDynamicsModelRLS_NoSingular.
InputNowStore = InputNowStore(2:end,:);

[AdynamicsNew,errorOldModel,errorNewModel] = ...
    RLSupdate(Aold,beta,InputNowStore,...
    OutputNextStore,muMeasurement,numStepsToUse);

end
