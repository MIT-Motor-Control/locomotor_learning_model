function [AdynamicsNew,errorOldModel,errorNewModel] = ...
    computeModelDynamicsModelRLS_NoSingular(AdynamicsNow, ...
    stateVarNowStore,stateVarNextStore,pInputControllerNowStore,numStepsToUse)
% Fit the next midstance state as a linear function of the current state and controller.
% Inputs:  AdynamicsNow - current model [A B C]
%          stateVarNowStore, stateVarNextStore - midstance states before and after each iteration
%          pInputControllerNowStore - controllers tried, one iteration per column
%          numStepsToUse - number of most recent iterations in the fit
% Outputs: AdynamicsNew - updated model; errorOldModel, errorNewModel - most recent residuals

beta = 0.5;

Aold = AdynamicsNow;

muMeasurement = 0;

InputNowStore = [stateVarNowStore; pInputControllerNowStore];
OutputNextStore = stateVarNextStore;

% The leg angle is zero at midstance; drop it to keep the regression nonsingular.
InputNowStore = InputNowStore(2:end,:);
OutputNextStore = OutputNextStore(2:end,:);

[AdynamicsNew,errorOldModel,errorNewModel] = ...
    RLSupdate(Aold,beta,InputNowStore,...
    OutputNextStore,muMeasurement,numStepsToUse);

end
