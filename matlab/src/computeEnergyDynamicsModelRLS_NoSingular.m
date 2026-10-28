function [AdynamicsNew,errorOldModel,errorNewModel] = ...
    computeEnergyDynamicsModelRLS_NoSingular(AdynamicsNow, ...
    stateVarNowStore,EdotStore,pInputControllerNowStore,numStepsToUse)
% Fit the measured objective as a linear function of the current state and controller.
% Inputs:  AdynamicsNow             - current coefficients
%          stateVarNowStore         - midstance states, one column per iteration
%          EdotStore                - measured objective, one value per iteration
%          pInputControllerNowStore - controllers walked, one column per iteration
%          numStepsToUse            - estimator window (iterations)
% Outputs: AdynamicsNew  - coefficients mapping [x(2:end); p; 1] to the objective
%          errorOldModel - prediction error of the latest iteration before the update
%          errorNewModel - prediction error of the latest iteration after the update

beta = 0.5;

Aold = AdynamicsNow;

muMeasurement = 0;

InputNowStore = [stateVarNowStore; pInputControllerNowStore];
OutputNextStore = EdotStore;

% Drop the leg angle, as in computeModelDynamicsModelRLS_NoSingular.
InputNowStore = InputNowStore(2:end,:);

[AdynamicsNew,errorOldModel,errorNewModel] = ...
    RLSupdate(Aold,beta,InputNowStore,...
    OutputNextStore,muMeasurement,numStepsToUse);

end
