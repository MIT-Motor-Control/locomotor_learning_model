function [AdynamicsNew,errorOldModel,errorNewModel] = ...
    computeModelDynamicsModelRLS_NoSingular(AdynamicsNow, ...
    stateVarNowStore,stateVarNextStore,pInputControllerNowStore,numStepsToUse)
% Fit the next midstance state as a linear function of the current state and controller.
% Inputs:  AdynamicsNow             - current coefficients
%          stateVarNowStore         - midstance states, one column per iteration
%          stateVarNextStore        - midstance states at the end of each iteration
%          pInputControllerNowStore - controllers walked, one column per iteration
%          numStepsToUse            - estimator window (iterations)
% Outputs: AdynamicsNew  - coefficients mapping [x(2:end); p; 1] to x_next(2:end)
%          errorOldModel - prediction error of the latest iteration before the update
%          errorNewModel - prediction error of the latest iteration after the update

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
