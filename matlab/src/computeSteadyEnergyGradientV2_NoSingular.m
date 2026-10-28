function gEnergy = ...
    computeSteadyEnergyGradientV2_NoSingular(AdynamicsModel,AdynamicsEnergy, ...
    pInputNow,numLearningDimensions,numStateDimensions,includeInternalModelOrNot)
% Return the gradient of the steady-state objective with respect to the controller.
% Inputs:  AdynamicsModel  - state model from computeModelDynamicsModelRLS_NoSingular
%          AdynamicsEnergy - objective model from computeEnergyDynamicsModelRLS_NoSingular
%          pInputNow       - unused; the gradient of the linear models is constant
%          numLearningDimensions     - number of learned controller parameters
%          numStateDimensions        - number of midstance state variables
%          includeInternalModelOrNot - 1 includes the state-mediated term, 0 omits it
% Outputs: gEnergy - gradient, numLearningDimensions x 1

% The leg angle is excluded from the fitted state.
numStateDimensions = numStateDimensions-1;

Inow = eye(numStateDimensions);

% State model: x_next = A*x + B*p + C.
A = AdynamicsModel(1:numStateDimensions,1:numStateDimensions);
B = AdynamicsModel(1:numStateDimensions,numStateDimensions+1:numStateDimensions+numLearningDimensions);

% Energy model: E = G*x + H*p + K.
G = AdynamicsEnergy(1,1:numStateDimensions);
H = AdynamicsEnergy(1,numStateDimensions+1:numStateDimensions+numLearningDimensions);
% At steady state, dE/dp = G*((I-A)\B)+H; intercepts drop out.
gEnergy = (G*((Inow-A)\B)*includeInternalModelOrNot+H)';

end
