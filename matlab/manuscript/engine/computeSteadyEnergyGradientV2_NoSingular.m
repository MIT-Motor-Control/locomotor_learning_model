function gEnergy = ...
    computeSteadyEnergyGradientV2_NoSingular(AdynamicsModel,AdynamicsEnergy, ...
    pInputNow,numLearningDimensions,numStateDimensions,includeInternalModelOrNot)
% Differentiate the steady-state objective with respect to the controller parameters.
% Inputs:  AdynamicsModel - state model [A B C]; AdynamicsEnergy - objective model [G H K]
%          pInputNow - unused
%          numLearningDimensions - number of controller parameters
%          numStateDimensions - state size including the omitted leg angle
%          includeInternalModelOrNot - 1 to include the state-mediated term, 0 for H only
% Outputs: gEnergy - gradient, column vector

% The fitted state omits the midstance leg angle.
numStateDimensions = numStateDimensions-1;

Inow = eye(numStateDimensions);

% State model: x_next = A*x + B*p + C.
A = AdynamicsModel(1:numStateDimensions,1:numStateDimensions);
B = AdynamicsModel(1:numStateDimensions,numStateDimensions+1:numStateDimensions+numLearningDimensions);

% Objective model: E = G*x + H*p + K.
G = AdynamicsEnergy(1,1:numStateDimensions);
H = AdynamicsEnergy(1,numStateDimensions+1:numStateDimensions+numLearningDimensions);

% At steady state, dE/dp = G*((I-A)\B)+H; intercepts drop out.
gEnergy = (G*((Inow-A)\B)*includeInternalModelOrNot+H)';

end
