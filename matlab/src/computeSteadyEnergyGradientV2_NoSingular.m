function gEnergy = ...
    computeSteadyEnergyGradientV2_NoSingular(AdynamicsModel,AdynamicsEnergy, ...
    pInputNow,numLearningDimensions,numStateDimensions,includeInternalModelOrNot)
% Differentiate steady-state energy with respect to controller parameters.

% The fixed midstance coordinate is excluded from the fitted state.
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
