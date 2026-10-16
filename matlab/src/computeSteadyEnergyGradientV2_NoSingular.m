function gEnergy = ...
    computeSteadyEnergyGradientV2_NoSingular(AdynamicsModel,AdynamicsEnergy, ...
    pInputNow,numLearningDimensions,numStateDimensions,includeInternalModelOrNot)
% Differentiate steady-state energy with respect to controller parameters.

% because we removed one state to avoid singular matrices
numStateDimensions = numStateDimensions-1;

Inow = eye(numStateDimensions);

% state dynamics: equation: x_i+1 = A x_i + B p_i + C
A = AdynamicsModel(1:numStateDimensions,1:numStateDimensions);
B = AdynamicsModel(1:numStateDimensions,numStateDimensions+1:numStateDimensions+numLearningDimensions);

% energy dynamics: equation: E_i = G x_i + H p_i + K
G = AdynamicsEnergy(1,1:numStateDimensions);
H = AdynamicsEnergy(1,numStateDimensions+1:numStateDimensions+numLearningDimensions);
% At steady state, dE/dp = G*((I-A)\B)+H; constant terms drop out.
gEnergy = (G*((Inow-A)\B)*includeInternalModelOrNot+H)';

end
