function stateVar0_Model = loadInitialBodyState(pInputControllerAsymmetricNominal)
% Set the initial state at midstance.

vSwingInitial = 0.35;

% State: leg angle, angular rate, foot position, integrated foot position, swing speed.
stateVar0_Model = [0
    pInputControllerAsymmetricNominal(2)
    0
    0
    vSwingInitial];

end
