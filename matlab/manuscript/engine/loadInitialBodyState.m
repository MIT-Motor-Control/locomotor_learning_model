function stateVar0_Model = loadInitialBodyState(pInputControllerAsymmetricNominal)
% Set the initial midstance state; the state ordering is described in run_sim.

vSwingInitial = 0.35;

stateVar0_Model = [0
    pInputControllerAsymmetricNominal(2)
    0
    0
    vSwingInitial];

end
