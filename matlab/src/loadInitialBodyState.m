function stateVar0_Model = loadInitialBodyState(pInputControllerAsymmetricNominal)
% Return the initial midstance state.
% Inputs:  pInputControllerAsymmetricNominal - 10-D nominal controller; entry 2
%          (nominal midstance speed) sets the initial angular rate
% Outputs: stateVar0_Model - [stance-leg angle; angular rate; stance-foot position;
%          summed midstance foot-position error; swing-foot speed (lab frame)]

vSwingInitial = 0.35;

stateVar0_Model = [0
    pInputControllerAsymmetricNominal(2)
    0
    0
    vSwingInitial];

end
