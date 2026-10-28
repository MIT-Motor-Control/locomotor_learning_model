function pInputControllerAsymmetricNominal = ...
    loadLearnableParametersInitial(paramFixed)
% Return the initial 10-D nominal controller for paramFixed.swingCost.Coeff.
% Entries 1-5 (odd steps) and 6-10 (even steps) are the nominal endstance leg
% angle, midstance speed relative to the belt, push-off impulse, midstance foot
% position and summed midstance foot position.

switch paramFixed.swingCost.Coeff
    case 0.9
        pInputControllerAsymmetricNominal = [0.328221262798818
            0.310751796902254
            0.153556843539029
            0
            0
            0.328221491356562
            0.310751694570805
            0.153557221281688
            -0.000000038953735
            -0.000000038953735];
end

end
