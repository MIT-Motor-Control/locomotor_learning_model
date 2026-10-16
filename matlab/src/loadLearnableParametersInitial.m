function pInputControllerAsymmetricNominal = ...
    loadLearnableParametersInitial(paramFixed)

% Odd-step parameters followed by even-step parameters: endstance angle,
% midstance belt-frame speed, push-off impulse, position and position integral.
% Odd-step position and integral define the origin and are fixed at zero.
% Odd and even nominal controllers may differ.
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
