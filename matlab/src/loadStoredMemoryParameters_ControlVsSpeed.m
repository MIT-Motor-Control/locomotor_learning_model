function paramFixed = loadStoredMemoryParameters_ControlVsSpeed(paramFixed)
% Add the initial controller memory to paramFixed.storedmemory.
% The memory predicts the 8-D controller for a belt-speed context c as
% nominalControl + controlSlopeVsContext*(c - nominalContext).

switch paramFixed.swingCost.Coeff
    case 0.9
        paramFixed.storedmemory.nominalControl = [0.328221262798818
            0.310751796902254
            0.153556843539029
            0.328221491356562
            0.310751694570805
            0.153557221281688
            -0.000000038953735
            -0.000000038953735];
end

paramFixed.storedmemory.nominalContext = [-0.35; -0.35];

paramFixed.storedmemory.controlSlopeVsContext = ...
    zeros(8,2);

end
