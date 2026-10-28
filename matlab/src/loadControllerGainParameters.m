function paramController = loadControllerGainParameters(paramFixed)
% Return the fixed step-to-step feedback gains for paramFixed.swingCost.Coeff.
% legAngle_gain_* and pushoff_gain_* adjust the endstance leg angle and the
% push-off impulse in proportion to the midstance deviations of speed (ydot),
% foot position (y), summed foot position (SUMy) and belt speed from nominal.

switch paramFixed.swingCost.Coeff
    case 0.9
        paramController.pushoff_gain_ydot  = -0.521088310893437;
        paramController.legAngle_gain_ydot =  0.279308152405233;
        paramController.pushoff_gain_y =  -0.085939152240409;
        paramController.legAngle_gain_y =  -0.028999999999996;
        paramController.pushoff_gain_SUMy =  -0.007302228127922;
        paramController.legAngle_gain_SUMy =  0.002019507100848;
        paramController.legAngle_gain_BeltSpeed = -0.075044622818653;
        paramController.pushoff_gain_BeltSpeed = -1.055930819923483;
end

% Not read by the simulation.
paramController.ControllerBeltFrameVsLabFrame = 1;

end
