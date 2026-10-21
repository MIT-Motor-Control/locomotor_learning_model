function [ value, isterminal, direction ] = DetectEndstance(t,stateVar,paramFixed,paramController)
% Stop integration when the stance leg reaches the controller target angle.

angleTheta = stateVar(1);

value = angleTheta-paramController.theta_end_thisStep;
isterminal = 1;
direction = +1;

end
