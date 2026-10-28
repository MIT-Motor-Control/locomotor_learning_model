function [ value, isterminal, direction ] = DetectEndstance(t,stateVar,paramFixed,paramController)
% Stop integration when the stance-leg angle reaches theta_end_thisStep (ode45 event).
angleTheta = stateVar(1);

value = angleTheta-paramController.theta_end_thisStep;
isterminal = 1;
direction = +1;

end
