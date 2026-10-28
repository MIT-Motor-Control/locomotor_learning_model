function [ value, isterminal, direction ] = DetectMidstance(t,stateVar,paramFixed,paramController)
% Stop integration when the body passes over the stance foot (ode45 event).
angleTheta = stateVar(1);

yNow = (paramFixed.leglength*sin(angleTheta+paramFixed.angleSlope));

value = yNow;
isterminal = 1;
direction = +1;

end
