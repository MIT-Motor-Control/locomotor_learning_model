function [ value, isterminal, direction ] = DetectMidstance(t,stateVar,paramFixed,paramController)
% Stop integration when the body passes over the stance foot.
angleTheta = stateVar(1); 

yNow = (paramFixed.leglength*sin(angleTheta+paramFixed.angleSlope));

value = yNow; % same as theta+paramFixed.angleSlope becoming zero
isterminal = 1;
direction = +1;

end
