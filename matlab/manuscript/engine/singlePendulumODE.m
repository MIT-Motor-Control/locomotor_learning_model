function dStateVar = singlePendulumODE(t,stateVar,paramFixed,paramController)
% Evaluate inverted-pendulum stance dynamics and stance-foot motion on the moving belt.
% Inputs:  stateVar - [leg angle; angular rate; stance-foot position]
%          paramController - belt-speed and acceleration lists of the push-off foot
% Outputs: dStateVar - time derivative of stateVar

angleTheta = stateVar(1);
dangleTheta = stateVar(2);

% Acceleration is constant within each linear segment of the belt-speed ramp.
paramController.PushoffAccelerationNow = interp1(paramController.tList_BeltSpeed, ...
    paramController.PushoffAccelerationNowList,t,'previous','extrap');

% Backward belt acceleration produces a forward inertial force.
forceDueToAcceleration = -paramController.PushoffAccelerationNow*paramFixed.mbody;
torqueDueToAcceleration = ...
    forceDueToAcceleration*cos(angleTheta)/(paramFixed.mbody*paramFixed.leglength);

if ~paramFixed.includeAccelerationTorque
    torqueDueToAcceleration = 0;
end

ddAngleTheta = ...
    paramFixed.gravg/paramFixed.leglength*sin(angleTheta) + ...
    torqueDueToAcceleration;

paramController.PushoffFootSpeedNow = interp1(paramController.tList_BeltSpeed, ...
    paramController.PushoffFootSpeedNowList,t,'linear','extrap');

dyFoot = paramController.PushoffFootSpeedNow;

dStateVar = [dangleTheta; ddAngleTheta; dyFoot];

end
