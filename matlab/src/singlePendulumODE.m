function dStateVar = singlePendulumODE(t,stateVar,paramFixed,paramController)
% Return the state derivative of the inverted-pendulum stance phase on a moving belt.
% The stance foot moves at the belt speed in PushoffFootSpeedNowList, and belt
% acceleration adds an inertial torque.
% Inputs:  t               - time (nondimensional)
%          stateVar        - [stance-leg angle; angular rate; stance-foot position]
%          paramFixed      - model parameters
%          paramController - belt-speed and belt-acceleration tables for this step
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
