function dStateVar = singlePendulumODE(t,stateVar,paramFixed,paramController)
% Inverted-pendulum dynamics with a massless stance leg and point body mass.

% unwrapping the state variables
angleTheta = stateVar(1); % angle wrt ground
dangleTheta = stateVar(2);

%% inertial force for acceleration 

% interpolate to get the right acceleration
paramController.PushoffAccelerationNow = interp1(paramController.tList_BeltSpeed, ...
    paramController.PushoffAccelerationNowList,t,'previous','extrap'); % 'previous' because we want constant acceleration during each piecewise linear domain

% forward 'inertial force' due to backward belt acceleration
forceDueToAcceleration = -paramController.PushoffAccelerationNow*paramFixed.mbody;
torqueDueToAcceleration = ...
    forceDueToAcceleration*cos(angleTheta)/(paramFixed.mbody*paramFixed.leglength);

if ~paramFixed.includeAccelerationTorque
    torqueDueToAcceleration = 0;
end

% angular acceleration of the stance leg
ddAngleTheta = ...
    paramFixed.gravg/paramFixed.leglength*sin(angleTheta) + ...
    torqueDueToAcceleration; 

% interpolate to get the right speed
paramController.PushoffFootSpeedNow = interp1(paramController.tList_BeltSpeed, ...
    paramController.PushoffFootSpeedNowList,t,'linear','extrap');

% integrating yFoot position so that it is easily accessible
dyFoot = paramController.PushoffFootSpeedNow; % treadmill speed

dStateVar = [dangleTheta; ddAngleTheta; dyFoot];

end
