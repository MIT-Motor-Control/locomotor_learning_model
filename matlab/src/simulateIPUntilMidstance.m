function [tlistTillMidstance, statevarlistTillMidstance] = ...
    simulateIPUntilMidstance(statevar0,tspan,paramFixed,paramController)
% Integrate the inverted-pendulum stance dynamics from heel strike until midstance.
% Inputs and outputs are as in simulateIPUntilEndstance.

options = odeset('Events',@DetectMidstance,'reltol',1e-10,'abstol',1e-10);
[tlistTillMidstance,statevarlistTillMidstance] = ...
    ode45(@singlePendulumODE,tspan,statevar0,options,paramFixed,paramController);

end
