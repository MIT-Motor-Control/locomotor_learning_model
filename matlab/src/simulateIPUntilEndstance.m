function [tlistTillEndstance, statevarlistTillEndstance] = ...
    simulateIPUntilEndstance(statevar0,tspan,paramFixed,paramController)
% Integrate inverted-pendulum dynamics until endstance.

options = odeset('Events',@DetectEndstance,'reltol',1e-10,'abstol',1e-10);
[tlistTillEndstance,statevarlistTillEndstance] = ...
    ode45(@singlePendulumODE,tspan,statevar0,options,paramFixed,paramController);

end
