function [tlistTillMidstance, statevarlistTillMidstance] = ...
    simulateIPUntilMidstance(statevar0,tspan,paramFixed,paramController)
% Integrate inverted-pendulum dynamics until midstance.

options = odeset('Events',@DetectMidstance,'reltol',1e-8,'abstol',1e-8);
[tlistTillMidstance,statevarlistTillMidstance] = ...
    ode45(@singlePendulumODE,tspan,statevar0,options,paramFixed,paramController);

end
