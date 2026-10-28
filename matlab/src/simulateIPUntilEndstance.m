function [tlistTillEndstance, statevarlistTillEndstance] = ...
    simulateIPUntilEndstance(statevar0,tspan,paramFixed,paramController)
% Integrate the inverted-pendulum stance dynamics from midstance until endstance.
% Inputs:  statevar0       - [stance-leg angle; angular rate; stance-foot position]
%          tspan           - output times (nondimensional); the event ends integration
%          paramFixed      - model parameters
%          paramController - belt tables and this step's endstance angle
% Outputs: tlistTillEndstance, statevarlistTillEndstance - times and states up to the event

options = odeset('Events',@DetectEndstance,'reltol',1e-10,'abstol',1e-10);
[tlistTillEndstance,statevarlistTillEndstance] = ...
    ode45(@singlePendulumODE,tspan,statevar0,options,paramFixed,paramController);

end
