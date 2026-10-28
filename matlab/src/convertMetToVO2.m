function [tSpan,EmetVO2List] = convertMetToVO2(params)
% Filter the energy rate through first-order VO2 kinetics (44-s time constant).
% Inputs:  params.tList        - iteration end times (nondimensional)
%          params.EmetRateList - iteration-averaged energy rates
% Outputs: tSpan       - output times
%          EmetVO2List - filtered energy rate at tSpan
if numel(params.tList) == 1
    tSpan = params.tList;
    EmetVO2List = params.EmetRateList;
    return
end

timeConstant = 44; % seconds
g = 9.81; LegLength = 0.95;
timeScaling = sqrt(LegLength/g);
timeConstant = timeConstant/timeScaling;
params.k = 1/timeConstant;

% Initialize from up to 30 iteration-averaged energy rates.
EmetS_0 = mean(params.EmetRateList(1:min(30, numel(params.EmetRateList))));
tSpan = params.tList;
options = odeset('reltol',1e-8,'abstol',1e-8);
if numel(tSpan) == 2
    % With a two-element span ode45 returns every internal step; keep the two end points.
    [~,EmetVO2List] = ode45(@odeMetToVO2MET,[tSpan(1) mean(tSpan) tSpan(2)],EmetS_0,options,params);
    EmetVO2List = EmetVO2List([1 3]);
    tSpan = tSpan(:);
else
    [tSpan,EmetVO2List] = ode45(@odeMetToVO2MET,tSpan,EmetS_0,options,params);
end

end

function dEmetVO2dt = odeMetToVO2MET(t,EmetS,params)
% Return the rate at which the filtered value relaxes toward the current energy rate.

EmetNow = interp1(params.tList,params.EmetRateList,t,'linear','extrap');

dEmetVO2dt = params.k*(EmetNow-EmetS);

end
