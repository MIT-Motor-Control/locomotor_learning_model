function [tSpan,EmetVO2List] = convertMetToVO2(params)
% Apply a first-order metabolic transient with a 44-second time constant.
% Inputs:  params.tList - model times; params.EmetRateList - metabolic rates at those times
% Outputs: tSpan - output times; EmetVO2List - lagged rate, approximating indirect calorimetry
if numel(params.tList) == 1
    tSpan = params.tList;
    EmetVO2List = params.EmetRateList;
    return
end

timeConstant = 44; % seconds
g = 9.81; LegLength = 0.95;
% Convert the time constant to model time units of sqrt(LegLength/g) seconds.
timeScaling = sqrt(LegLength/g);
timeConstant = timeConstant/timeScaling;
params.k = 1/timeConstant;

% Initialize from up to 30 iteration-averaged energy rates.
EmetS_0 = mean(params.EmetRateList(1:min(30, numel(params.EmetRateList))));
tSpan = params.tList;
options = odeset('reltol',1e-8,'abstol',1e-8);
[tSpan,EmetVO2List] = ode45(@odeMetToVO2MET,tSpan,EmetS_0,options,params);

end

function dEmetVO2dt = odeMetToVO2MET(t,EmetS,params)
% Relax the measured rate toward the instantaneous rate.

EmetNow = interp1(params.tList,params.EmetRateList,t,'linear','extrap');

dEmetVO2dt = params.k*(EmetNow-EmetS);

end
