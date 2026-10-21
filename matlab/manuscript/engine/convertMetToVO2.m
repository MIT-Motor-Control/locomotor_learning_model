function [tSpan,EmetVO2List] = convertMetToVO2(params)
% Apply a first-order metabolic transient with a 44-second time constant.
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
[tSpan,EmetVO2List] = ode45(@odeMetToVO2MET,tSpan,EmetS_0,options,params);

end

function dEmetVO2dt = odeMetToVO2MET(t,EmetS,params)

EmetNow = interp1(params.tList,params.EmetRateList,t,'linear','extrap');

dEmetVO2dt = params.k*(EmetNow-EmetS);

end
