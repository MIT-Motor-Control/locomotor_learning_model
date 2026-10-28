function [footSpeed1,footSpeed2] = getTreadmillSpeed(t,beltSpeedsImposed)
% Interpolate the two belt speeds at the requested times.
% Inputs:  t                 - times (nondimensional), scalar or vector
%          beltSpeedsImposed - profile from makeTreadmillSpeed_Split or makeTreadmillSpeed_Tied
% Outputs: footSpeed1, footSpeed2 - belt 1 and belt 2 speeds at t

footSpeed1 = interp1(beltSpeedsImposed.tList,...
    beltSpeedsImposed.footSpeed1List,t,'linear','extrap');
footSpeed2 = interp1(beltSpeedsImposed.tList,...
    beltSpeedsImposed.footSpeed2List,t,'linear','extrap');

end
