function [footSpeed1,footSpeed2] = getTreadmillSpeed(t,beltSpeedsImposed)
% Interpolate the two belt speeds at the requested times.

footSpeed1 = interp1(beltSpeedsImposed.tList,...
    beltSpeedsImposed.footSpeed1List,t,'linear','extrap');
footSpeed2 = interp1(beltSpeedsImposed.tList,...
    beltSpeedsImposed.footSpeed2List,t,'linear','extrap');

end
