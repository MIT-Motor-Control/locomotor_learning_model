function C = swingCostDoke(tDurationOfChange,vInitial,vFinal,vBody,paramFixed)
% Approximate leg swing cost as a smoothed power of force rate (Doke et al. 2005).
% Inputs:  tDurationOfChange - duration of the foot-speed change; vInitial, vFinal - foot speeds
%          vBody - body speed; paramFixed - mFoot and swingCost.Coeff, swingCost.alpha
% Outputs: C - swing cost

vInitialRelative = vInitial-vBody;
vFinalRelative = vFinal-vBody;

deltaV = vFinalRelative-vInitialRelative;
Force = paramFixed.mFoot*deltaV/tDurationOfChange;
ForceRate = Force/tDurationOfChange;

% Smooths the cost near zero force rate.
epsilon = 0.01;

C = paramFixed.swingCost.Coeff*sqrt(ForceRate^2+epsilon^2)^paramFixed.swingCost.alpha;

end
