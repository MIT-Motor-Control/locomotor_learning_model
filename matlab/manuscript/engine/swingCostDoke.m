function C = swingCostDoke(tDurationOfChange,vInitial,vFinal,vBody,paramFixed)
% Approximate swing cost with a smoothed force-rate power law.

vInitialRelative = vInitial-vBody;
vFinalRelative = vFinal-vBody;

deltaV = vFinalRelative-vInitialRelative;
Force = paramFixed.mFoot*deltaV/tDurationOfChange;
ForceRate = Force/tDurationOfChange;

epsilon = 0.01;

C = paramFixed.swingCost.Coeff*sqrt(ForceRate^2+epsilon^2)^paramFixed.swingCost.alpha;

end
