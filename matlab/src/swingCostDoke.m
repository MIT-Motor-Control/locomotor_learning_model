function C = swingCostDoke(tDurationOfChange,vInitial,vFinal,vBody,paramFixed)
% Return the leg-swing cost as a smoothed power of the foot force rate (after Doke et al., 2005).
% Inputs:  tDurationOfChange - duration of the speed change (nondimensional)
%          vInitial, vFinal  - initial and final swing-foot speeds (lab frame)
%          vBody             - body speed; it cancels in the speed change
%          paramFixed        - uses mFoot, swingCost.Coeff and swingCost.alpha
% Outputs: C - swing cost

vInitialRelative = vInitial-vBody;
vFinalRelative = vFinal-vBody;

deltaV = vFinalRelative-vInitialRelative;
Force = paramFixed.mFoot*deltaV/tDurationOfChange;
ForceRate = Force/tDurationOfChange;

% epsilon keeps the cost differentiable at zero force rate.
epsilon = 0.01;

C = paramFixed.swingCost.Coeff*sqrt(ForceRate^2+epsilon^2)^paramFixed.swingCost.alpha;

end
