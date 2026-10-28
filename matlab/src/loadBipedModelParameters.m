function paramFixed = loadBipedModelParameters(paramFixed)
% Add the dimensionless biped parameters and objective weights to paramFixed.

paramFixed.includeAccelerationTorque = 1;

% Body mass, leg length and gravity are the units of the dimensionless model.
paramFixed.mbody = 1; paramFixed.leglength = 1; paramFixed.gravg = 1;

% Muscle efficiencies of negative and positive work; their inverses weight
% heel-strike and push-off work in the energy cost.
paramFixed.efficiency_neg = 1.2; paramFixed.efficiency_pos = 0.25;
paramFixed.bPos = 1/paramFixed.efficiency_pos;
paramFixed.bNeg = 1/paramFixed.efficiency_neg;

% Foot mass (body-mass units) and swing-cost parameters for swingCostDoke. The
% controller gains, initial controller and memory are defined only for Coeff = 0.9.
paramFixed.mFoot = 0.05;
paramFixed.swingCost.Coeff = 0.9;
paramFixed.swingCost.alpha = 1.0;

% Weight 1 selects energy; weight 0 selects periodicity.
paramFixed.lambdaEnergyVsPeriodicity = 1;

% Weight 1 selects energy; weight 0 selects symmetry. In paper Eq. 1 the
% symmetry weight is psi = 1 - lambdaEnergyVsSymmetry and the scale M is
% symmetryMultiplier, which brings the symmetry term to the energy scale.
paramFixed.lambdaEnergyVsSymmetry = 0.75;
paramFixed.symmetryMultiplier = 10;

end
