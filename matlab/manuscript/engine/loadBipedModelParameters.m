function paramFixed = loadBipedModelParameters(paramFixed)
% Set the dimensionless biped parameters and objective weights.

paramFixed.includeAccelerationTorque = 1;

% Mass, leg length and gravity define the dimensionless model.
paramFixed.mbody = 1; paramFixed.leglength = 1; paramFixed.gravg = 1;

paramFixed.efficiency_neg = 1.2; paramFixed.efficiency_pos = 0.25;
paramFixed.bPos = 1/paramFixed.efficiency_pos;
paramFixed.bNeg = 1/paramFixed.efficiency_neg;

paramFixed.mFoot = 0.05;
paramFixed.swingCost.Coeff = 0.9;
paramFixed.swingCost.alpha = 1.0;

% Weight 1 selects energy; weight 0 selects periodicity.
paramFixed.lambdaEnergyVsPeriodicity = 1;

% Weight 1 selects energy; weight 0 selects symmetry.
paramFixed.lambdaEnergyVsSymmetry = 0.75;
% Scale symmetry gradients to a magnitude comparable with energy gradients.
paramFixed.symmetryMultiplier = 10;

end
