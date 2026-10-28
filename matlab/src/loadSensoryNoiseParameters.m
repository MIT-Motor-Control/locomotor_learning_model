function paramFixed = loadSensoryNoiseParameters(paramFixed)
% Add sensory noise levels to paramFixed.
% noiseEnergySensory is the relative (multiplicative) noise SD of the measured
% objective and the additive noise SD of the sensed belt speeds;
% velocitySensoryNoise is the additive noise SD of midstance speed.

paramFixed.noiseEnergySensory = 0.0001;

paramFixed.velocitySensoryNoise = 0;

end
