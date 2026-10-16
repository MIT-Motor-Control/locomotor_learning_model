function paramFixed = loadSensoryNoiseParameters(paramFixed)
% Standard deviations for multiplicative energy noise and additive velocity noise.

%% multiplicative noise for energy measurements
paramFixed.noiseEnergySensory = 0.0001; % this is multiplicative noise

%% additive velocity noise
paramFixed.velocitySensoryNoise = 0; 

end
