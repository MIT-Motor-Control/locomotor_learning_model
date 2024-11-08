% Setup
loadInitialConditions; % Load or define initial conditions
loadParameters; % Load model, controller, and simulation parameters

% Flags
includeSwingCost = [true, false]; % Array to toggle swing cost
resultingAsymmetries = zeros(1, length(includeSwingCost)); % Store results

% Simulation Loop
for i = 1:length(includeSwingCost)
    paramFixed.includeSwingCost = includeSwingCost(i); % Toggle swing cost inclusion
    [results, stepLengthAsymmetry] = simulateWalking(paramFixed); % Custom function to run simulation and return asymmetry
    resultingAsymmetries(i) = stepLengthAsymmetry; % Store resulting asymmetry
end

% Plot Results
figure;
plot(includeSwingCost, resultingAsymmetries, '-o');
xlabel('Include Swing Cost');
ylabel('Resulting Step Length Asymmetry');
title('Effect of Swing Cost on Step Length Asymmetry');
