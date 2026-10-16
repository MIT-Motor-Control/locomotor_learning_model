function paramFixed = loadHowLongParameters(paramFixed)
% Override the protocol duration when called after loadProtocolParameters.

%% How many steps to simulate
paramFixed.numStepsToLearn = 2000;

%% optimization iterations
% Balance both legs before converting the physical-step budget to iterations.
if mod(paramFixed.Learner.numStepsPerIteration,2)~=0
    paramFixed.Learner.numStepsPerIteration = paramFixed.Learner.numStepsPerIteration + 1;
end
paramFixed.numIterations = floor(paramFixed.numStepsToLearn/paramFixed.Learner.numStepsPerIteration);

end
