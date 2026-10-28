function paramFixed = loadHowLongParameters(paramFixed)
% Replace the protocol-derived step budget with 2000 steps.

paramFixed.numStepsToLearn = 2000;

if mod(paramFixed.Learner.numStepsPerIteration,2)~=0
    paramFixed.Learner.numStepsPerIteration = paramFixed.Learner.numStepsPerIteration + 1;
end
paramFixed.numIterations = floor(paramFixed.numStepsToLearn/paramFixed.Learner.numStepsPerIteration);

end
