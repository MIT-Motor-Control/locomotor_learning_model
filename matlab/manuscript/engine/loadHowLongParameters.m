function paramFixed = loadHowLongParameters(paramFixed)
% Override the protocol duration with a fixed physical-step budget.

paramFixed.numStepsToLearn = 2000;

if mod(paramFixed.Learner.numStepsPerIteration,2)~=0
    paramFixed.Learner.numStepsPerIteration = paramFixed.Learner.numStepsPerIteration + 1;
end
paramFixed.numIterations = floor(paramFixed.numStepsToLearn/paramFixed.Learner.numStepsPerIteration);

end
