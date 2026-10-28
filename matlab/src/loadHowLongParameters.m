function paramFixed = loadHowLongParameters(paramFixed)
% Replace the protocol-derived duration with a fixed budget of 2000 physical steps.
% Optional; call after loadProtocolParameters. The default run does not use it.

paramFixed.numStepsToLearn = 2000;

if mod(paramFixed.Learner.numStepsPerIteration,2)~=0
    paramFixed.Learner.numStepsPerIteration = paramFixed.Learner.numStepsPerIteration + 1;
end
paramFixed.numIterations = floor(paramFixed.numStepsToLearn/paramFixed.Learner.numStepsPerIteration);

end
