function paramFixed = loadProtocolParameters(paramFixed)
% Add the treadmill protocol and the learning-iteration budget to paramFixed.
% Requires paramFixed.Learner.numStepsPerIteration (loadLearnerParameters).

paramFixed.SplitOrTied = 'split';
paramFixed.speedProtocol = 'classic split belt';
paramFixed.transitionTime = 15; % seconds
paramFixed.imposedFootSpeeds = makeTreadmillSpeed_Split(paramFixed);

drawnow;

% Level ground; nonzero slopes have not been tested.
paramFixed.angleSlope = 0;

% Approximate step duration (nondimensional) used to convert protocol time to steps.
nominalStepTime = 1.7;
paramFixed.numStepsToLearn = paramFixed.imposedFootSpeeds.tList(end)/nominalStepTime;
paramFixed.numStepsToLearn = round(paramFixed.numStepsToLearn/100)*100;

% Use complete odd/even step pairs in each learning iteration.
if mod(paramFixed.Learner.numStepsPerIteration,2)~=0
    paramFixed.Learner.numStepsPerIteration = paramFixed.Learner.numStepsPerIteration + 1;
end
paramFixed.numIterations = floor(paramFixed.numStepsToLearn/paramFixed.Learner.numStepsPerIteration);

end
