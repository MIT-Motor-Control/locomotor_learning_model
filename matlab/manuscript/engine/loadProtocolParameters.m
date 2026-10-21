function paramFixed = loadProtocolParameters(paramFixed)
% Set the treadmill protocol and derive the learning-iteration budget.

paramFixed.SplitOrTied = 'split';
paramFixed.speedProtocol = 'classic split belt';
paramFixed.transitionTime = 15; % seconds
paramFixed.imposedFootSpeeds = makeTreadmillSpeed_Split(paramFixed);

drawnow;

% The supported protocol uses level ground.
paramFixed.angleSlope = 0;

nominalStepTime = 1.7;
paramFixed.numStepsToLearn = paramFixed.imposedFootSpeeds.tList(end)/nominalStepTime;
paramFixed.numStepsToLearn = round(paramFixed.numStepsToLearn/100)*100;

% Use complete odd/even step pairs in each learning iteration.
if mod(paramFixed.Learner.numStepsPerIteration,2)~=0
    paramFixed.Learner.numStepsPerIteration = paramFixed.Learner.numStepsPerIteration + 1;
end
paramFixed.numIterations = floor(paramFixed.numStepsToLearn/paramFixed.Learner.numStepsPerIteration);

end
