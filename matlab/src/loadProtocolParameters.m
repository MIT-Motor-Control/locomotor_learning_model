function paramFixed = loadProtocolParameters(paramFixed)
% adaptation protocol parameters: how the treadmill speed is changed, and
% whether the adaptation protocol is on a split-belt treadmill or a tied
% belt treadmill.

%% what speed protocol to use: split belt changes
paramFixed.SplitOrTied = 'split';
paramFixed.speedProtocol = 'classic split belt';
paramFixed.transitionTime = 15; % in seconds. 
paramFixed.imposedFootSpeeds = makeTreadmillSpeed_Split(paramFixed);

% For tied walking, use SplitOrTied = 'tied', speedProtocol =
% 'four speed changes', a 3-second transition and makeTreadmillSpeed_Tied.

drawnow;

%%
paramFixed.angleSlope = 0;  
% do not change: the code has not been tested for non-zero values

% Duration follows the protocol. Call loadHowLongParameters afterward
% to use its step-count override.

%% How many steps to simulate
nominalStepTime = 1.7; 
paramFixed.numStepsToLearn = paramFixed.imposedFootSpeeds.tList(end)/nominalStepTime;
paramFixed.numStepsToLearn = round(paramFixed.numStepsToLearn/100)*100; % round to the nearest hundred

%% optimization iterations
% Balance both legs before converting the physical-step budget to iterations.
if mod(paramFixed.Learner.numStepsPerIteration,2)~=0
    paramFixed.Learner.numStepsPerIteration = paramFixed.Learner.numStepsPerIteration + 1;
end
paramFixed.numIterations = floor(paramFixed.numStepsToLearn/paramFixed.Learner.numStepsPerIteration);

end
