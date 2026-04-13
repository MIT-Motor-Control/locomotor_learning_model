function paramFixed = loadProtocolParameters(paramFixed)
% adaptation protocol parameters: how the treadmill speed is changed, and
% whether the adaptation protocol is on a split-belt treadmill or a tied
% belt treadmill.

%% what speed protocol to use: split belt changes
paramFixed.SplitOrTied = 'split';
paramFixed.speedProtocol = 'classic split belt';
paramFixed.transitionTime = 15; % in seconds. 
paramFixed.imposedFootSpeeds = makeTreadmillSpeed_Split(paramFixed);

%% what speed protocol to use: tied belt changes
% more familiar task of walking on a regular treadmill with speed changes
% paramFixed.transitionTime = 3; % in seconds. 
% paramFixed.SplitOrTied = 'tied';
% paramFixed.speedProtocol = 'four speed changes';
% paramFixed.imposedFootSpeeds = makeTreadmillSpeed_Tied(paramFixed);

drawnow;

%%
paramFixed.angleSlope = 0;  
% do not change: the code has not been tested for non-zero values

%% We get the simulation duration from the protocol, but you can override 
% this by changing the values in loadHowLongParameters.m function and
% uncommenting that function in the root program

%% How many steps to simulate
nominalStepTime = 1.7; 
paramFixed.numStepsToLearn = paramFixed.imposedFootSpeeds.tList(end)/nominalStepTime;
paramFixed.numStepsToLearn = round(paramFixed.numStepsToLearn/100)*100; % round to the nearest hundred

%% optimization iterations
paramFixed.numIterations = floor(paramFixed.numStepsToLearn/paramFixed.Learner.numStepsPerIteration);

if mod(paramFixed.Learner.numStepsPerIteration,2)~=0 
    paramFixed.Learner.numStepsPerIteration = paramFixed.numStepsPerIteration + 1;
end


end  % checked and essential