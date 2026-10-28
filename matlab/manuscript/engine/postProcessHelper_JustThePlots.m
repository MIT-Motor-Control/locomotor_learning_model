function [time_list, stepLengthAsymmetryList, stepTimeAsymmetryList, energy_result, steptime_fast, steptime_slow, stepLength_fast, stepLength_slow] = postProcessHelper_JustThePlots(stateVar0,tStore,stateStore, ...
    EmetStore,EmetPerTimeStore,tStepStore,paramController,paramFixed,delta, ...
    EworkPushoffStore,EworkHeelstrikeStore, EdotStore_IterationAverage, ...
    tTotalIterationStore)
% Compute step asymmetries from replayed steps and return them with the iteration-averaged power.
% Inputs:  tStore, stateStore - stance trajectories; EmetStore - per-step energies
%          paramFixed - leg length and slope
%          EdotStore_IterationAverage, tTotalIterationStore - power and duration per iteration
%          The remaining inputs are unused.
% Outputs: time_list - cumulative model time at the end of each iteration
%          stepLengthAsymmetryList - (fast-slow)/(fast+slow) step length, one value per stride
%          stepTimeAsymmetryList - (slow-fast)/(slow+fast) stance time, one value per stride
%          energy_result - metabolic power averaged over each iteration
%          steptime_fast, steptime_slow - stance durations on each belt
%          stepLength_fast, stepLength_slow - lengths of steps landing on each belt
% The convertMetToVO2 transient and the knee trajectories are computed but not returned.

paramFixed.numSteps = length(EmetStore);
for iStep = 1:paramFixed.numSteps
    thetaList = stateStore{iStep}(:,1);

    yBodyList_inFootFrame = paramFixed.leglength*sin(thetaList+paramFixed.angleSlope);
    zBodyList_inSlopeFrame = paramFixed.leglength*cos(thetaList+paramFixed.angleSlope);

    if rem(iStep,2)==0
        s1 = 'r';
    else
        s1 = 'b';
    end
end

for iStep = 1:paramFixed.numSteps
    yFootStanceList{iStep} = stateStore{iStep}(:,3);
end

for iStep = 1:paramFixed.numSteps

    if (iStep<=1)||(iStep>=paramFixed.numSteps)

        yFootSwingList{iStep} = NaN*ones(size(tStore{iStep}));
        zFootSwingList{iStep} = NaN*ones(size(tStore{iStep}));

    else

        swingFootBegin = yFootStanceList{iStep-1}(end);
        swingFootEnd = yFootStanceList{iStep+1}(1);

        yFootSwingList{iStep} = interp1([tStore{iStep}(1) tStore{iStep}(end)],[swingFootBegin swingFootEnd],tStore{iStep});

        footHeight = 0.03;
        Amean = (0+footHeight)/2;
        Atemp = (0-footHeight)/2;
        Tswing = (tStore{iStep}(end)-tStore{iStep}(1));
        zFootSwingList{iStep} = Amean+Atemp*cos(2*pi*(tStore{iStep}-tStore{iStep}(1))/(Tswing));

    end

    thetaList = stateStore{iStep}(:,1);
    yBodyList_inFootFrame = paramFixed.leglength*sin(thetaList+paramFixed.angleSlope);
    zBodyList_inSlopeFrame = paramFixed.leglength*cos(thetaList);

    yBodyList_inSlopeFrame = yBodyList_inFootFrame+yFootStanceList{iStep};

    [ySwingKneeList{iStep},zSwingKneeList{iStep}] = ...
        kneeGivenBodyFoot(yBodyList_inSlopeFrame,zBodyList_inSlopeFrame,yFootSwingList{iStep},zFootSwingList{iStep});

end

tStanceList = zeros(paramFixed.numSteps,1);
stepLengthList = zeros(paramFixed.numSteps,1);
for iStep = 1:paramFixed.numSteps
    tStanceList(iStep) = range(tStore{iStep});
    theta_end = stateStore{iStep}(end,1);
    stepLengthList(iStep) = abs(2*paramFixed.leglength*sin(theta_end));
end

% Odd stance intervals use the fast belt and end with contact on the slow belt.
tStance_fast = tStanceList(1:2:end);
tStance_slow = tStanceList(2:2:end);

stepTimeAsymmetryList = (tStance_slow-tStance_fast)./(tStance_slow+tStance_fast);
% Assign step lengths by the belt contacted at the end of each interval.
stepLength_slow = stepLengthList(1:2:end);
stepLength_fast = stepLengthList(2:2:end);
stepLengthAsymmetryList = (stepLength_fast-stepLength_slow)./(stepLength_fast+stepLength_slow);

params.tList = cumsum(tTotalIterationStore);
params.EmetRateList = EdotStore_IterationAverage;
[tSpan_Smoothed,EmetSList_Smoothed] = convertMetToVO2(params);

skipPlot = 10;

energy_result = params.EmetRateList;
time_list = params.tList;
steptime_fast = tStance_fast;
steptime_slow = tStance_slow;
end
