function postProcessHelper_JustThePlots(stateVar0,tStore,stateStore, ...
    EmetStore,EmetPerTimeStore,tStepStore,paramController,paramFixed,doAnimate, ...
    EworkPushoffStore,EworkHeelstrikeStore, EdotStore_IterationAverage, ...
    tTotalIterationStore)
% Compute step asymmetries and the VO2-filtered energy rate, and plot the results.
% Split-belt runs show step length asymmetry and energy rate, tied-belt runs show
% step frequency; both show the belt speeds.
% Inputs:  tStore, stateStore - per-stance time and state trajectories
%          EmetStore          - per-step energy; only its length is used
%          paramFixed         - model and protocol parameters
%          EdotStore_IterationAverage - energy rate of each learning iteration
%          tTotalIterationStore       - duration of each learning iteration
%          stateVar0, EmetPerTimeStore, tStepStore, paramController, doAnimate,
%          EworkPushoffStore, EworkHeelstrikeStore - unused

paramFixed.numSteps = length(EmetStore);

tStanceList = zeros(paramFixed.numSteps,1);
stepLengthList = zeros(paramFixed.numSteps,1);
for iStep = 1:paramFixed.numSteps
    tStanceList(iStep) = range(tStore{iStep});
    theta_end = stateStore{iStep}(end,1);
    stepLengthList(iStep) = abs(2*paramFixed.leglength*sin(theta_end));
end

% Odd stance intervals are on belt 1 (fast) and end with heel strike on belt 2
% (slow), so odd intervals give slow step lengths. Step length asymmetry
% follows paper Eq. 3.
tStance_fast = tStanceList(1:2:end);
tStance_slow = tStanceList(2:2:end);

stepTimeAsymmetryList = (tStance_slow-tStance_fast)./(tStance_slow+tStance_fast);
stepLength_slow = stepLengthList(1:2:end);
stepLength_fast = stepLengthList(2:2:end);
stepLengthAsymmetryList = (stepLength_fast-stepLength_slow)./(stepLength_fast+stepLength_slow);

params.tList = cumsum(tTotalIterationStore);
params.EmetRateList = EdotStore_IterationAverage;
[tSpan_Smoothed,EmetSList_Smoothed] = convertMetToVO2(params);

skipPlot = 10;

if strcmp(paramFixed.SplitOrTied,'split')

    % The first interval starts at midstance, so its stance time is incomplete.
    stepTimeAsymmetryList(1) = NaN;
    strideCountList = 1:length(stepTimeAsymmetryList);

    figure(200);
    subplot(132); plot(strideCountList(2:skipPlot:end),stepLengthAsymmetryList(2:skipPlot:end),'-','linewidth',2); hold on;
    xlabel('stride index'); ylabel('step length symmetry');
    ylim([-0.5 0.5]); axis square;
    xlim([0 max(strideCountList)]);

    figure(200); hold on;
    subplot(133); strideList = (1:round(length(params.EmetRateList)))*paramFixed.Learner.numStepsPerIteration/2;
    plot(strideList(1:skipPlot:end),params.EmetRateList(1:skipPlot:end),'-'); hold on;
    plot(strideList(1:skipPlot:end),EmetSList_Smoothed(1:skipPlot:end),'linewidth',2);
    xlabel('stride index'); ylabel('Edot, met rate')
    legend([num2str(paramFixed.Learner.numStepsPerIteration) ' step average'],'Edot smoothed by VO2');
    ylim([0 max(params.EmetRateList)]); axis square;

    figure(200); hold on;
    subplot(131); [foot1SpeedList,foot2SpeedList] = getTreadmillSpeed(params.tList,paramFixed.imposedFootSpeeds);
    plot(params.tList(1:skipPlot:end),abs(foot1SpeedList(1:skipPlot:end)),'linewidth',2); hold on;
    plot(params.tList(1:skipPlot:end),abs(foot2SpeedList(1:skipPlot:end)),'linewidth',2); ylim([0.3 0.5]); axis square;
    xlim([0 (max(params.tList))]);
    set(gca, 'xticklabel', []);
    xlabel('time');
    ylabel('treadmill belt speeds');
    legend('fast belt','slow belt');
    ylim([0 0.6]);

end

if strcmp(paramFixed.SplitOrTied,'tied')
    figure(201); subplot(121); hold on;
    [foot1SpeedList,foot2SpeedList] = getTreadmillSpeed(params.tList,paramFixed.imposedFootSpeeds);
    plot(params.tList(1:skipPlot:end),abs(foot1SpeedList(1:skipPlot:end)),'linewidth',2); hold on;
    plot(params.tList(1:skipPlot:end),abs(foot2SpeedList(1:skipPlot:end)),'linewidth',2); ylim([0.3 0.5]); axis square;
    xlim([0 (max(params.tList))]);
    set(gca, 'xticklabel', []);
    xlabel('time');
    ylabel('treadmill belt speeds');
    legend('fast belt','slow belt');
    ylim([0 0.6]);
    axis square;

    figure(201); subplot(122); hold on;
    tStancePerStride = (tStance_fast+tStance_slow)/2;
    tList_stepBegin = cumsum(tStanceList);
    plot(tList_stepBegin(1:2:end),1./tStancePerStride);
    xlabel('time (non dim)'); ylabel('step freq, averaged over 2 steps');
    ylim([0 0.8]); axis square;

    sgtitle('Tied Treadmill: Step frequency changes in response to speed changes');
end

end
