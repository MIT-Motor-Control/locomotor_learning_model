function postProcessHelper_JustThePlots(stateVar0,tStore,stateStore, ...
    EmetStore,EmetPerTimeStore,tStepStore,paramController,paramFixed,doAnimate, ...
    EworkPushoffStore,EworkHeelstrikeStore, EdotStore_IterationAverage, ...
    tTotalIterationStore)
% this program makes the plots/figures

% Animation is disabled in this baseline. Only compute values used in the
% scientific summary plots; discarded swing-foot/knee geometry is omitted.
paramFixed.numSteps = length(EmetStore);

%% step lengths, step length asymmetry, step time, step time asymmetry
tStanceList = zeros(paramFixed.numSteps,1);
stepLengthList = zeros(paramFixed.numSteps,1);
for iStep = 1:paramFixed.numSteps
    tStanceList(iStep) = range(tStore{iStep});
    %     theta_initial = stateStore{iStep}(1,1);
    theta_end = stateStore{iStep}(end,1); % last row, first column
    stepLengthList(iStep) = abs(2*paramFixed.leglength*sin(theta_end)); % now based on END theta of a step
end

% Note: the following assumes that the first belt is faster
tStance_fast = tStanceList(1:2:end); % stepping on to slow belt at the end: BEING on fast belt throughout
tStance_slow = tStanceList(2:2:end); % stepping on to fast belt at the end: BEING on slow belt throughout

stepTimeAsymmetryList = (tStance_slow-tStance_fast)./(tStance_slow+tStance_fast);
stepLength_slow = stepLengthList(1:2:end); % stepping on to slow belt at the end of odd steps: odd stances are on slow
stepLength_fast = stepLengthList(2:2:end); % stepping on to fast belt at the end of even steps: even stances are on slow
stepLengthAsymmetryList = (stepLength_fast-stepLength_slow)./(stepLength_fast+stepLength_slow);

%% put in a metabolic VO2 transient with a 40 sec time constant
params.tList = cumsum(tTotalIterationStore);
params.EmetRateList = EdotStore_IterationAverage;
[tSpan_Smoothed,EmetSList_Smoothed] = convertMetToVO2(params);

skipPlot = 10;

if strcmp(paramFixed.SplitOrTied,'split')
    
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
    % set(gcf,'WindowState','fullscreen');
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
    % set(gcf,'WindowState','fullscreen');
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


end % essential % we've made the doAnimate thing not generate the animation and
% removed the diagnostic plots. see older version for additional details.
