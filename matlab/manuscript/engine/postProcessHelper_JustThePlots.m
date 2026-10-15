function [time_list, stepLengthAsymmetryList, stepTimeAsymmetryList, energy_result, steptime_fast, steptime_slow, stepLength_fast, stepLength_slow] = postProcessHelper_JustThePlots(stateVar0,tStore,stateStore, ...
    EmetStore,EmetPerTimeStore,tStepStore,paramController,paramFixed,delta, ...
    EworkPushoffStore,EworkHeelstrikeStore, EdotStore_IterationAverage, ...
    tTotalIterationStore)

% this program makes the plots/figures

%% do post-processing and plotting of things
% tend = 0;
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

%% getting things in the lab frame
for iStep = 1:paramFixed.numSteps
    yFootStanceList{iStep} = stateStore{iStep}(:,3);
end

%% getting a interpolated swing foot trajectory, from stance foot to stance foot
% just for visualization purposes

for iStep = 1:paramFixed.numSteps

    if (iStep<=1)||(iStep>=paramFixed.numSteps)

        % no swing foot info for the first step and thee last step
        % or we can just choose to animate steps from 2 to end-1
        yFootSwingList{iStep} = NaN*ones(size(tStore{iStep}));
        zFootSwingList{iStep} = NaN*ones(size(tStore{iStep}));

    else

        % end of last stance is beginning of current swing
        % begin of next stance is end of current swing
        swingFootBegin = yFootStanceList{iStep-1}(end);
        swingFootEnd = yFootStanceList{iStep+1}(1);

        % interpolating swing foot, constant velocity over the whole distance
        yFootSwingList{iStep} = interp1([tStore{iStep}(1) tStore{iStep}(end)],[swingFootBegin swingFootEnd],tStore{iStep});

        % interpolating swing foot, beginning and ending at rest
        %         % achieved using a shifted cosine
        %         Atemp = (swingFootBegin-swingFootEnd)/2; % amplitude of the necessary cosine
        %         Amean = (swingFootEnd+swingFootBegin)/2; % mean of necessary cosine
        %         Tswing = (tStore{iStep}(end)-tStore{iStep}(1));
        %         yFootSwingList{iStep} = Amean+Atemp*cos(2*pi*(tStore{iStep}-tStore{iStep}(1))/(2*Tswing));
        % okay the above interpolation is NOT good, because we want the foot to
        % have belt speed at beginning and end, not zero speed!! perhaps a
        % polynomial fit is best

        % vertical foot excursion
        footHeight = 0.03;
        Amean = (0+footHeight)/2;
        Atemp = (0-footHeight)/2;
        Tswing = (tStore{iStep}(end)-tStore{iStep}(1));
        zFootSwingList{iStep} = Amean+Atemp*cos(2*pi*(tStore{iStep}-tStore{iStep}(1))/(Tswing));

    end

    thetaList = stateStore{iStep}(:,1);
    yBodyList_inFootFrame = paramFixed.leglength*sin(thetaList+paramFixed.angleSlope);
    zBodyList_inSlopeFrame = paramFixed.leglength*cos(thetaList); % z body

    yBodyList_inSlopeFrame = yBodyList_inFootFrame+yFootStanceList{iStep}; % y body in foot frame

    % fake knee position
    [ySwingKneeList{iStep},zSwingKneeList{iStep}] = ...
        kneeGivenBodyFoot(yBodyList_inSlopeFrame,zBodyList_inSlopeFrame,yFootSwingList{iStep},zFootSwingList{iStep});

end

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

% color_pool = {
%     [240, 128, 128] / 255;   % Light Coral
%     [205, 92, 92] / 255;     % Indian Red
%     [178, 34, 34] / 255;     % Firebrick
%     [128, 0, 0] / 255;       % Maroon
%     [139, 0, 0] / 255        % Dark Red
% };
% color_pool_blue = {
%     [173, 216, 230] / 255;   % Light Blue
%     [135, 206, 235] / 255;   % Sky Blue
%     [30, 144, 255] / 255;    % Dodger Blue
%     [0, 0, 255] / 255;       % Blue
%     [0, 0, 139] / 255        % Dark Blue
% };
% if strcmp(paramFixed.SplitOrTied,'split')
% 
%     stepTimeAsymmetryList(1) = NaN;
%     strideCountList = 1:length(stepTimeAsymmetryList);
% 
%     figure(1);
%     subplot(132); plot(strideCountList(2:skipPlot:end),stepLengthAsymmetryList(2:skipPlot:end),'-','linewidth',2, 'Color', color_pool{delta}); hold on;
%     xlabel('stride index'); ylabel('step length symmetry');
%     ylim([-0.5 0.5]); axis square;
%     xlim([0 max(strideCountList)]);
% 
%     figure(1); hold on;
%     subplot(133); strideList = (1:round(length(params.EmetRateList)))*paramFixed.Learner.numStepsPerIteration/2;
%     % plot(strideList(1:skipPlot:end),params.EmetRateList(1:skipPlot:end),'-'); hold on;
%     plot(strideList(1:skipPlot:end),EmetSList_Smoothed(1:skipPlot:end),'linewidth',2, 'Color', color_pool{delta});
%     xlabel('stride index'); ylabel('Edot, met rate')
%     legend([num2str(paramFixed.Learner.numStepsPerIteration) ' step average'],'Edot smoothed by VO2');
%     ylim([0 max(params.EmetRateList)]); axis square;
% 
%     figure(1); hold on;
%     subplot(131); [foot1SpeedList,foot2SpeedList] = getTreadmillSpeed(params.tList,paramFixed.imposedFootSpeeds);
%     plot(params.tList(1:skipPlot:end),abs(foot1SpeedList(1:skipPlot:end)),'linewidth',2, 'Color', color_pool_blue{delta}); hold on;
%     plot(params.tList(1:skipPlot:end),abs(foot2SpeedList(1:skipPlot:end)),'linewidth',2, 'Color', color_pool{delta}); ylim([0.3 0.5]); axis square;
%     xlim([0 (max(params.tList))]);
%     % set(gcf,'WindowState','fullscreen');
%     set(gca, 'xticklabel', []);
%     xlabel('time');
%     ylabel('treadmill belt speeds');
%     legend('fast belt','slow belt');
%     ylim([0 0.6]);
% 
% end

energy_result = params.EmetRateList;
time_list = params.tList;
steptime_fast = tStance_fast;
steptime_slow = tStance_slow;
end % essential % we've made the doAnimate thing not generate the animation and
% removed the diagnostic plots. see older version for additional details.
