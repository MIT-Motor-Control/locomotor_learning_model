function [time_list, sla_list, sta_list, energy_list, speed_profile, state_list, steptime_fast, steptime_slow, stepLength_fast, stepLength_slow, pInputControllerStore_OnesTried, fall_flag, min_vert_pos]...
    = run_sim(delta, split_duration, slow_speed, fast_speed, exp_noise, learning_rate, energy_sym_ratio, param_set, SwingCostCoeff, options)

%% initialize empty parameter values
if nargin < 10; options = struct(); end
if ~isfield(options, 'seed'); options.seed = 0; end
if ~isfield(options, 'max_iterations'); options.max_iterations = Inf; end
if ~isfield(options, 'transition_seconds'); options.transition_seconds = 25; end
if ~isfield(options, 'speed_divisor'); options.speed_divisor = 2; end
paramFixed = [];
rng(options.seed, 'twister'); 

%% biped model parameters
% parameters related to the biped dynamics, like biped segment masses and
% lengths, gravity, and performance cost parameters (energy or symmetry)
paramFixed.includeAccelerationTorque = 1; % treadmill acceleration include

% Body parameter, efficiency, swing leg cost
paramFixed.mbody = 1;
paramFixed.leglength = 1;
paramFixed.gravg = 1;
paramFixed.efficiency_neg = 1.2;
paramFixed.efficiency_pos = 0.25;
paramFixed.bPos = 1/paramFixed.efficiency_pos; 
paramFixed.bNeg = 1/paramFixed.efficiency_neg;
paramFixed.mFoot = 0.05;
paramFixed.swingCost.Coeff = 0.9; % for best 
paramFixed.swingCost.alpha = 1.0;
paramFixed.lambdaEnergyVsPeriodicity = 1; % 1 = energy, 0 = periodicity

% energy vs symmetry
paramFixed.lambdaEnergyVsSymmetry = energy_sym_ratio; % 1 = energy, 0 = symmetry, it's the ratio (0 to 1)
paramFixed.symmetryMultiplier = 10; % just multiplies the symmetry objective, 
% so that its gradients are comparable to the energy cost gradients

%% sensory noise parameters
% sensory noise just makes learning challenging. Setting these to zero
% just lowers the noise in the behavior without affecting the qualitative
% trends, but increasing these eventually kills learning or makes the biped
% unstable and fall

% noise for energy measurements
paramFixed.noiseEnergySensory = 0.01; % this is multiplicative noise (0 ~ 0.0001, factor of 10)
paramFixed.velocitySensoryNoise = 0; % this is additive noise

%% controller parameters
% These are the feedback gains in the feedback controller that keeps the
% biped stable. They are obtained from how 
% Controller parameters, gains, Fixed during the optimization

paramController.pushoff_gain_ydot  = -0.521088310893437;
paramController.legAngle_gain_ydot =  0.279308152405233;
paramController.pushoff_gain_y =  -0.085939152240409;
paramController.legAngle_gain_y =  -0.028999999999996;
paramController.pushoff_gain_SUMy =  -0.007302228127922;
paramController.legAngle_gain_SUMy =  0.002019507100848;
paramController.legAngle_gain_BeltSpeed = -0.075044622818653;
paramController.pushoff_gain_BeltSpeed = -1.055930819923483;

%% learner parameters
paramFixed.Learner.LearningRate = learning_rate; % default learning rate (0 ~ 0.004)
paramFixed.Learner.noiseSTDExploratory = exp_noise; % higher for tied walking 0.002
paramFixed.Learner.includeInternalModel = 1; % takes values zero or one
paramFixed.Learner.numStepsToUseForEstimator = 30;

paramFixed.Learner.LearningRateTowardMemory = 0; % default: some progress toward stored memory (0, 0.01, 0.02, 0.005, 1)
paramFixed.Learner.powerToTheMoveToMemory = 10; % truncated cosine tuning
paramFixed.Learner.powerToTheMemoryFormation = 10;
paramFixed.Learner.LearningRateForMemoryFormationUpdates = 0.0; % no new memories (or 0.03)

paramFixed.Learner.numStepsPerIteration = 2; % how frequently do we update the controller parameter
paramFixed.Learner.trustRegionSize = 0.5*paramFixed.Learner.noiseSTDExploratory; % trust region
paramFixed.Learner.shouldWeUseTrustRegion = 1;

paramFixed.Learner.predictionErrorThreshold = 2*paramFixed.noiseEnergySensory; % prediction error threshold
paramFixed.Learner.shouldWeThresholdPredictionError = 0; % set 0 if not interested in this thresholding

%% adaptation protocol 
% adaptation protocol parameters: how the treadmill speed is changed, and
% whether the adaptation protocol is on a split-belt treadmill or a tied
% belt treadmill.

% make treadmill speed based on the setting
L = 1; g = 9.81; timeScaling = sqrt(L/g);
paramFixed.transitionTime = options.transition_seconds;

vFast = -fast_speed/options.speed_divisor; % need to scale down
vSlow = -slow_speed/options.speed_divisor;
vNormal = (vFast + vSlow)/2;
tDurationTransition = paramFixed.transitionTime/timeScaling;

tDuration1 = 60/timeScaling; % phase 1: tied-belt walking (30 sec just to get up and running)
footSpeed1_phase1 = vNormal;
footSpeed2_phase1 = vNormal;

tDuration2 = split_duration*60/timeScaling; % phase 2: split-belt adaptation
footSpeed1_phase2 = vFast;
footSpeed2_phase2 = vSlow;

tStore = [0; tDuration1; tDuration2];
tStore = cumsum(tStore);

footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1;...
    footSpeed1_phase2];
footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1;...
    footSpeed2_phase2];

% adding transition phases
tStore_new = 0;
footSpeed1Store_new = footSpeed1Store(1);
footSpeed2Store_new = footSpeed2Store(1);
for iTran = 2:length(tStore)
    if iTran<length(tStore)
        tStore_new = [tStore_new; tStore(iTran); tStore(iTran)+tDurationTransition];
        footSpeed1Store_new = [footSpeed1Store_new; footSpeed1Store(iTran); footSpeed1Store(iTran+1)];
        footSpeed2Store_new = [footSpeed2Store_new; footSpeed2Store(iTran); footSpeed2Store(iTran+1)];
    else
        tStore_new = [tStore_new; tStore(iTran)];
        footSpeed1Store_new = [footSpeed1Store_new; footSpeed1Store(iTran)];
        footSpeed2Store_new = [footSpeed2Store_new; footSpeed2Store(iTran)];
    end
end

% store in a structure
beltSpeedsImposed.tList = tStore_new;
beltSpeedsImposed.footSpeed1List = footSpeed1Store_new;
beltSpeedsImposed.footSpeed2List = footSpeed2Store_new;
a1List = ...
    diff(beltSpeedsImposed.footSpeed1List)./diff(beltSpeedsImposed.tList);
a2List = ...
    diff(beltSpeedsImposed.footSpeed2List)./diff(beltSpeedsImposed.tList);
a1List = [a1List; 0]; a2List = [a2List; 0];
beltSpeedsImposed.footAcc1List = a1List;
beltSpeedsImposed.footAcc2List = a2List;

% save the full imposed speed into param
paramFixed.imposedFootSpeeds = beltSpeedsImposed;
paramFixed.angleSlope = 0; % do not change: the code has not been tested for non-zero values

% How many steps to simulate
nominalStepTime = 2; % need to scale
paramFixed.numStepsToLearn = paramFixed.imposedFootSpeeds.tList(end)/nominalStepTime; % exact number of steps
% paramFixed.numStepsToLearn = round(paramFixed.numStepsToLearn/100)*100; % round to the nearest hundred

% optimization iterations
paramFixed.numIterations = min(options.max_iterations, floor(paramFixed.numStepsToLearn/paramFixed.Learner.numStepsPerIteration));
if mod(paramFixed.Learner.numStepsPerIteration,2)~=0 
    paramFixed.Learner.numStepsPerIteration = paramFixed.numStepsPerIteration + 1;
end

% convert time values to step indices
% stepIndices = beltSpeedsImposed.tList / nominalStepTime;
stepIndices = beltSpeedsImposed.tList/2;
steps = 1:paramFixed.numStepsToLearn/2; % number of stride
speed_profile.foot1 = interp1(stepIndices, beltSpeedsImposed.footSpeed1List, steps, 'linear', 'extrap');
speed_profile.foot2 = interp1(stepIndices, beltSpeedsImposed.footSpeed2List, steps, 'linear', 'extrap');

% find the index where split-belt starts
changeIndex = 2; % a1List's second index
changeTime = beltSpeedsImposed.tList(changeIndex);
speedChangeIdx = round(changeTime / nominalStepTime) - 1;

%% load current learnable parameters
pInputControllerAsymmetricNominal = [param_set(1:3); 0; 0; param_set(4:end)];

%% load initial state and time conditons for the simluation
% NOTE: these are MID-STANCE initial conditions
tStart = 0;
vSwingInitial = 0.5; % initial swing speed - shouldn't matter much, ideally equal to the nominal velocity?
stateVar0_Model = [0                        % angleTheta0 = stance leg angle
    pInputControllerAsymmetricNominal(2)    % dAngleTheta0 = stance leg angular rate
    0                                       % yFoot0 = 0; % in lab frame
    0                                       % sum of yFoot in lab frame (integral feedback for station keeping)
    vSwingInitial];                         % vSwing 

stateVar0_Model_BeforeLearning = stateVar0_Model; % storing initial state for later use

%% context for the gait
[vA,vB] = getTreadmillSpeed(0,paramFixed.imposedFootSpeeds);
contextNow = [vA; vB];
contextLength = length(contextNow);

%% Simulate learning step by step
[pInputControllerStore_OnesTried, fall_flag,min_vert_pos] = ...
    simulateLearningStepByStep(paramFixed,pInputControllerAsymmetricNominal, ...
                                stateVar0_Model,contextNow,paramController,SwingCostCoeff,speedChangeIdx);

%% convert the 8D back up to 10D to use the old functions
pInputControllerStore_8D = pInputControllerStore_OnesTried;
pInputControllerStore_10D = [pInputControllerStore_8D(1:3,:);
                            zeros(2,size(pInputControllerStore_8D,2));
                            pInputControllerStore_8D(4:8,:)];

%% post-process outputs and make some plots
[time_list, sla_list, sta_list, energy_list, steptime_fast, steptime_slow, stepLength_fast, stepLength_slow] = ...
    postProcessAfterLearning(pInputControllerStore_10D, ...
                            stateVar0_Model_BeforeLearning, ...
                            paramController,paramFixed,delta,SwingCostCoeff);

state_list = pInputControllerStore_8D;
end