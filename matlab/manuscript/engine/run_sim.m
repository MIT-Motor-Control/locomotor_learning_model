function [time_list, sla_list, sta_list, energy_list, speed_profile, state_list, steptime_fast, steptime_slow, stepLength_fast, stepLength_slow, pInputControllerStore_OnesTried, fall_flag, min_vert_pos]...
    = run_sim(delta, split_duration, slow_speed, fast_speed, exp_noise, learning_rate, energy_sym_ratio, param_set, SwingCostCoeff, options)
% Simulate split-belt adaptation with fall detection, then replay the evaluated controllers.
% Model, learner and protocol parameters are set here; load*Parameters.m files are not called.
% Inputs:  delta - unused; split_duration - split-belt duration (min)
%          slow_speed, fast_speed - experimental belt speeds (m/s)
%          exp_noise - exploration noise SD; learning_rate - scalar, or [initial, decay]
%          energy_sym_ratio - energy weight lambdaEnergyVsSymmetry (see fObjective_AsymmetricNominal)
%          param_set - initial 8 learned controller targets; SwingCostCoeff - swing-cost multiplier
%          options - seed, max_iterations, transition_seconds, speed_divisor
% Outputs: time_list, sla_list, sta_list, energy_list, steptime_*, stepLength_* - as in
%              postProcessHelper_JustThePlots
%          speed_profile - belt speeds foot1, foot2 at model times 2*k, k = 1 to numStepsToLearn/2
%          state_list, pInputControllerStore_OnesTried - 8 x N controllers evaluated (identical)
%          fall_flag, min_vert_pos - as in simulateLearningStepByStep

if nargin < 10; options = struct(); end
if ~isfield(options, 'seed'); options.seed = 0; end
if ~isfield(options, 'max_iterations'); options.max_iterations = Inf; end
if ~isfield(options, 'transition_seconds'); options.transition_seconds = 25; end
if ~isfield(options, 'speed_divisor'); options.speed_divisor = 2; end
paramFixed = [];
rng(options.seed, 'twister');

paramFixed.includeAccelerationTorque = 1;

paramFixed.mbody = 1;
paramFixed.leglength = 1;
paramFixed.gravg = 1;
% Muscle efficiencies of negative and positive work.
paramFixed.efficiency_neg = 1.2;
paramFixed.efficiency_pos = 0.25;
paramFixed.bPos = 1/paramFixed.efficiency_pos;
paramFixed.bNeg = 1/paramFixed.efficiency_neg;
paramFixed.mFoot = 0.05;
paramFixed.swingCost.Coeff = 0.9;
paramFixed.swingCost.alpha = 1.0;
paramFixed.lambdaEnergyVsPeriodicity = 1;

paramFixed.lambdaEnergyVsSymmetry = energy_sym_ratio;
paramFixed.symmetryMultiplier = 10;

paramFixed.noiseEnergySensory = 0.01;
paramFixed.velocitySensoryNoise = 0;

paramController.pushoff_gain_ydot  = -0.521088310893437;
paramController.legAngle_gain_ydot =  0.279308152405233;
paramController.pushoff_gain_y =  -0.085939152240409;
paramController.legAngle_gain_y =  -0.028999999999996;
paramController.pushoff_gain_SUMy =  -0.007302228127922;
paramController.legAngle_gain_SUMy =  0.002019507100848;
paramController.legAngle_gain_BeltSpeed = -0.075044622818653;
paramController.pushoff_gain_BeltSpeed = -1.055930819923483;

paramFixed.Learner.LearningRate = learning_rate;
paramFixed.Learner.noiseSTDExploratory = exp_noise;
paramFixed.Learner.includeInternalModel = 1;
paramFixed.Learner.numStepsToUseForEstimator = 30;

paramFixed.Learner.LearningRateTowardMemory = 0;
paramFixed.Learner.powerToTheMoveToMemory = 10;
paramFixed.Learner.powerToTheMemoryFormation = 10;
paramFixed.Learner.LearningRateForMemoryFormationUpdates = 0.0;

paramFixed.Learner.numStepsPerIteration = 2;
paramFixed.Learner.trustRegionSize = 0.5*paramFixed.Learner.noiseSTDExploratory;
paramFixed.Learner.shouldWeUseTrustRegion = 1;

paramFixed.Learner.predictionErrorThreshold = 2*paramFixed.noiseEnergySensory;
paramFixed.Learner.shouldWeThresholdPredictionError = 0;

% Model time unit is sqrt(L/g) s; belt speeds are divided by speed_divisor.
L = 1; g = 9.81; timeScaling = sqrt(L/g);
paramFixed.transitionTime = options.transition_seconds;

vFast = -fast_speed/options.speed_divisor;
vSlow = -slow_speed/options.speed_divisor;
vNormal = (vFast + vSlow)/2;
tDurationTransition = paramFixed.transitionTime/timeScaling;

% One minute of tied-belt walking precedes the split; the transition ramp is part of split_duration.
tDuration1 = 60/timeScaling;
footSpeed1_phase1 = vNormal;
footSpeed2_phase1 = vNormal;

tDuration2 = split_duration*60/timeScaling;
footSpeed1_phase2 = vFast;
footSpeed2_phase2 = vSlow;

tStore = [0; tDuration1; tDuration2];
tStore = cumsum(tStore);

footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1;...
    footSpeed1_phase2];
footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1;...
    footSpeed2_phase2];

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

paramFixed.imposedFootSpeeds = beltSpeedsImposed;
paramFixed.angleSlope = 0;

% Iteration budget: protocol duration at a nominal 2 model time units per step.
nominalStepTime = 2;
paramFixed.numStepsToLearn = paramFixed.imposedFootSpeeds.tList(end)/nominalStepTime;

paramFixed.numIterations = min(options.max_iterations, floor(paramFixed.numStepsToLearn/paramFixed.Learner.numStepsPerIteration));
if mod(paramFixed.Learner.numStepsPerIteration,2)~=0
    paramFixed.Learner.numStepsPerIteration = paramFixed.Learner.numStepsPerIteration + 1;
end

stepIndices = beltSpeedsImposed.tList/2;
steps = 1:paramFixed.numStepsToLearn/2;
speed_profile.foot1 = interp1(stepIndices, beltSpeedsImposed.footSpeed1List, steps, 'linear', 'extrap');
speed_profile.foot2 = interp1(stepIndices, beltSpeedsImposed.footSpeed2List, steps, 'linear', 'extrap');

% Nominal step index of the split onset, used only by a two-element learning rate.
changeIndex = 2;
changeTime = beltSpeedsImposed.tList(changeIndex);
speedChangeIdx = round(changeTime / nominalStepTime) - 1;

pInputControllerAsymmetricNominal = [param_set(1:3); 0; 0; param_set(4:end)];

tStart = 0;
vSwingInitial = 0.5;
% Midstance state: leg angle, angular rate, stance-foot position, summed midstance
% position error (integral feedback), swing-foot speed.
stateVar0_Model = [0
    pInputControllerAsymmetricNominal(2)
    0
    0
    vSwingInitial];

stateVar0_Model_BeforeLearning = stateVar0_Model;

[vA,vB] = getTreadmillSpeed(0,paramFixed.imposedFootSpeeds);
contextNow = [vA; vB];
contextLength = length(contextNow);

[pInputControllerStore_OnesTried, fall_flag,min_vert_pos] = ...
    simulateLearningStepByStep(paramFixed,pInputControllerAsymmetricNominal, ...
                                stateVar0_Model,contextNow,paramController,SwingCostCoeff,speedChangeIdx);

% Restore the two zero odd-step position targets for replay.
pInputControllerStore_8D = pInputControllerStore_OnesTried;
pInputControllerStore_10D = [pInputControllerStore_8D(1:3,:);
                            zeros(2,size(pInputControllerStore_8D,2));
                            pInputControllerStore_8D(4:8,:)];

[time_list, sla_list, sta_list, energy_list, steptime_fast, steptime_slow, stepLength_fast, stepLength_slow] = ...
    postProcessAfterLearning(pInputControllerStore_10D, ...
                            stateVar0_Model_BeforeLearning, ...
                            paramController,paramFixed,delta,SwingCostCoeff);

state_list = pInputControllerStore_8D;
end
