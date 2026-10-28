function [stateVar0,tStore,stateStore,EmetTotalStore, ...
    EmetPerTimeStore,tStepStore,EworkPushoffStore,EworkHeelstrikeStore,fall_info,yPosStore] = ...
    simulateManySteps_AsymmetricControl(stateVar0,paramController,paramFixed,tStart,SwingCostCoeff,iStride)
% Simulate paramFixed.numSteps steps, alternating the odd- and even-step controller targets.
% Inputs:  stateVar0 - midstance state (ordering in run_sim)
%          paramController - feedback gains with Odd and Even targets
%          paramFixed - model settings; tStart - start time
%          SwingCostCoeff - swing-cost multiplier; iStride - learning iteration
% Outputs: stateVar0 - final midstance state
%          tStore, stateStore - stance trajectories, numSteps+1 cells
%          EmetTotalStore, EmetPerTimeStore - metabolic energy and power per step
%          tStepStore - step durations
%          EworkPushoffStore, EworkHeelstrikeStore - push-off and heel-strike work per step
%          fall_info - fall flag of the last step; yPosStore - minimum height per step

tStore = cell(paramFixed.numSteps+1,1);
stateStore = cell(paramFixed.numSteps+1,1);
EmetTotalStore = zeros(paramFixed.numSteps,1);
EmetPerTimeStore = zeros(paramFixed.numSteps,1);
tStepStore = zeros(paramFixed.numSteps,1);

EworkPushoffStore = zeros(paramFixed.numSteps,1);
EworkHeelstrikeStore = zeros(paramFixed.numSteps,1);
yPosStore = zeros(paramFixed.numSteps,1);

for iStep = 1:paramFixed.numSteps

    if mod(iStep,2)==0
        paramController.theta_end_nominal = ...
            paramController.Even.theta_end_nominal;
        paramController.ydot_atMidstance_nominal_beltframe = ...
            paramController.Even.ydot_atMidstance_nominal_beltframe;
        paramController.PushoffImpulseMagnitude_nominal = ...
            paramController.Even.PushoffImpulseMagnitude_nominal;
        paramController.y_atMidstance_nominal_slopeframe = ...
            paramController.Even.y_atMidstance_nominal_slopeframe;
        paramController.SUMy_atMidstance_nominal_slopeframe = ...
            paramController.Even.SUMy_atMidstance_nominal_slopeframe;
    else
        % Odd-step position targets are held at zero.
        paramController.theta_end_nominal = ...
            paramController.Odd.theta_end_nominal;
        paramController.ydot_atMidstance_nominal_beltframe = ...
            paramController.Odd.ydot_atMidstance_nominal_beltframe;
        paramController.PushoffImpulseMagnitude_nominal = ...
            paramController.Odd.PushoffImpulseMagnitude_nominal;
        paramController.y_atMidstance_nominal_slopeframe = ...
            0;
        paramController.SUMy_atMidstance_nominal_slopeframe = ...
            0;
    end

    [stateVar0, tlistTillEndstance, statevarlistTillEndstance, ...
       tlistTillMidstance, statevarlistMidstance, ...
       Emet_totalNow,Emet_perTime,tTotal,EworkPushoff,EworkHeelstrike,fall_info, y_pos] = ...
        simulateOneStep_MidstanceToMidstance_WithEnergy(stateVar0,iStep,tStart, ...
        paramController,paramFixed,SwingCostCoeff,iStride);

    % Join the two half-stances of each leg at midstance, so each cell holds one stance phase.
    tStore{iStep} = [tStore{iStep}; tlistTillEndstance];
    stateStore{iStep} = [stateStore{iStep}; statevarlistTillEndstance];

    if iStep<paramFixed.numSteps
        tStore{iStep+1} = [tStore{iStep+1}; tlistTillMidstance(1:end-1)];
        stateStore{iStep+1} = [stateStore{iStep+1}; statevarlistMidstance(1:end-1,:)];
    else
        tStore{iStep+1} = [tStore{iStep+1}; tlistTillMidstance(1:end)];
        stateStore{iStep+1} = [stateStore{iStep+1}; statevarlistMidstance(1:end,:)];
    end

    tStart = tlistTillMidstance(end);

    EmetTotalStore(iStep) = Emet_totalNow;
    EmetPerTimeStore(iStep) = Emet_perTime;
    tStepStore(iStep) = tTotal;

    EworkPushoffStore(iStep) = EworkPushoff;
    EworkHeelstrikeStore(iStep) = EworkHeelstrike;
    yPosStore(iStep) = y_pos;

end

end
