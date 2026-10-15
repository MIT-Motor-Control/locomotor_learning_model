function [time_list, sla_list, sta_list, energy_list, steptime_fast, steptime_slow, stepLength_fast, stepLength_slow] = postProcessAfterLearning(pInputStore,stateVar0_Model, ...
    paramController,paramFixed,delta,SwingCostCoeff)

numIterations = size(pInputStore,2);
% fObjective_AsymmetricNominal(,stateVar0_Model, ...
%     paramController,paramFixed)

%%
tStore = [];
stateStore = [];
EmetTotalStore = [];
EmetPerTimeStore = [];
tStepStore = [];
EworkPushoffStore = [];
EworkHeelstrikeStore = [];

tStart = 0; % just the first time

disp('Post-processing all the walking data ...');

%%
for iStride = 1:numIterations
    
    if mod(iStride, 200)==0
        disp(['iStride = ' num2str(iStride)]);
    end
    
    %% get the stored current learned controller parameters
    pInputControllerAsymmetricNominal = pInputStore(:,iStride);
    
    %%
    paramController.Odd.theta_end_nominal = ...
        pInputControllerAsymmetricNominal(1);
    paramController.Odd.ydot_atMidstance_nominal_beltframe = ...
        pInputControllerAsymmetricNominal(2);
    paramController.Odd.PushoffImpulseMagnitude_nominal = ...
        pInputControllerAsymmetricNominal(3);
    paramController.Odd.y_atMidstance_nominal_slopeframe = ...
        pInputControllerAsymmetricNominal(4); % needs to be fixed to zero, is it?
    paramController.Odd.SUMy_atMidstance_nominal_slopeframe = ...
        pInputControllerAsymmetricNominal(5); % needs to be fixed to zero, is it?
    
    paramController.Even.theta_end_nominal = ...
        pInputControllerAsymmetricNominal(6);
    paramController.Even.ydot_atMidstance_nominal_beltframe = ...
        pInputControllerAsymmetricNominal(7);
    paramController.Even.PushoffImpulseMagnitude_nominal = ...
        pInputControllerAsymmetricNominal(8);
    paramController.Even.y_atMidstance_nominal_slopeframe = ...
        pInputControllerAsymmetricNominal(9);
    paramController.Even.SUMy_atMidstance_nominal_slopeframe = ...
        pInputControllerAsymmetricNominal(10);
    
    
    %%
    paramFixed.numSteps = paramFixed.Learner.numStepsPerIteration;
    [stateVar0_Model,tStore_Now,stateStore_Now,EmetTotalStore_Now, EmetPerTimeStore_Now, ...
        tStepStore_Now,EworkPushoffStore_Now,EworkHeelstrikeStore_Now,fall_info] = ...
        simulateManySteps_AsymmetricControl(stateVar0_Model,paramController,paramFixed,tStart,SwingCostCoeff,iStride);
    
    %% reset things
    tStart = tStore_Now{end}(end);
    
    %%
    
    EdotStore_IterationAverage(iStride) = sum(EmetTotalStore_Now)/sum(tStepStore_Now);
    tTotalIterationStore(iStride) = sum(tStepStore_Now);
    
    %% assemble all the Store variables
    if iStride ==1
        tStore = [tStore; tStore_Now];
        stateStore = [stateStore; stateStore_Now];
        EmetTotalStore = [EmetTotalStore; EmetTotalStore_Now];
        EmetPerTimeStore = [EmetPerTimeStore; EmetPerTimeStore_Now];
        tStepStore = [tStepStore; tStepStore_Now]; % this is step time midstance to midstance, not useful
        
        EworkPushoffStore = [EworkPushoffStore; EworkPushoffStore_Now];
        EworkHeelstrikeStore = [EworkHeelstrikeStore; EworkHeelstrikeStore_Now];

    else
        
        % to merge the half steps at the end and the beginning to compile
        % things for steps defined as heel strike to heel strike
        tStore{end} = [tStore{end}; tStore_Now{1}];
        stateStore{end} = [stateStore{end}; stateStore_Now{1}];
        % the other steps
        tStore = [tStore; tStore_Now(2:end)];
        stateStore = [stateStore; stateStore_Now(2:end)];
        
        % the rest are just arrays from midstance to midstance
        EmetTotalStore = [EmetTotalStore; EmetTotalStore_Now];
        EmetPerTimeStore = [EmetPerTimeStore; EmetPerTimeStore_Now];
        tStepStore = [tStepStore; tStepStore_Now];
        
        EworkPushoffStore = [EworkPushoffStore; EworkPushoffStore_Now];
        EworkHeelstrikeStore = [EworkHeelstrikeStore; EworkHeelstrikeStore_Now];
        
    end
    
end

%% save if you wish to not run the whole simulation 
% save(['EverythingNeededForJustThePlots_' paramFixed.speedProtocol num2str(round(rand*10000)) '.mat']);

%%
[time_list, sla_list, sta_list, energy_list, steptime_fast, steptime_slow, stepLength_fast, stepLength_slow] = postProcessHelper_JustThePlots(stateVar0_Model,tStore,stateStore, ...
    EmetTotalStore,EmetPerTimeStore,tStepStore,paramController,paramFixed,delta,EworkPushoffStore,EworkHeelstrikeStore, ...
    EdotStore_IterationAverage, tTotalIterationStore);

% step_time = tTotalIterationStore;
end