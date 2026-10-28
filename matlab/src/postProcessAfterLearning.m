function postProcessAfterLearning(pInputStore,stateVar0_Model, ...
    paramController,paramFixed,doAnimate)
% Replay the walked controllers, assemble gait and energy time series, and plot them.
% Inputs:  pInputStore     - 10-D controllers, one column per learning iteration
%          stateVar0_Model - initial midstance state
%          paramController - feedback gains
%          paramFixed      - model and protocol parameters
%          doAnimate       - unused

numIterations = size(pInputStore,2);

tStore = [];
stateStore = [];
EmetTotalStore = [];
EmetPerTimeStore = [];
tStepStore = [];
EworkPushoffStore = [];
EworkHeelstrikeStore = [];

tStart = 0;

disp('Post-processing all the walking data ...');

for iStride = 1:numIterations

    if mod(iStride,50)==0
        disp(['iStride = ' num2str(iStride)]);
    end

    pInputControllerAsymmetricNominal = pInputStore(:,iStride);

    paramController.Odd.theta_end_nominal = ...
        pInputControllerAsymmetricNominal(1);
    paramController.Odd.ydot_atMidstance_nominal_beltframe = ...
        pInputControllerAsymmetricNominal(2);
    paramController.Odd.PushoffImpulseMagnitude_nominal = ...
        pInputControllerAsymmetricNominal(3);
    paramController.Odd.y_atMidstance_nominal_slopeframe = ...
        pInputControllerAsymmetricNominal(4);
    paramController.Odd.SUMy_atMidstance_nominal_slopeframe = ...
        pInputControllerAsymmetricNominal(5);

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

    paramFixed.numSteps = paramFixed.Learner.numStepsPerIteration;
    [stateVar0_Model,tStore_Now,stateStore_Now,EmetTotalStore_Now, EmetPerTimeStore_Now, ...
        tStepStore_Now,EworkPushoffStore_Now,EworkHeelstrikeStore_Now] = ...
        simulateManySteps_AsymmetricControl(stateVar0_Model,paramController,paramFixed,tStart);

    tStart = tStore_Now{end}(end);

    EdotStore_IterationAverage(iStride) = sum(EmetTotalStore_Now)/sum(tStepStore_Now);
    tTotalIterationStore(iStride) = sum(tStepStore_Now);

    if iStride ==1
        tStore = [tStore; tStore_Now];
        stateStore = [stateStore; stateStore_Now];
        EmetTotalStore = [EmetTotalStore; EmetTotalStore_Now];
        EmetPerTimeStore = [EmetPerTimeStore; EmetPerTimeStore_Now];
        tStepStore = [tStepStore; tStepStore_Now];

        EworkPushoffStore = [EworkPushoffStore; EworkPushoffStore_Now];
        EworkHeelstrikeStore = [EworkHeelstrikeStore; EworkHeelstrikeStore_Now];

    else

        % Merge the trailing half-step with the next midstance-to-endstance segment.
        tStore{end} = [tStore{end}; tStore_Now{1}];
        stateStore{end} = [stateStore{end}; stateStore_Now{1}];
        tStore = [tStore; tStore_Now(2:end)];
        stateStore = [stateStore; stateStore_Now(2:end)];

        EmetTotalStore = [EmetTotalStore; EmetTotalStore_Now];
        EmetPerTimeStore = [EmetPerTimeStore; EmetPerTimeStore_Now];
        tStepStore = [tStepStore; tStepStore_Now];

        EworkPushoffStore = [EworkPushoffStore; EworkPushoffStore_Now];
        EworkHeelstrikeStore = [EworkHeelstrikeStore; EworkHeelstrikeStore_Now];

    end

end

postProcessHelper_JustThePlots(stateVar0_Model,tStore,stateStore, ...
    EmetTotalStore,EmetPerTimeStore,tStepStore,paramController,paramFixed,doAnimate,EworkPushoffStore,EworkHeelstrikeStore, ...
    EdotStore_IterationAverage, tTotalIterationStore)

end
