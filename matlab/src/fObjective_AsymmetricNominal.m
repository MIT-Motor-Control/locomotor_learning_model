function [f_objective,stateVar0,tEnd,f_energy,f_constraint] = ...
    fObjective_AsymmetricNominal(pInputControllerNominal,stateVar0_Model, ...
    paramController,paramFixed,tStart)
% Walk one learning iteration and return the energy, periodicity and symmetry objective.
% Inputs:  pInputControllerNominal - 10-D controller (layout in loadLearnableParametersInitial)
%          stateVar0_Model         - initial midstance state
%          paramController         - feedback gains
%          paramFixed              - model and learner parameters
%          tStart                  - start time (nondimensional)
% Outputs: f_objective  - weighted objective (weights in loadBipedModelParameters)
%          stateVar0    - midstance state at the end of the iteration
%          tEnd         - end time
%          f_energy     - energy per unit time over the iteration
%          f_constraint - squared change in midstance angular rate and foot position

paramController.Odd.theta_end_nominal = ...
    pInputControllerNominal(1);
paramController.Odd.ydot_atMidstance_nominal_beltframe = ...
    pInputControllerNominal(2);
paramController.Odd.PushoffImpulseMagnitude_nominal = ...
    pInputControllerNominal(3);
paramController.Odd.y_atMidstance_nominal_slopeframe = ...
    pInputControllerNominal(4);
paramController.Odd.SUMy_atMidstance_nominal_slopeframe = ...
    pInputControllerNominal(5);

paramController.Even.theta_end_nominal = ...
    pInputControllerNominal(6);
paramController.Even.ydot_atMidstance_nominal_beltframe = ...
    pInputControllerNominal(7);
paramController.Even.PushoffImpulseMagnitude_nominal = ...
    pInputControllerNominal(8);
paramController.Even.y_atMidstance_nominal_slopeframe = ...
    pInputControllerNominal(9);
paramController.Even.SUMy_atMidstance_nominal_slopeframe = ...
    pInputControllerNominal(10);

paramFixed.numSteps = paramFixed.Learner.numStepsPerIteration;
[stateVar0,tStore,stateStore,EmetTotalStore, EmetPerTimeStore, ...
    tStepStore] = ...
    simulateManySteps_AsymmetricControl(stateVar0_Model,paramController,paramFixed,tStart);

f_energy = sum(EmetTotalStore)/sum(tStepStore);
tEnd = tStore{end}(end);

f_constraint = (norm(stateVar0_Model(2:3)-stateVar0(2:3)))^2;

lambdaEnergyVsPeriodicity = paramFixed.lambdaEnergyVsPeriodicity;
f_objective = lambdaEnergyVsPeriodicity*f_energy+(1-lambdaEnergyVsPeriodicity)*f_constraint;

% Symmetry term: squared difference between the two stance-end leg angles.
angleThetaEnd_1 = stateStore{1}(end,1); angleThetaEnd_2 = stateStore{2}(end,1);
f_symmetry = (angleThetaEnd_1-angleThetaEnd_2)^2;

f_objective = paramFixed.lambdaEnergyVsSymmetry*f_objective + ...
    (1-paramFixed.lambdaEnergyVsSymmetry)*f_symmetry*paramFixed.symmetryMultiplier;

end
