function [f_objective,stateVar0,tEnd,f_energy,f_constraint,fall_info,yPosStore] = ...
    fObjective_AsymmetricNominal(pInputControllerNominal,stateVar0_Model, ...
    paramController,paramFixed,tStart,SwingCostCoeff,iStride)
% Evaluate the energy, periodicity and symmetry objective over one learning iteration.
% Inputs:  pInputControllerNominal - 10 controller targets; 1:5 odd steps, 6:10 even steps,
%              each ordered as endstance leg angle, midstance speed, push-off impulse,
%              midstance position, summed position error
%          stateVar0_Model - midstance state at the start of the iteration
%          paramController - feedback gains; paramFixed - model and learner settings
%          tStart - start time; SwingCostCoeff - swing-cost multiplier
%          iStride - learning iteration, recorded on a fall
% Outputs: f_objective - weighted objective; stateVar0 - final midstance state
%          tEnd - end time; f_energy - metabolic power; f_constraint - squared periodicity error
%          fall_info, yPosStore - fall flag and minimum heights from simulateManySteps_AsymmetricControl

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
[stateVar0,tStore,stateStore,EmetTotalStore,EmetPerTimeStore,tStepStore,EworkPushoffStore,EworkHeelstrikeStore,fall_info,yPosStore] = ...
    simulateManySteps_AsymmetricControl(stateVar0_Model,paramController,paramFixed,tStart,SwingCostCoeff,iStride);

f_energy = sum(EmetTotalStore)/sum(tStepStore);
tEnd = tStore{end}(end);

f_constraint = (norm(stateVar0_Model(2:3)-stateVar0(2:3)))^2;

lambdaEnergyVsPeriodicity = paramFixed.lambdaEnergyVsPeriodicity;
f_objective = lambdaEnergyVsPeriodicity*f_energy+(1-lambdaEnergyVsPeriodicity)*f_constraint;

angleThetaEnd_1 = stateStore{1}(end,1); angleThetaEnd_2 = stateStore{2}(end,1);
f_symmetry = (angleThetaEnd_1-angleThetaEnd_2)^2;

% Manuscript Eq. 1, (1-psi)*e + psi*M*s, with psi = 1-lambdaEnergyVsSymmetry,
% M = symmetryMultiplier and e = f_energy when lambdaEnergyVsPeriodicity = 1.
f_objective = paramFixed.lambdaEnergyVsSymmetry*f_objective + ...
    (1-paramFixed.lambdaEnergyVsSymmetry)*f_symmetry*paramFixed.symmetryMultiplier;

end
