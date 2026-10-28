function [f_objective,stateVar0,tEnd,f_energy,f_constraint] = ...
    fObjective_AsymmetricNominal_8DTo10D(pInputControllerNominal,stateVar0_Model, ...
    paramController,paramFixed,tStart)
% Evaluate fObjective_AsymmetricNominal for an 8-D learned controller.
% The odd-step position targets (entries 4-5 of the 10-D layout) are set to
% zero; other inputs and all outputs are as in fObjective_AsymmetricNominal.

temp = zeros(10,1);
temp(1:3) = pInputControllerNominal(1:3);
temp(6:10) = pInputControllerNominal(4:8);
pInputControllerNominal = temp;

[f_objective,stateVar0,tEnd,f_energy,f_constraint] = ...
    fObjective_AsymmetricNominal(pInputControllerNominal,stateVar0_Model, ...
    paramController,paramFixed,tStart);

end
