function [f_objective,stateVar0,tEnd,f_energy,f_constraint,fall_info,yPosStore] = ...
    fObjective_AsymmetricNominal_8DTo10D(pInputControllerNominal,stateVar0_Model, ...
    paramController,paramFixed,tStart,SwingCostCoeff,iStride)

temp = zeros(10,1);
temp(1:3) = pInputControllerNominal(1:3);
temp(6:10) = pInputControllerNominal(4:8);
pInputControllerNominal = temp;

[f_objective,stateVar0,tEnd,f_energy,f_constraint,fall_info,yPosStore] = ...
    fObjective_AsymmetricNominal(pInputControllerNominal,stateVar0_Model, ...
    paramController,paramFixed,tStart,SwingCostCoeff,iStride);

end % checked and essential