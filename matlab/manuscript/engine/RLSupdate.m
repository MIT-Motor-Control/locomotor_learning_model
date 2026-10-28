function [Anew, errorOldLinear_MostRecentStep, errorNewLinear_MostRecentStep] = RLSupdate(Aold,beta,InputNowStore,...
    OutputNextStore,muMeasurement,numStepsToUse)
% Fit a linear model with intercept by least squares over the most recent observations.
% Inputs:  Aold - previous model, used only for the prediction errors
%          beta, muMeasurement - unused
%          InputNowStore, OutputNextStore - inputs and outputs, one observation per column
%          numStepsToUse - number of most recent observations in the fit
% Outputs: Anew - [numOutputs x numInputs+1] model; the last column is the intercept
%          errorOldLinear_MostRecentStep - residual of Aold on the most recent observation
%          errorNewLinear_MostRecentStep - residual of Anew on the most recent observation

numInputDimensions = size(InputNowStore,1);
numOutputDimensions = size(OutputNextStore,1);

InputNowStore = InputNowStore(:,end-numStepsToUse+1:end);
OutputNextStore = OutputNextStore(:,end-numStepsToUse+1:end);

InputNowStore = InputNowStore';
OutputNextStore = OutputNextStore';

InputNowStore = [InputNowStore ones(size(InputNowStore,1),1)];

Anew = InputNowStore\OutputNextStore;

Aold = Aold';
errorOldLinear_AllSteps = OutputNextStore-InputNowStore*Aold;
errorOldLinear_MostRecentStep = errorOldLinear_AllSteps(end,:)';

errorNewLinear_AllSteps = OutputNextStore-InputNowStore*Anew;
errorNewLinear_MostRecentStep = errorNewLinear_AllSteps(end,:)';

Anew = Anew';

end
