function [Anew, errorOldLinear_MostRecentStep, errorNewLinear_MostRecentStep] = RLSupdate(Aold,beta,InputNowStore,...
    OutputNextStore,muMeasurement,numStepsToUse)
% Fit a linear model to a finite window of recent observations.

% beta and muMeasurement are unused compatibility arguments.
numInputDimensions = size(InputNowStore,1);
numOutputDimensions = size(OutputNextStore,1);

% Input and output columns correspond to successive observations.
InputNowStore = InputNowStore(:,end-numStepsToUse+1:end);
OutputNextStore = OutputNextStore(:,end-numStepsToUse+1:end);

InputNowStore = InputNowStore';
OutputNextStore = OutputNextStore';

% Include an intercept after converting samples to rows.
InputNowStore = [InputNowStore ones(size(InputNowStore,1),1)];

Anew = InputNowStore\OutputNextStore;

Aold = Aold';
errorOldLinear_AllSteps = OutputNextStore-InputNowStore*Aold;
errorOldLinear_MostRecentStep = errorOldLinear_AllSteps(end,:)';

errorNewLinear_AllSteps = OutputNextStore-InputNowStore*Anew;
errorNewLinear_MostRecentStep = errorNewLinear_AllSteps(end,:)';

Anew = Anew';

end
