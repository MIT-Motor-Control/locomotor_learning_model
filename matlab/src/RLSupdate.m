function [Anew, errorOldLinear_MostRecentStep, errorNewLinear_MostRecentStep] = RLSupdate(Aold,beta,InputNowStore,...
    OutputNextStore,muMeasurement,numStepsToUse)
% Fit a linear model to a finite window of recent steps.
% beta and muMeasurement are unused compatibility arguments.

numInputDimensions = size(InputNowStore,1);
numOutputDimensions = size(OutputNextStore,1);

% picking only a certain number of steps: each column is a step
InputNowStore = InputNowStore(:,end-numStepsToUse+1:end);
OutputNextStore = OutputNextStore(:,end-numStepsToUse+1:end);

% transpose, so that each row is a step
InputNowStore = InputNowStore';
OutputNextStore = OutputNextStore';

% adding a column of ones for the constant term
InputNowStore = [InputNowStore ones(size(InputNowStore,1),1)]; % each row is a new step

% Refit the rolling window directly, without exponential forgetting.
Anew = InputNowStore\OutputNextStore; 

%% find the error on all steps and then the most recent step using the pre-change 
Aold = Aold';
errorOldLinear_AllSteps = OutputNextStore-InputNowStore*Aold;
errorOldLinear_MostRecentStep = errorOldLinear_AllSteps(end,:)'; % rows are steps because of the old transpose

errorNewLinear_AllSteps = OutputNextStore-InputNowStore*Anew;
errorNewLinear_MostRecentStep = errorNewLinear_AllSteps(end,:)'; % rows are steps because of the old transpose
% doing another transpose, so the most recent error is a column vector

%% transpose it anyway, so steps are columns instead of rows
Anew = Anew';

end
