function [Anew, errorOldLinear_MostRecentStep, errorNewLinear_MostRecentStep] = RLSupdate(Aold,beta,InputNowStore,...
    OutputNextStore,muMeasurement,numStepsToUse)
% Fit a linear model with intercept by least squares to the most recent observations.
% There is no recursive update or forgetting; beta and muMeasurement are not used.
% Inputs:  Aold            - previous coefficients, used only for the old error
%          InputNowStore   - model inputs, one column per observation
%          OutputNextStore - model outputs, one column per observation
%          numStepsToUse   - number of most recent observations in the fit
% Outputs: Anew            - coefficients, [outputs x (inputs + 1)], intercept last
%          errorOldLinear_MostRecentStep - residual of the latest observation under Aold
%          errorNewLinear_MostRecentStep - residual of the latest observation under Anew

numInputDimensions = size(InputNowStore,1);
numOutputDimensions = size(OutputNextStore,1);

InputNowStore = InputNowStore(:,end-numStepsToUse+1:end);
OutputNextStore = OutputNextStore(:,end-numStepsToUse+1:end);

% Rows are observations; append the intercept column.
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
