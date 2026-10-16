function run_simulation()
%RUN_SIMULATION Run the baseline MATLAB model and its summary plots.

repoDir = fileparts(mfilename('fullpath'));
addpath(fullfile(repoDir, 'src'));
rootSimulateLearningWhileWalking;
end
