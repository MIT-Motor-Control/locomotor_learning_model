function run_simulation()
%RUN_SIMULATION Add the MATLAB sources to the path and execute the manuscript pipeline.

repoDir = fileparts(mfilename('fullpath'));
addpath(fullfile(repoDir, 'src'));
rootSimulateLearningWhileWalking;
end
