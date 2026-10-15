function validate_manuscript_model(output_dir)
if nargin < 1; output_dir = tempname; end
if ~isfolder(output_dir); mkdir(output_dir); end
single = run_manuscript_trial(fullfile(output_dir, 'single.mat'), seed=42, max_iterations=1);
assert(size(single.param, 2) == 1);
first = run_manuscript_trial(fullfile(output_dir, 'first.mat'), seed=42, max_iterations=8);
repeat = run_manuscript_trial(fullfile(output_dir, 'repeat.mat'), seed=42, max_iterations=8);
assert(isequaln(first.param, repeat.param));
assert(isequaln(first.energy, repeat.energy));
loaded = run_manuscript_trial(fullfile(output_dir, 'first.mat'), seed=42, max_iterations=8);
assert(isequaln(loaded.param, first.param));
different = run_manuscript_trial(fullfile(output_dir, 'different.mat'), seed=43, max_iterations=8);
assert(~isequaln(first.param, different.param));
try
    run_manuscript_trial(fullfile(output_dir, 'first.mat'), seed=43, max_iterations=8);
    error('LocAd:ExpectedRejection', 'Mismatched checkpoint should fail.');
catch error_info
    assert(strcmp(error_info.identifier, 'LocAd:CheckpointMismatch'));
end
sweep_dir = fullfile(output_dir, 'sweep');
first_sweep = run_manuscript_sweep(sweep_dir, speed_differences=0.7, learning_rates=0.02, energy_weights=0.5, trials=2, max_iterations=8);
second_sweep = run_manuscript_sweep(sweep_dir, speed_differences=0.7, learning_rates=0.02, energy_weights=0.5, trials=2, max_iterations=8);
assert(isequal(first_sweep, second_sweep));
disp('Manuscript engine: seeded repeatability, saved-trial reuse, mismatch rejection and sweep resume passed.');
end
