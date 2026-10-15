function result = run_manuscript_trial(output_file, options)
arguments
    output_file (1,1) string
    options.seed (1,1) double {mustBeInteger, mustBeNonnegative} = 0
    options.speed_difference (1,1) double {mustBePositive, mustBeLessThan(options.speed_difference, 2)} = 0.7
    options.learning_rate (1,1) double {mustBeNonnegative, mustBeFinite} = 0.02
    options.energy_weight (1,1) double {mustBeGreaterThanOrEqual(options.energy_weight, 0), mustBeLessThanOrEqual(options.energy_weight, 1)} = 0.5
    options.split_minutes (1,1) double {mustBePositive, mustBeFinite} = 45
    options.exploration_noise (1,1) double {mustBeNonnegative, mustBeFinite} = 0.01
    options.swing_cost (1,1) double {mustBePositive, mustBeFinite} = 1
    options.max_iterations (1,1) double {mustBePositive} = Inf
    options.transition_seconds (1,1) double {mustBePositive, mustBeFinite} = 25
    options.speed_divisor (1,1) double {mustBePositive, mustBeFinite} = 2
end
if options.seed > 2^32 - 1 || (~isinf(options.max_iterations) && mod(options.max_iterations, 1) ~= 0)
    error('LocAd:InvalidConfiguration', 'Seed must fit uint32 and max_iterations must be an integer or Inf.');
end
here = fileparts(mfilename('fullpath'));
old_path = path;
old_rng = rng;
cleanup = onCleanup(@() restore_session(old_path, old_rng));
addpath(fullfile(here, 'engine'), '-begin');
engine_files = dir(fullfile(here, 'engine', '*.m'));
source = struct();
for i = 1:numel(engine_files)
    file = fullfile(engine_files(i).folder, engine_files(i).name);
    source.(erase(engine_files(i).name, '.m')) = fileread(file);
end
if isfile(output_file)
    result = load(output_file);
    if ~isfield(result, 'metadata') || ~isequaln(result.metadata.options, options) || ...
            ~isequal(result.metadata.engine_source, source)
        error('LocAd:CheckpointMismatch', 'Existing trial uses different settings or engine source. Choose a new output file.');
    end
    validate_result(result);
    fprintf('Loaded completed trial: %s\n', output_file);
    return
end
controller = [0.328221262798818; 0.310751796902254; 0.153556843539029; ...
    0.328221491356562; 0.310751694570805; 0.153557221281688; ...
    -0.000000038953735; -0.000000038953735];
slow = 1 - options.speed_difference / 2;
fast = 1 + options.speed_difference / 2;
[result.time, result.sla, result.sta, result.energy, result.speed, result.state, ...
    result.steptime_fast, result.steptime_slow, result.steplength_fast, result.steplength_slow, ...
    result.param, result.fall_flag, result.min_vert_pos] = run_sim( ...
    options.speed_difference, options.split_minutes, slow, fast, options.exploration_noise, ...
    options.learning_rate, options.energy_weight, controller, options.swing_cost, options);
result.metadata = struct('schema_version', 1, 'options', options, ...
    'matlab_version', version, 'engine_source', source, ...
    'completed_iterations', size(result.param, 2), 'rng_final', rng, ...
    'kind', 'completed_trial', 'created_at', char(datetime('now', 'TimeZone', 'UTC')));
validate_result(result);
parent = fileparts(output_file);
if strlength(parent) == 0; parent = "."; end
if ~isfolder(parent); mkdir(parent); end
temporary = string(tempname(parent)) + ".mat";
remove_temporary = onCleanup(@() delete_if_present(temporary));
save(temporary, '-struct', 'result', '-v7');
movefile(temporary, output_file);
fprintf('Saved %d completed iterations: %s\n', size(result.param, 2), output_file);
end

function validate_result(result)
if ~isfield(result, 'metadata') || result.metadata.schema_version ~= 1 || ~strcmp(result.metadata.kind, 'completed_trial')
    error('LocAd:InvalidResult', 'Unsupported saved-trial schema.');
end
required = {'energy', 'time', 'sla', 'sta', 'param', 'fall_flag'};
for i = 1:numel(required)
    if ~isfield(result, required{i}) || isempty(result.(required{i})) || any(~isfinite(result.(required{i})), 'all')
        error('LocAd:InvalidResult', 'Missing, empty or nonfinite result: %s', required{i});
    end
end
if size(result.param, 1) ~= 8 || numel(result.energy) ~= numel(result.time)
    error('LocAd:InvalidResult', 'Unexpected controller or time-series dimensions.');
end
end

function restore_session(old_path, old_rng)
path(old_path);
rng(old_rng);
end

function delete_if_present(file)
if isfile(file); delete(file); end
end
