# Manuscript simulation engine

This is the supplied fall-detecting MATLAB manuscript engine, with explicit seeds and portable saved trials. It is separate from the MATLAB/Python baseline implementations. Its default transition duration and belt-speed scaling match the updated manuscript Methods; see the [reproducibility guide](../../docs/reproducibility.md).

From the repository root in MATLAB:

```matlab
addpath('matlab/manuscript')
result = run_manuscript_trial('outputs/trial42.mat', seed=42, max_iterations=40);
```

Omit `max_iterations` for the complete 45-minute protocol. Early falls terminate learning according to the supplied detection rule. All settings, MATLAB version, engine source, final RNG state and numerical results are saved. An existing completed trial is reused only when its settings and engine match.

For an independent-trial sweep:

```matlab
run_manuscript_sweep('outputs/small_sweep', ...
    speed_differences=[0.4 0.7], learning_rates=0.02, ...
    energy_weights=0.5, trials=2, seed=42, max_iterations=40);
```

Rerunning skips verified completed trials. A failed or interrupted trial reruns from its fixed seed; this engine does not resume partway through a trial. For iteration-level learner checkpoints, use the [Python model](../../docs/model.md). The full default sweep has 60,000 trials; it is intentionally not part of the quick start or figure reconstruction.

`energy_weight` multiplies the energy objective; the manuscript convention is **ψ = 1 − energy_weight**. The supplied defaults are a **25-second** transition and speed divisor **2**. Both can be set explicitly for new experiments, and are saved with each trial.

Validation:

```matlab
validate_manuscript_model('outputs/validation/manuscript')
```

This checks step-budget consistency, seed repeatability, saved-trial reuse, mismatched-setting rejection and sweep restart. `engine/source_manifest.json` identifies original files and changes. The metabolic transient is initialized from up to 30 iteration-averaged samples.
