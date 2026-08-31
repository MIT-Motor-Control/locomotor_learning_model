"""Compare a deterministic MATLAB reference export against the Python port."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from locomotor_learning_model.initializing.get_treadmill_speed import (
    get_treadmill_speed,
)
from locomotor_learning_model.initializing.load_initial_body_state import (
    load_initial_body_state,
)
from locomotor_learning_model.initializing.load_learnable_parameters_initial import (
    load_learnable_parameters_initial,
)
from locomotor_learning_model.learning.f_objective_asymmetric_nominal import (
    f_objective_asymmetric_nominal,
)
from locomotor_learning_model.learning.simulate_learning_step_by_step import (
    simulate_learning_step_by_step,
)
from locomotor_learning_model.parameter_loading.load_biped_model_parameters import (
    load_biped_model_parameters,
)
from locomotor_learning_model.parameter_loading.load_controller_gain_parameters import (
    load_controller_gain_parameters,
)
from locomotor_learning_model.parameter_loading.load_learner_parameters import (
    load_learner_parameters,
)
from locomotor_learning_model.parameter_loading.load_protocol_parameters import (
    load_protocol_parameters,
)
from locomotor_learning_model.parameter_loading.load_sensory_noise_parameters import (
    load_sensory_noise_parameters,
)
from locomotor_learning_model.parameter_loading.load_stored_memory_parameters_control_vs_speed import (
    load_stored_memory_parameters_control_vs_speed,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Python implementation against a deterministic MATLAB reference "
            "exported with matlab/validation/export_reference_run.m."
        )
    )
    parser.add_argument(
        "reference",
        type=Path,
        help="Path to the MATLAB .mat reference file.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-10,
        help="Maximum allowed absolute difference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Compare the current Python implementation against a MATLAB export."""
    args = build_parser().parse_args(argv)
    matlab_reference = loadmat(args.reference)

    param_fixed = {}
    param_fixed = load_biped_model_parameters(param_fixed)
    param_fixed = load_sensory_noise_parameters(param_fixed)
    param_controller_gains = load_controller_gain_parameters(param_fixed)
    param_fixed = load_learner_parameters(param_fixed)
    param_fixed = load_protocol_parameters(param_fixed)
    param_fixed = load_stored_memory_parameters_control_vs_speed(param_fixed)
    param_fixed["Learner"]["noiseSTDExploratory"] = 0
    param_fixed["noiseEnergySensory"] = 0
    param_fixed["num_iterations"] = int(matlab_reference["controllerStore8D"].shape[1])
    param_fixed["numIterations"] = param_fixed["num_iterations"]

    p_input = load_learnable_parameters_initial(param_fixed)
    state_var0 = load_initial_body_state(p_input)
    v_a, v_b = get_treadmill_speed(0, param_fixed["imposedFootSpeeds"])
    context_now = np.array([v_a, v_b])
    objective_value = f_objective_asymmetric_nominal(
        p_input,
        state_var0,
        param_controller_gains,
        param_fixed,
        0.0,
    )[0]
    controller_store_8d = simulate_learning_step_by_step(
        param_fixed,
        p_input,
        state_var0,
        context_now,
        param_controller_gains,
    )

    comparisons = {
        "objectiveValue": abs(
            float(matlab_reference["objectiveValue"].squeeze()) - float(objective_value)
        ),
        "pInput": float(
            np.max(np.abs(matlab_reference["pInput"].squeeze() - p_input))
        ),
        "stateVar0": float(
            np.max(np.abs(matlab_reference["stateVar0"].squeeze() - state_var0))
        ),
        "contextNow": float(
            np.max(np.abs(matlab_reference["contextNow"].squeeze() - context_now))
        ),
        "controllerStore8D": float(
            np.max(np.abs(matlab_reference["controllerStore8D"] - controller_store_8d))
        ),
    }

    print("MATLAB/Python deterministic parity check")
    for name, value in comparisons.items():
        print(f"{name}: {value:.3e}")

    if any(value > args.tolerance for value in comparisons.values()):
        print("Parity check failed.")
        return 1

    print("Parity check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
