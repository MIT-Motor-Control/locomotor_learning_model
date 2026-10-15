"""High-level workflow helpers for the Python implementation."""

from __future__ import annotations

import platform
import secrets
from copy import deepcopy
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from locomotor_learning_model.initializing.get_treadmill_speed import (
    get_treadmill_speed,
)
from locomotor_learning_model.initializing.load_initial_body_state import (
    load_initial_body_state,
)
from locomotor_learning_model.initializing.load_learnable_parameters_initial import (
    load_learnable_parameters_initial,
)
from locomotor_learning_model.learning.simulate_learning_step_by_step import (
    simulate_learning_step_by_step,
    validate_learning_state,
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
from locomotor_learning_model.post_processing.post_process_after_learning import (
    post_process_after_learning,
)
from locomotor_learning_model.post_processing.post_process_helper_plots import (
    post_process_helper_plots,
)

from .artifacts import (
    MODEL_VERSION,
    SCHEMA_VERSION,
    configuration_digest,
    load_archive,
    save_archive,
    save_json,
)


@dataclass
class SimulationResults:
    """Container for the full Python simulation output."""

    param_fixed: dict[str, Any]
    param_controller_gains: dict[str, Any]
    initial_controller_parameters: np.ndarray
    initial_state: np.ndarray
    controller_history_8d: np.ndarray
    controller_history_10d: np.ndarray
    final_state: np.ndarray
    t_store: list[np.ndarray]
    state_store: list[np.ndarray]
    emet_total_store: np.ndarray
    emet_per_time_store: np.ndarray
    t_step_store: np.ndarray
    ework_pushoff_store: np.ndarray
    ework_heelstrike_store: np.ndarray
    edot_store_iteration_average: np.ndarray
    t_total_iteration_store: np.ndarray
    summary: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)
    learning_diagnostics: dict[str, Any] = field(default_factory=dict)


def _expand_controller_history(controller_history_8d: np.ndarray) -> np.ndarray:
    """Convert the 8D learning state back to the 10D controller representation."""
    return np.vstack(
        [
            controller_history_8d[:3, :],
            np.zeros((2, controller_history_8d.shape[1])),
            controller_history_8d[3:8, :],
        ]
    )


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _without_runtime(parameters: dict[str, Any]) -> dict[str, Any]:
    return deepcopy({key: value for key, value in parameters.items() if key != "_rng"})


def _configuration(split_or_tied, speed_protocol, transition_time, split_speed_diff_ms):
    param_fixed: dict[str, Any] = {}
    param_fixed = load_biped_model_parameters(param_fixed)
    param_fixed = load_sensory_noise_parameters(param_fixed)
    param_controller_gains = load_controller_gain_parameters(param_fixed)
    param_fixed = load_learner_parameters(param_fixed)
    param_fixed = load_protocol_parameters(
        param_fixed,
        split_or_tied=split_or_tied,
        speed_protocol=speed_protocol,
        transition_time=transition_time,
        split_speed_diff_ms=split_speed_diff_ms,
    )
    param_fixed = load_stored_memory_parameters_control_vs_speed(param_fixed)

    p_input = load_learnable_parameters_initial(param_fixed)
    initial_state = load_initial_body_state(p_input)
    return {
        "param_fixed": param_fixed,
        "param_controller_gains": param_controller_gains,
        "initial_controller_parameters": p_input,
        "initial_state": initial_state,
    }


def _scientific_configuration(configuration):
    value = deepcopy(configuration)
    value["param_fixed"].pop("num_iterations", None)
    value["param_fixed"].pop("numIterations", None)
    return value


def _validate_checkpoint_parameters(saved):
    """Check immutable settings separately from the memory that learning updates."""
    initial = saved["configuration"]["param_fixed"]
    actual = deepcopy(saved["param_fixed"])
    expected_keys = set(initial) | {
        "num_steps",
        "_lastFallPenalty",
        "_lastFallPenaltyMode",
        "_lastMinCom",
    }
    if set(actual) != expected_keys:
        raise ValueError("Checkpoint parameter fields are incomplete or unsupported")
    for key in ("num_steps", "_lastFallPenalty", "_lastFallPenaltyMode", "_lastMinCom"):
        actual.pop(key)
    memory = actual["storedmemory"]["controlSlopeVsContext"]
    if (
        not isinstance(memory, np.ndarray)
        or memory.shape != (8, 2)
        or not np.all(np.isfinite(memory))
    ):
        raise ValueError("Invalid checkpoint learned memory")
    actual["storedmemory"]["controlSlopeVsContext"] = initial["storedmemory"][
        "controlSlopeVsContext"
    ]
    actual["num_iterations"] = initial["num_iterations"]
    actual["numIterations"] = initial["numIterations"]
    if configuration_digest(actual) != configuration_digest(initial):
        raise ValueError("Checkpoint parameters do not agree with scientific configuration")
    for key, value in saved["configuration"]["param_controller_gains"].items():
        if key not in ("Odd", "Even") and saved["param_controller_gains"].get(key) != value:
            raise ValueError(f"Checkpoint controller gain changed: {key}")


def _read_checkpoint(path):
    saved = load_archive(path, "checkpoint")
    required = {
        "metadata",
        "configuration",
        "configuration_sha256",
        "param_fixed",
        "param_controller_gains",
        "learning_state",
        "rng_state",
    }
    if set(saved) != required or not all(
        isinstance(saved[key], dict) for key in required - {"configuration_sha256", "rng_state"}
    ):
        raise ValueError("Checkpoint fields are incomplete or unsupported")
    config = saved["configuration"]
    if set(config) != {
        "param_fixed",
        "param_controller_gains",
        "initial_controller_parameters",
        "initial_state",
    }:
        raise ValueError("Checkpoint configuration fields are incomplete or unsupported")
    if not isinstance(config["param_fixed"], dict) or not isinstance(
        config["param_controller_gains"], dict
    ):
        # Preserve the invalid-checkpoint ValueError contract.
        raise ValueError("Checkpoint parameters must be dictionaries")  # noqa: TRY004
    for name, shape in (("initial_controller_parameters", (10,)), ("initial_state", (5,))):
        value = config[name]
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype.kind not in "fiu"
            or not np.all(np.isfinite(value))
        ):
            raise ValueError(f"Invalid checkpoint {name}")
    metadata = saved["metadata"]
    for key in ("completed_iterations", "target_iterations"):
        _positive_integer(metadata.get(key), f"checkpoint {key}")
    if metadata["completed_iterations"] > metadata["target_iterations"]:
        raise ValueError("Checkpoint completed count exceeds its target")
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Invalid checkpoint seed")
    rng_state = saved["rng_state"]
    if not isinstance(rng_state, tuple) or len(rng_state) != 5:
        raise ValueError("Invalid checkpoint random generator state")
    algorithm, keys, position, has_gauss, cached_gaussian = rng_state
    if (
        algorithm != "MT19937"
        or not isinstance(keys, np.ndarray)
        or keys.shape != (624,)
        or keys.dtype != np.dtype("uint32")
        or not isinstance(position, int)
        or not 0 <= position <= 624
        or has_gauss not in (0, 1)
        or not isinstance(cached_gaussian, (int, float))
        or not np.isfinite(cached_gaussian)
    ):
        raise ValueError("Invalid checkpoint random generator state")
    return saved


def run_simulation(
    seed: int | None = None,
    num_iterations: int | None = None,
    split_or_tied: str | None = None,
    speed_protocol: str | None = None,
    transition_time: float | None = None,
    split_speed_diff_ms: float | None = None,
    make_plots: bool = True,
    output_dir: str | Path | None = None,
    *,
    resume_from: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    stop_after: int | None = None,
) -> SimulationResults:
    """Run or resume the baseline Python model without changing scientific defaults.

    ``num_iterations`` is the total target, including completed iterations on resume.
    ``stop_after`` stops at an absolute iteration boundary and requires a checkpoint
    path. Checkpoints capture learning state before post-processing; the latter is
    replayed from the initial body state after each call. Random draws use a private
    MT19937 generator, preserving the historical seeded sequence.
    """
    if split_or_tied is not None and split_or_tied not in ("split", "tied"):
        raise ValueError("split_or_tied must be 'split' or 'tied'")
    if speed_protocol is not None and (
        not isinstance(speed_protocol, str) or not speed_protocol.strip()
    ):
        raise ValueError("speed_protocol must be a non-empty protocol name")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32
    ):
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    if num_iterations is not None:
        num_iterations = _positive_integer(num_iterations, "num_iterations")
    if checkpoint_every is not None:
        checkpoint_every = _positive_integer(checkpoint_every, "checkpoint_every")
        if checkpoint_path is None:
            raise ValueError("checkpoint_every requires checkpoint_path")
    if stop_after is not None:
        stop_after = _positive_integer(stop_after, "stop_after")
        if checkpoint_path is None:
            raise ValueError("stop_after requires checkpoint_path")
    for name, value in (
        ("transition_time", transition_time),
        ("split_speed_diff_ms", split_speed_diff_ms),
    ):
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")

    saved = _read_checkpoint(resume_from) if resume_from is not None else None
    if saved is not None:
        if (
            saved["metadata"].get("numpy") != np.__version__
            or saved["metadata"].get("scipy") != scipy.__version__
        ):
            raise ValueError(
                "Exact checkpoint resume requires the recorded NumPy and SciPy versions"
            )
        configuration = saved["configuration"]
        original = configuration["param_fixed"]
        if (
            configuration_digest(_scientific_configuration(configuration))
            != saved["configuration_sha256"]
        ):
            raise ValueError("Checkpoint configuration checksum does not match")
        requested = {
            "SplitOrTied": split_or_tied,
            "speedProtocol": speed_protocol,
            "transitionTime": transition_time,
            "splitSpeedDiffMS": split_speed_diff_ms,
        }
        for key, value in requested.items():
            if value is not None and value != original.get(key):
                raise ValueError(f"Cannot change {key} when resuming a checkpoint")
        if seed is not None and int(seed) != saved["metadata"]["seed"]:
            raise ValueError("Cannot change seed when resuming a checkpoint")
        seed = saved["metadata"]["seed"]
        # Prevent a change to default scientific parameters from silently reusing old state.
        current = _configuration(
            original["SplitOrTied"],
            original["speedProtocol"],
            original["transitionTime"],
            original.get("splitSpeedDiffMS") if original["SplitOrTied"] == "split" else None,
        )
        if (
            configuration_digest(_scientific_configuration(current))
            != saved["configuration_sha256"]
        ):
            raise ValueError("Checkpoint scientific configuration is incompatible with this model")
        _validate_checkpoint_parameters(saved)
        param_fixed = deepcopy(saved["param_fixed"])
        param_controller_gains = deepcopy(saved["param_controller_gains"])
        learning_state = saved["learning_state"]
        validate_learning_state(learning_state, param_fixed)
        if saved["metadata"]["completed_iterations"] != learning_state["completed_iterations"]:
            raise ValueError("Checkpoint iteration metadata does not agree with learner state")
        target = (
            num_iterations if num_iterations is not None else saved["metadata"]["target_iterations"]
        )
        rng = np.random.RandomState()
        try:
            rng.set_state(saved["rng_state"])
        except (ValueError, TypeError, IndexError) as error:
            raise ValueError("Invalid checkpoint random generator state") from error
    else:
        seed = int(seed) if seed is not None else secrets.randbits(32)
        configuration = _configuration(
            split_or_tied or "split", speed_protocol, transition_time, split_speed_diff_ms
        )
        param_fixed = deepcopy(configuration["param_fixed"])
        param_controller_gains = deepcopy(configuration["param_controller_gains"])
        target = num_iterations if num_iterations is not None else param_fixed["num_iterations"]
        learning_state = None
        rng = np.random.RandomState(seed)
    target = _positive_integer(target, "num_iterations")
    completed = learning_state["completed_iterations"] if learning_state is not None else 0
    limit = stop_after if stop_after is not None else target
    if not completed <= limit <= target:
        raise ValueError("Require completed checkpoint iterations <= stop_after <= num_iterations")
    param_fixed["num_iterations"] = target
    param_fixed["numIterations"] = target
    param_fixed["_rng"] = rng
    p_input_controller_asymmetric_nominal = configuration["initial_controller_parameters"].copy()
    state_var0_model_before_learning = configuration["initial_state"].copy()
    state_var0_model = state_var0_model_before_learning.copy()
    v_a, v_b = get_treadmill_speed(0, param_fixed["imposedFootSpeeds"])
    context_now = np.array([v_a, v_b])
    digest = configuration_digest(_scientific_configuration(configuration))
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "artifact_role": "baseline_model_example_not_a_published_fit",
        "seed": seed,
        "rng": "numpy.random.RandomState/MT19937",
        "target_iterations": target,
        "completed_iterations": limit,
        "learning_complete": limit == target,
        "checkpoint_boundary": "after_complete_learning_iteration_before_postprocessing",
        "steps_per_iteration": param_fixed["Learner"]["numStepsPerIteration"],
        "configuration_sha256": digest,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "created_at": datetime.now(UTC).isoformat(),
        "resumed_from": str(Path(resume_from).resolve()) if resume_from is not None else None,
    }

    def write_checkpoint(state):
        validate_learning_state(state, param_fixed)
        checkpoint_metadata = dict(
            metadata,
            completed_iterations=state["completed_iterations"],
            learning_complete=state["completed_iterations"] == target,
        )
        payload = {
            "metadata": checkpoint_metadata,
            "configuration": configuration,
            "configuration_sha256": digest,
            "param_fixed": _without_runtime(param_fixed),
            "param_controller_gains": deepcopy(param_controller_gains),
            "learning_state": state,
            "rng_state": rng.get_state(),
        }
        save_archive(checkpoint_path, "checkpoint", payload)
        save_json(Path(checkpoint_path).with_suffix(".metadata.json"), checkpoint_metadata)

    controller_history_8d, learning_diagnostics = simulate_learning_step_by_step(
        param_fixed,
        p_input_controller_asymmetric_nominal,
        state_var0_model,
        context_now,
        param_controller_gains,
        return_diagnostics=True,
        resume_state=learning_state,
        checkpoint_callback=write_checkpoint if checkpoint_path is not None else None,
        checkpoint_every=checkpoint_every,
        stop_after=limit,
    )
    controller_history_10d = _expand_controller_history(controller_history_8d)

    (
        final_state,
        t_store,
        state_store,
        emet_total_store,
        emet_per_time_store,
        t_step_store,
        ework_pushoff_store,
        ework_heelstrike_store,
        edot_store_iteration_average,
        t_total_iteration_store,
    ) = post_process_after_learning(
        controller_history_10d,
        state_var0_model_before_learning,
        param_controller_gains,
        param_fixed,
        False,
    )

    summary = post_process_helper_plots(
        final_state,
        t_store,
        state_store,
        emet_total_store,
        emet_per_time_store,
        t_step_store,
        param_controller_gains,
        param_fixed,
        False,
        ework_pushoff_store,
        ework_heelstrike_store,
        edot_store_iteration_average,
        t_total_iteration_store,
        create_plots=make_plots or output_dir is not None,
        show_plots=make_plots,
        output_dir=output_dir,
    )

    result = SimulationResults(
        param_fixed=_without_runtime(param_fixed),
        param_controller_gains=param_controller_gains,
        initial_controller_parameters=p_input_controller_asymmetric_nominal,
        initial_state=state_var0_model_before_learning,
        controller_history_8d=controller_history_8d,
        controller_history_10d=controller_history_10d,
        final_state=final_state,
        t_store=t_store,
        state_store=state_store,
        emet_total_store=emet_total_store,
        emet_per_time_store=emet_per_time_store,
        t_step_store=t_step_store,
        ework_pushoff_store=ework_pushoff_store,
        ework_heelstrike_store=ework_heelstrike_store,
        edot_store_iteration_average=edot_store_iteration_average,
        t_total_iteration_store=t_total_iteration_store,
        summary=summary,
        metadata=metadata,
        learning_diagnostics=learning_diagnostics,
    )
    if output_dir is not None:
        save_simulation_results(result, output_dir)
    return result


def save_simulation_results(results: SimulationResults, output_dir: str | Path) -> Path:
    """Save complete numerical results and concise metadata; no pickled objects."""
    directory = Path(output_dir)
    payload = {item.name: getattr(results, item.name) for item in fields(results)}
    path = directory / "run.npz"
    save_archive(path, "results", payload)
    scalar_summary = {key: value for key, value in results.summary.items() if np.isscalar(value)}
    save_json(
        directory / "metadata.json",
        dict(
            results.metadata,
            summary=scalar_summary,
            condition=results.param_fixed["SplitOrTied"],
            protocol=results.param_fixed["speedProtocol"],
        ),
    )
    return path


def load_simulation_results(path: str | Path) -> SimulationResults:
    """Load an existing result for analysis/replotting without training the learner."""
    payload = load_archive(path, "results")
    if set(payload) != {item.name for item in fields(SimulationResults)}:
        raise ValueError("Result archive fields are incomplete or unsupported")
    result = SimulationResults(**payload)
    n = result.metadata["completed_iterations"]
    if result.controller_history_8d.shape != (8, n) or result.controller_history_10d.shape != (
        10,
        n,
    ):
        raise ValueError("Invalid result controller history dimensions")
    return result


def plot_simulation_results(
    results: SimulationResults, output_dir: str | Path | None = None, show_plots: bool = False
) -> dict[str, Any]:
    """Replot saved trajectories; performs no model simulation or learning."""
    return post_process_helper_plots(
        results.final_state,
        results.t_store,
        results.state_store,
        results.emet_total_store,
        results.emet_per_time_store,
        results.t_step_store,
        results.param_controller_gains,
        results.param_fixed,
        False,
        results.ework_pushoff_store,
        results.ework_heelstrike_store,
        results.edot_store_iteration_average,
        results.t_total_iteration_store,
        create_plots=True,
        show_plots=show_plots,
        output_dir=output_dir,
    )
