"""High-level workflow helpers for the Python implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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


def _expand_controller_history(controller_history_8d: np.ndarray) -> np.ndarray:
    """Convert the 8D learning state back to the 10D controller representation."""
    return np.vstack(
        [
            controller_history_8d[:3, :],
            np.zeros((2, controller_history_8d.shape[1])),
            controller_history_8d[3:8, :],
        ]
    )


def run_simulation(
    seed: int | None = None,
    num_iterations: int | None = None,
    split_or_tied: str = "split",
    speed_protocol: str | None = None,
    transition_time: float | None = None,
    make_plots: bool = True,
    output_dir: str | Path | None = None,
) -> SimulationResults:
    """Run the full Python simulation and post-processing pipeline."""
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

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
    )
    param_fixed = load_stored_memory_parameters_control_vs_speed(param_fixed)

    if num_iterations is not None:
        param_fixed["num_iterations"] = int(num_iterations)
        param_fixed["numIterations"] = int(num_iterations)

    p_input_controller_asymmetric_nominal = load_learnable_parameters_initial(param_fixed)
    state_var0_model = load_initial_body_state(p_input_controller_asymmetric_nominal)
    state_var0_model_before_learning = state_var0_model.copy()

    v_a, v_b = get_treadmill_speed(0, param_fixed["imposedFootSpeeds"])
    context_now = np.array([v_a, v_b])

    controller_history_8d = simulate_learning_step_by_step(
        param_fixed,
        p_input_controller_asymmetric_nominal,
        state_var0_model,
        context_now,
        param_controller_gains,
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

    return SimulationResults(
        param_fixed=param_fixed,
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
    )
