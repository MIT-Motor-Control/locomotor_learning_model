from __future__ import annotations

from typing import List

import numpy as np

from learning.get_treadmill_speed import get_treadmill_speed
from learning.f_objective_asymmetric_nominal_8d_to_10d import f_objective_asymmetric_nominal_8d_to_10d
from learning.compute_model_dynamics_model_rls_no_singular import compute_model_dynamics_model_rls_no_singular
from learning.compute_energy_dynamics_model_rls_no_singular import compute_energy_dynamics_model_rls_no_singular
from learning.compute_steady_energy_gradient_v2_no_singular import compute_steady_energy_gradient_v2_no_singular
from learning.gradient_of_error_from_memory_compute import gradient_of_error_from_memory_compute
from learning.error_from_memory_compute import error_from_memory_compute


def simulate_learning_step_by_step(
    param_fixed,
    p_input_controller_asymmetric_nominal: np.ndarray,
    state_var0_model: np.ndarray,
    context_now: np.ndarray,
    param_controller_gains,
) -> np.ndarray:
    """Python version of ``simulateLearningStepByStep.m``.

    This routine mirrors the MATLAB implementation line by line.  It performs a
    noisy gradient descent update of the controller parameters while updating the
    internal dynamics and energy models.  Memory formation is also handled
    analogously to the MATLAB code.
    """

    print(
        f"Simulating Learning While Walking ... ({param_fixed.num_iterations} strides)"
    )

    noise_std = param_fixed.Learner.noise_std_exploratory
    num_iterations = param_fixed.num_iterations
    include_internal_model = param_fixed.Learner.include_internal_model

    p_input_controller_asymmetric_nominal = p_input_controller_asymmetric_nominal[
        [0, 1, 2, 5, 6, 7, 8, 9]
    ]
    num_learning_dimensions = len(p_input_controller_asymmetric_nominal)
    num_state_dimensions = len(state_var0_model)

    alpha_energy_learning_rate = param_fixed.Learner.learning_rate

    p_input_now = p_input_controller_asymmetric_nominal.copy()
    t_start = 0.0

    adynamics_model_now = np.zeros(
        (num_state_dimensions - 1, num_learning_dimensions + num_state_dimensions)
    )
    adynamics_energy_now = np.zeros((1, num_learning_dimensions + num_state_dimensions))

    g_energy_gradient_now = np.zeros_like(p_input_controller_asymmetric_nominal)

    p_input_controller_store_ones_tried: List[np.ndarray] = []
    state_var_now_store: List[np.ndarray] = []
    state_var_next_store: List[np.ndarray] = []
    edot_store: List[float] = []
    p_input_controller_store_ones_considered_good: List[np.ndarray] = []
    gradient_energy_estimate_store: List[np.ndarray] = []
    adynamics_model_store: List[np.ndarray] = []
    adynamics_energy_store: List[np.ndarray] = []

    error_old_dynamics_model_store: List[np.ndarray] = []
    error_new_dynamics_model_store: List[np.ndarray] = []
    error_old_energy_model_store: List[np.ndarray] = []
    error_new_energy_model_store: List[np.ndarray] = []
    learning_on_or_not_store: List[int] = []
    memory_to_gradient_direction_cosine_store: List[float] = []

    g_gradient_for_memory_update_store: List[np.ndarray] = []
    error_from_memory_store: List[float] = []
    p_input_memory_now_store: List[np.ndarray] = []

    num_strides_to_use = param_fixed.Learner.num_steps_to_use_for_estimator
    state_var0_model_now = state_var0_model.copy()
    p_input_now_considered_good = p_input_now.copy()

    for i_stride in range(1, num_iterations + 1):
        if i_stride % 50 == 0:
            print(f"iStride = {i_stride}")

        delta_pinput_noise = noise_std * np.random.randn(num_learning_dimensions)
        delta_pinput_gradient = alpha_energy_learning_rate * (-g_energy_gradient_now)

        if param_fixed.Learner.should_we_use_trust_region:
            norm_grad = np.linalg.norm(delta_pinput_gradient)
            if norm_grad > param_fixed.Learner.trust_region_size * np.sqrt(num_learning_dimensions):
                delta_pinput_gradient = (
                    delta_pinput_gradient
                    / norm_grad
                    * param_fixed.Learner.trust_region_size
                    * np.sqrt(num_learning_dimensions)
                )

        p_input_memory_now = (
            param_fixed.storedmemory.nominal_control
            + param_fixed.storedmemory.control_slope_vs_context
            @ (context_now - param_fixed.storedmemory.nominal_context)
        )
        p_input_memory_now_store.append(p_input_memory_now)

        dir_toward_memory = p_input_memory_now - p_input_now_considered_good
        temp = float(
            np.dot(dir_toward_memory, -g_energy_gradient_now)
            / (np.linalg.norm(dir_toward_memory) * np.linalg.norm(g_energy_gradient_now) + 1e-8)
        )
        memory_to_gradient_direction_cosine_store.append(temp)

        power_move_to_memory = param_fixed.Learner.power_to_the_move_to_memory
        if np.isnan(temp):
            temp = 1 * param_fixed.Learner.learning_rate_toward_memory
        else:
            temp = (1 + temp) / 2
            temp = temp ** power_move_to_memory
            temp = temp * param_fixed.Learner.learning_rate_toward_memory

        delta_pinput_memory = temp * dir_toward_memory

        p_input_now_considered_good = p_input_now_considered_good + delta_pinput_memory
        p_input_now_considered_good = p_input_now_considered_good + delta_pinput_gradient
        p_input_now_to_try = p_input_now_considered_good + delta_pinput_noise

        (
            edot_now,
            state_var0_model_next,
            t_end,
            _,
            _,
        ) = f_objective_asymmetric_nominal_8d_to_10d(
            p_input_now_to_try,
            state_var0_model_now,
            param_controller_gains,
            param_fixed,
            t_start,
        )
        edot_now = edot_now * (1 + np.random.randn() * param_fixed.noise_energy_sensory)

        state_var_now_store.append(state_var0_model_now)
        state_var_next_store.append(state_var0_model_next)
        p_input_controller_store_ones_tried.append(p_input_now_to_try)
        p_input_controller_store_ones_considered_good.append(p_input_now_considered_good)
        edot_store.append(edot_now)

        if i_stride > num_strides_to_use:
            adynamics_model_now, error_old_dynamics_model, error_new_dynamics_model = (
                compute_model_dynamics_model_rls_no_singular(
                    adynamics_model_now,
                    np.column_stack(state_var_now_store),
                    np.column_stack(state_var_next_store),
                    np.column_stack(p_input_controller_store_ones_tried),
                    num_strides_to_use,
                )
            )
            error_old_dynamics_model_store.append(error_old_dynamics_model)
            error_new_dynamics_model_store.append(error_new_dynamics_model)

        if i_stride > num_strides_to_use:
            adynamics_energy_now, error_old_energy_model, error_new_energy_model = (
                compute_energy_dynamics_model_rls_no_singular(
                    adynamics_energy_now,
                    np.column_stack(state_var_now_store),
                    np.array(edot_store),
                    np.column_stack(p_input_controller_store_ones_tried),
                    num_strides_to_use,
                )
            )
            error_old_energy_model_store.append(error_old_energy_model)
            error_new_energy_model_store.append(error_new_energy_model)

        if i_stride > num_strides_to_use:
            g_energy_gradient_now = compute_steady_energy_gradient_v2_no_singular(
                adynamics_model_now,
                adynamics_energy_now,
                p_input_now_to_try,
                num_learning_dimensions,
                num_state_dimensions,
                include_internal_model,
            )
        gradient_energy_estimate_store.append(g_energy_gradient_now)
        adynamics_model_store.append(adynamics_model_now)
        adynamics_energy_store.append(adynamics_energy_now)

        if param_fixed.Learner.should_we_threshold_prediction_error and i_stride > num_strides_to_use:
            prediction_error_list_now = np.array(
                [error_old_energy_model, error_new_energy_model]
            )
            if np.any(np.abs(prediction_error_list_now) > param_fixed.Learner.prediction_error_threshold):
                g_energy_gradient_now = 0 * g_energy_gradient_now
                learning_on_or_not_store.append(0)
            else:
                learning_on_or_not_store.append(1)

        g_gradient_for_memory_update = gradient_of_error_from_memory_compute(
            param_fixed.storedmemory.control_slope_vs_context,
            p_input_now_considered_good,
            param_fixed,
            context_now,
        )

        n_memory_to_current_controller = -dir_toward_memory
        n_memory_to_current_controller = n_memory_to_current_controller / (
            np.linalg.norm(n_memory_to_current_controller) + 1e-8
        )

        n_v_controller = delta_pinput_gradient + delta_pinput_memory
        n_v_controller = n_v_controller / (np.linalg.norm(n_v_controller) + 1e-8)

        dot_memory_to_current_v_current = float(
            np.dot(n_memory_to_current_controller, n_v_controller)
        )

        temp = dot_memory_to_current_v_current
        power_for_memory_formation = param_fixed.Learner.power_to_the_memory_formation
        if np.isnan(temp):
            temp = 0.0
        else:
            temp = (1 + temp) / 2
            temp = temp ** power_for_memory_formation
            temp = temp * param_fixed.Learner.learning_rate_for_memory_formation_updates

        param_fixed.storedmemory.control_slope_vs_context = (
            param_fixed.storedmemory.control_slope_vs_context
            + temp * (-g_gradient_for_memory_update)
        )

        g_gradient_for_memory_update_store.append(g_gradient_for_memory_update)
        error_from_memory_store.append(
            error_from_memory_compute(
                param_fixed.storedmemory.control_slope_vs_context,
                p_input_now_considered_good,
                param_fixed,
                context_now,
            )
        )

        t_start = t_end
        state_var0_model_now = state_var0_model_next

        v_a, v_b = get_treadmill_speed(t_start, param_fixed.imposed_foot_speeds)
        context_now = np.array([v_a, v_b])
        context_now = context_now + param_fixed.noise_energy_sensory * np.random.randn(
            context_now.size
        )

    return np.column_stack(p_input_controller_store_ones_tried)