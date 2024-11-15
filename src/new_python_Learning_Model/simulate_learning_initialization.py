import numpy as np

from f_objective_asymmetric_nominal_8d_to_10d import f_objective_asymmetric_nominal_8d_to_10d 
from compute_model_dynamics_rls_no_singular import compute_model_dynamics_rls_no_singular
from compute_energy_dynamics_model_rls_no_singular import compute_energy_dynamics_model_rls_no_singular
from get_treadmill_speed import get_treadmill_speed 
from compute_steady_energy_gradient_v2_no_singular import compute_steady_energy_gradient_v2_no_singular
from gradient_of_error_from_memory_compute import gradient_of_error_from_memory_compute
from error_from_memory_compute import error_from_memory_compute


def simulate_learning_initialization(param_fixed, p_input_controller_asymmetric_nominal, state_var0_model, context_now, param_controller_gains):
    """
    Initialize key variables and parameters for simulating the learning step by step.

    Args:
        param_fixed (dict): Fixed parameters for the simulation.
        p_input_controller_asymmetric_nominal (list): Nominal learnable parameters for the input controller.
        state_var0_model (list): Initial state variables for the model.
        context_now (list): Current context for gait.
        param_controller_gains (dict): Controller gains parameters.
    """
    print(f"Simulating Learning While Walking ... ({param_fixed['numIterations']} strides)")

    # Key parameters
    noise_std = param_fixed['Learner']['noiseSTDExploratory']
    num_iterations = param_fixed['numIterations']
    include_internal_model = param_fixed['Learner']['includeInternalModel']

    # Adjust learnable parameters (remove last two variables of odd step, which are zeros)
    p_input_now = p_input_controller_asymmetric_nominal[:3] + p_input_controller_asymmetric_nominal[5:]
    num_learning_dimensions = len(p_input_now)
    num_state_dimensions = len(state_var0_model)

    # Learning rates
    alpha_energy_learning_rate = param_fixed['Learner']['LearningRate']

    # Initial time
    t_start = 0

    # Initial internal model
    a_dynamics_model_now = np.zeros((num_state_dimensions - 1, num_learning_dimensions + num_state_dimensions))
    a_dynamics_energy_now = np.zeros((1, num_learning_dimensions + num_state_dimensions))

    # Initialize gradient
    g_energy_gradient_now = np.zeros(len(p_input_now))

    # Initialize various empty arrays
    p_input_controller_store_ones_tried = []
    state_var_now_store = []
    state_var_next_store = []
    edot_store = []
    p_input_controller_store_ones_considered_good = []
    gradient_energy_estimate_store = []
    a_dynamics_model_store = []
    a_dynamics_energy_store = []

    error_old_dynamics_model_store = []
    error_new_dynamics_model_store = []
    error_old_energy_model_store = []
    error_new_energy_model_store = []
    learning_on_or_not_store = []
    memory_to_gradient_direction_cosine_store = []

    g_gradient_for_memory_update_store = []
    error_from_memory_store = []
    p_input_memory_now_store = []

    # Number of strides to use for gradient estimation
    num_strides_to_use = param_fixed['Learner']['numStepsToUseForEstimator']
    state_var0_model_now = np.array(state_var0_model)
    p_input_now_considered_good = np.array(p_input_now)

    # Loop over all strides
    for i_stride in range(1, num_iterations + 1):
        if i_stride % 50 == 0:
            print(f"iStride = {i_stride}")

        # Exploratory noise
        delta_p_input_now1_noise = noise_std * np.random.randn(num_learning_dimensions)

        # Compute the gradient step
        delta_p_input_now2_gradient = alpha_energy_learning_rate * (-g_energy_gradient_now)

        # Restrict gradient step size using trust region, if applicable
        if param_fixed['Learner']['shouldWeUseTrustRegion']:
            if np.linalg.norm(delta_p_input_now2_gradient) > param_fixed['Learner']['trustRegionSize'] * np.sqrt(num_learning_dimensions):
                delta_p_input_now2_gradient /= np.linalg.norm(delta_p_input_now2_gradient)
                delta_p_input_now2_gradient *= param_fixed['Learner']['trustRegionSize'] * np.sqrt(num_learning_dimensions)

        # Compute memory to move toward
        context_now = np.array(context_now)  # Ensure context_now is an array
        nominal_context = np.array(param_fixed['storedmemory']['nominalContext'])
        p_input_memory_now = param_fixed['storedmemory']['nominalControl'] + \
            np.dot(param_fixed['storedmemory']['controlSlopeVsContext'], (context_now - nominal_context))
        p_input_memory_now_store.append(p_input_memory_now)

        # Direction toward memory
        dir_toward_memory = p_input_memory_now - p_input_now_considered_good
        g_energy_gradient_norm = np.linalg.norm(g_energy_gradient_now)
        dir_toward_memory_norm = np.linalg.norm(dir_toward_memory)

        if dir_toward_memory_norm == 0 or g_energy_gradient_norm == 0:
            temp = 0
        else:
            temp = np.dot(dir_toward_memory, -g_energy_gradient_now) / (dir_toward_memory_norm * g_energy_gradient_norm)
        memory_to_gradient_direction_cosine_store.append(temp)

        power_move_to_memory = param_fixed['Learner']['powerToTheMoveToMemory']
        if np.isnan(temp):
            temp = 1 * param_fixed['Learner']['LearningRateTowardMemory']
        else:
            temp = (1 + temp) / 2
            temp = temp ** power_move_to_memory
            temp *= param_fixed['Learner']['LearningRateTowardMemory']

        # Compute step toward memory
        delta_p_input_now3_toward_memory = temp * dir_toward_memory

        # Add step toward memory
        p_input_now_considered_good = p_input_now_considered_good + delta_p_input_now3_toward_memory

        # Add gradient and noise steps
        p_input_now_considered_good += delta_p_input_now2_gradient
        p_input_now_to_try = p_input_now_considered_good + delta_p_input_now1_noise



        # Simulate walking
        edot_now, state_var0_model_next, t_end = f_objective_asymmetric_nominal_8d_to_10d(
            p_input_now_to_try, state_var0_model_now, param_controller_gains, param_fixed, t_start
        )

        # Multiplicative measurement noise in energy estimates
        edot_now *= (1 + np.random.randn() * param_fixed['noiseEnergySensory'])

        # Store all the data so far
        state_var_now_store.append(state_var0_model_now)
        state_var_next_store.append(state_var0_model_next)
        p_input_controller_store_ones_tried.append(p_input_now_to_try)
        edot_store.append(edot_now)
        p_input_controller_store_ones_considered_good.append(p_input_now_considered_good)

        # Update internal model dynamics (linear)
        if i_stride > num_strides_to_use:
            a_dynamics_model_now, error_old_dynamics_model, error_new_dynamics_model = compute_model_dynamics_rls_no_singular(
                a_dynamics_model_now, state_var_now_store, state_var_next_store, p_input_controller_store_ones_tried, num_strides_to_use
            )

        # Update energy model dynamics
        if i_stride > num_strides_to_use:
            a_dynamics_energy_now, error_old_energy_model, error_new_energy_model = compute_energy_dynamics_model_rls_no_singular(
                a_dynamics_energy_now, state_var_now_store, edot_store, p_input_controller_store_ones_tried, num_strides_to_use
            )

        # Compute the error in the linear models
        if i_stride > num_strides_to_use:
            error_old_dynamics_model_store.append(error_old_dynamics_model)
            error_new_dynamics_model_store.append(error_new_dynamics_model)
            error_old_energy_model_store.append(error_old_energy_model)
            error_new_energy_model_store.append(error_new_energy_model)

        # Get the gradient of the steady state energy
        g_energy_gradient_now = compute_steady_energy_gradient_v2_no_singular(
            a_dynamics_model_now, a_dynamics_energy_now, p_input_now_to_try,
            num_learning_dimensions, num_state_dimensions, include_internal_model
        )
        gradient_energy_estimate_store.append(g_energy_gradient_now)
        a_dynamics_model_store.append(a_dynamics_model_now)
        a_dynamics_energy_store.append(a_dynamics_energy_now)

        # Implement prediction error thresholding
        if param_fixed['Learner']['shouldWeThresholdPredictionError'] and i_stride > num_strides_to_use:
            prediction_error_list_now = [error_old_energy_model, error_new_energy_model]
            if any(abs(err) > param_fixed['Learner']['predictionErrorThreshold'] for err in prediction_error_list_now):
                g_energy_gradient_now = np.zeros_like(g_energy_gradient_now)
                learning_on_or_not_store.append(0)
            else:
                learning_on_or_not_store.append(1)

        # Reset for the next step
        t_start = t_end
        state_var0_model_now = state_var0_model_next

        # Update the memory
        g_gradient_for_memory_update = gradient_of_error_from_memory_compute(
            param_fixed['storedmemory']['controlSlopeVsContext'],
            p_input_now_considered_good, param_fixed, context_now
        )

        # Direction from current memory prediction to current controller experienced
        n_memorytocurrentcontroller = -dir_toward_memory / np.linalg.norm(dir_toward_memory)
        n_v_controller = delta_p_input_now2_gradient + delta_p_input_now3_toward_memory
        n_v_controller /= np.linalg.norm(n_v_controller)

        # Cosine of angle between memory->current and v_currentController
        dot_memory_to_current_v_current = np.dot(n_memorytocurrentcontroller, n_v_controller)

        # Modified cosine tuning: scaling how memory should move
        temp = dot_memory_to_current_v_current
        power_for_memory_formation = param_fixed['Learner']['powerToTheMemoryFormation']
        if np.isnan(temp):
            temp = 0
        else:
            temp = (1 + temp) / 2
            temp = temp ** power_for_memory_formation
            temp *= param_fixed['Learner']['LearningRateForMemoryFormationUpdates']

        # Update the memory
        param_fixed['storedmemory']['controlSlopeVsContext'] += temp * (-g_gradient_for_memory_update)

        g_gradient_for_memory_update_store.append(g_gradient_for_memory_update)
        error_from_memory_store.append(
            error_from_memory_compute(param_fixed['storedmemory']['controlSlopeVsContext'], p_input_now_considered_good, param_fixed, context_now)
        )

        # Update the memory context
        v_a, v_b = get_treadmill_speed(t_start, param_fixed['imposedFootSpeeds'])
        context_now = np.array([v_a, v_b])

        # Add some noise in the context
        context_now += param_fixed['noiseEnergySensory'] * np.random.randn(len(context_now))

        return p_input_controller_store_ones_tried
    
