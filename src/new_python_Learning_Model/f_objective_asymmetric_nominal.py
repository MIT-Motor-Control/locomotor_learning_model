import numpy as np
from simulate_many_steps_asymmetric_control import simulate_many_steps_asymmetric_control

def f_objective_asymmetric_nominal(p_input_controller_nominal, state_var0_model, param_controller, param_fixed, t_start):
    """
    Calculate the objective value, updated state variables, end time, energy value, and constraint value for asymmetric nominal control.

    Args:
        p_input_controller_nominal (ndarray): Nominal controller input parameters.
        state_var0_model (ndarray): Initial state variables for the model.
        param_controller (dict): Controller parameters.
        param_fixed (dict): Fixed parameters for the simulation.
        t_start (float): Start time of the current step.

    Returns:
        tuple: Objective value, updated state variables, end time, energy value, and constraint value.
    """

    # Initialize 'Odd' key if it does not exist
    if 'Odd' not in param_controller:
        param_controller['Odd'] = {}
    # Initialize 'Odd' key if it does not exist
    if 'Even' not in param_controller:
        param_controller['Even'] = {}

    # Unwrap the controller
    param_controller['Odd']['theta_end_nominal'] = p_input_controller_nominal[0]
    param_controller['Odd']['ydot_atMidstance_nominal_beltframe'] = p_input_controller_nominal[1]
    param_controller['Odd']['PushoffImpulseMagnitude_nominal'] = p_input_controller_nominal[2]
    param_controller['Odd']['y_atMidstance_nominal_slopeframe'] = p_input_controller_nominal[3]  # Needs to be fixed to zero
    param_controller['Odd']['SUMy_atMidstance_nominal_slopeframe'] = p_input_controller_nominal[4]  # Needs to be fixed to zero

    param_controller['Even']['theta_end_nominal'] = p_input_controller_nominal[5]
    param_controller['Even']['ydot_atMidstance_nominal_beltframe'] = p_input_controller_nominal[6]
    param_controller['Even']['PushoffImpulseMagnitude_nominal'] = p_input_controller_nominal[7]
    param_controller['Even']['y_atMidstance_nominal_slopeframe'] = p_input_controller_nominal[8]
    param_controller['Even']['SUMy_atMidstance_nominal_slopeframe'] = p_input_controller_nominal[9]

    # Simulate 4 steps to have a reasonable average to go off of, just to be safe
    param_fixed['numSteps'] = param_fixed['Learner']['numStepsPerIteration']
    state_var0, t_store, state_store, emet_total_store, emet_per_time_store, t_step_store = simulate_many_steps_asymmetric_control(
        state_var0_model, param_controller, param_fixed, t_start
    )

    f_energy = sum(emet_total_store) / sum(t_step_store)
    t_end = t_store[-1][-1]

    f_constraint = (np.linalg.norm(state_var0_model[1:3] - state_var0[1:3])) ** 2  # Targeting thetaDot and yFoot to be zeroed

    # Trading off energy and periodicity
    lambda_energy_vs_periodicity = param_fixed['lambdaEnergyVsPeriodicity']
    f_objective = lambda_energy_vs_periodicity * f_energy + (1 - lambda_energy_vs_periodicity) * f_constraint

    angle_theta_end_1 = state_store[0][-1, 0]
    angle_theta_end_2 = state_store[1][-1, 0]
    f_symmetry = (angle_theta_end_1 - angle_theta_end_2) ** 2

    # Trading off energy and symmetry
    f_objective = param_fixed['lambdaEnergyVsSymmetry'] * f_objective + \
        (1 - param_fixed['lambdaEnergyVsSymmetry']) * f_symmetry * param_fixed['symmetryMultiplier']

    return f_objective, state_var0, t_end, f_energy, f_constraint

