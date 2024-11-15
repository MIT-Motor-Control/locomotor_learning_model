import numpy as np
from simulate_one_step_midstance_to_midstance_with_energy import simulate_one_step_midstance_to_midstance_with_energy

def simulate_many_steps_asymmetric_control(state_var0, param_controller, param_fixed, t_start):
    """
    Simulate many steps for asymmetric control.

    Args:
        state_var0 (ndarray): Initial state variables for the model.
        param_controller (dict): Controller parameters.
        param_fixed (dict): Fixed parameters for the simulation.
        t_start (float): Start time of the current step.

    Returns:
        tuple: Various stores including state variables, energy estimates, and time steps.
    """
    # Initialize storage
    t_store = [None] * (param_fixed['numSteps'] + 1)
    state_store = [None] * (param_fixed['numSteps'] + 1)
    emet_total_store = np.zeros(param_fixed['numSteps'])
    emet_per_time_store = np.zeros(param_fixed['numSteps'])
    t_step_store = np.zeros(param_fixed['numSteps'])

    ework_pushoff_store = np.zeros(param_fixed['numSteps'])
    ework_heelstrike_store = np.zeros(param_fixed['numSteps'])

    # Loop over all steps
    for i_step in range(1, param_fixed['numSteps'] + 1):
        # Odd or even control
        if i_step % 2 == 0:
            param_controller['theta_end_nominal'] = param_controller['Even']['theta_end_nominal']
            param_controller['ydot_atMidstance_nominal_beltframe'] = param_controller['Even']['ydot_atMidstance_nominal_beltframe']
            param_controller['PushoffImpulseMagnitude_nominal'] = param_controller['Even']['PushoffImpulseMagnitude_nominal']
            param_controller['y_atMidstance_nominal_slopeframe'] = param_controller['Even']['y_atMidstance_nominal_slopeframe']
            param_controller['SUMy_atMidstance_nominal_slopeframe'] = param_controller['Even']['SUMy_atMidstance_nominal_slopeframe']
        else:
            param_controller['theta_end_nominal'] = param_controller['Odd']['theta_end_nominal']
            param_controller['ydot_atMidstance_nominal_beltframe'] = param_controller['Odd']['ydot_atMidstance_nominal_beltframe']
            param_controller['PushoffImpulseMagnitude_nominal'] = param_controller['Odd']['PushoffImpulseMagnitude_nominal']
            param_controller['y_atMidstance_nominal_slopeframe'] = 0  # Fixed to zero by fiat
            param_controller['SUMy_atMidstance_nominal_slopeframe'] = 0  # Fixed to zero by fiat

        # Simulate one step
        (
            state_var0,
            tlist_till_endstance,
            statevarlist_till_endstance,
            tlist_till_midstance,
            statevarlist_midstance,
            emet_total_now,
            emet_per_time,
            t_total,
            ework_pushoff,
            ework_heelstrike,
        ) = simulate_one_step_midstance_to_midstance_with_energy(
            state_var0, i_step, t_start, param_controller, param_fixed
        )

        # Store data
        if t_store[i_step - 1] is None:
            t_store[i_step - 1] = tlist_till_endstance
        else:
            t_store[i_step - 1] = np.concatenate((t_store[i_step - 1], tlist_till_endstance))

        if state_store[i_step - 1] is None:
            state_store[i_step - 1] = statevarlist_till_endstance
        else:
            state_store[i_step - 1] = np.concatenate((state_store[i_step - 1], statevarlist_till_endstance), axis=0)

        if i_step < param_fixed['numSteps']:
            t_store[i_step] = tlist_till_midstance[:-1]
            state_store[i_step] = statevarlist_midstance[:-1, :]
        else:
            t_store[i_step] = tlist_till_midstance
            state_store[i_step] = statevarlist_midstance

        # Reset time
        t_start = tlist_till_midstance[-1]

        # Metabolic cost estimate
        emet_total_store[i_step - 1] = emet_total_now
        emet_per_time_store[i_step - 1] = emet_per_time
        t_step_store[i_step - 1] = t_total

        ework_pushoff_store[i_step - 1] = ework_pushoff
        ework_heelstrike_store[i_step - 1] = ework_heelstrike

    return (
        state_var0,
        t_store,
        state_store,
        emet_total_store,
        emet_per_time_store,
        t_step_store,
        ework_pushoff_store,
        ework_heelstrike_store,
    )

