import numpy as np

from simulate_ip_until_midstance import simulate_ip_until_midstance
from simulate_ip_until_endstance import simulate_ip_until_endstance
from swing_cost_doke import swing_cost_doke

def simulate_one_step_midstance_to_midstance_with_energy(p_input0, i_step, t_start, param_controller, param_fixed):
    """
    Simulate one step from midstance to midstance with energy calculations.

    Args:
        p_input0 (ndarray): Initial controller input parameters.
        i_step (int): Current step number.
        t_start (float): Start time of the current step.
        param_controller (dict): Controller parameters.
        param_fixed (dict): Fixed parameters for the simulation.

    Returns:
        tuple: Various variables including updated controller input, time lists, state variables, and energy estimates.
    """
    # Update param_controller based on the step (odd/even)
    if i_step % 2 == 0:
        param_controller['tList_BeltSpeed'] = param_fixed['imposedFootSpeeds']['tList']
        param_controller['PushoffFootSpeedNowList'] = param_fixed['imposedFootSpeeds']['footSpeed2List']
        param_controller['HeelstrikeFootSpeedNowList'] = param_fixed['imposedFootSpeeds']['footSpeed1List']
        param_controller['PushoffAccelerationNowList'] = param_fixed['imposedFootSpeeds']['footAcc2List']
    else:
        param_controller['tList_BeltSpeed'] = param_fixed['imposedFootSpeeds']['tList']
        param_controller['PushoffFootSpeedNowList'] = param_fixed['imposedFootSpeeds']['footSpeed1List']
        param_controller['HeelstrikeFootSpeedNowList'] = param_fixed['imposedFootSpeeds']['footSpeed2List']
        param_controller['PushoffAccelerationNowList'] = param_fixed['imposedFootSpeeds']['footAcc1List']

    # Fake variables
    num_points_per_interval = 300
    t_stance = 30
    t_end = t_start + t_stance
    tspan = np.linspace(t_start, t_end, num_points_per_interval)
    tspan_2 = [t_start, t_end]

    # Mid-stance state
    angle_theta0 = p_input0[0]
    angle_theta_dot0 = p_input0[1]
    y_foot0 = p_input0[2]
    sum_y0 = p_input0[3]

    # Additional swing cost parameter
    v_swing_initial_lab_frame = p_input0[4]
    param_controller['PushoffFootSpeedNow'] = np.interp(t_start, param_controller['tList_BeltSpeed'], param_controller['PushoffFootSpeedNowList'])
    v_body_initial_lab_frame = angle_theta_dot0 + param_controller['PushoffFootSpeedNow']

    # Calculations for mid-stance
    y_at_midstance_wrt_foot = param_fixed['leglength'] * np.sin(angle_theta0 + param_fixed['angleSlope'])
    y_at_midstance_slopeframe = y_at_midstance_wrt_foot + y_foot0
    ydot_at_midstance = angle_theta_dot0 * param_fixed['leglength']

    # State error at mid-stance
    delta_ydot_at_midstance_beltframe = ydot_at_midstance - param_controller['ydot_atMidstance_nominal_beltframe']
    delta_y_at_midstance = y_at_midstance_slopeframe - param_controller['y_atMidstance_nominal_slopeframe']
    delta_sum_y = sum_y0 - param_controller['SUMy_atMidstance_nominal_slopeframe']

    # Delta ydot in lab frame
    ydot_at_midstance_lab_frame = ydot_at_midstance + param_controller['PushoffFootSpeedNow']
    v_foot_nominal = -0.35
    param_controller['ydot_atMidstance_nominal_labframe'] = param_controller['ydot_atMidstance_nominal_beltframe'] + v_foot_nominal
    delta_ydot_at_midstance_labframe = ydot_at_midstance_lab_frame - param_controller['ydot_atMidstance_nominal_labframe']

    # Adding sensory noise
    delta_ydot_at_midstance_beltframe += param_fixed['velocitySensoryNoise'] * np.random.randn()

    # Setting up the feedback controller with vision and speed memory
    delta_v_foot_lab_frame = delta_ydot_at_midstance_labframe - delta_ydot_at_midstance_beltframe
    param_controller['theta_end_thisStep'] = param_controller['theta_end_nominal'] + \
        param_controller['legAngle_gain_ydot'] * delta_ydot_at_midstance_beltframe + \
        param_controller['legAngle_gain_y'] * delta_y_at_midstance + \
        param_controller['legAngle_gain_SUMy'] * delta_sum_y + \
        param_controller['legAngle_gain_BeltSpeed'] * delta_v_foot_lab_frame

    if param_controller['theta_end_thisStep'] > 0.95 * np.pi / 4:
        param_controller['theta_end_thisStep'] = 0.95 * np.pi / 4

    # First half step: midstance to endstance
    state_var0 = p_input0[:3]
    tlist_till_endstance, statevarlist_till_endstance = simulate_ip_until_endstance(state_var0, tspan_2, param_fixed, param_controller)

    # Push off calculations
    angle_theta_end = statevarlist_till_endstance[-1, 0]
    d_angle_theta_end = statevarlist_till_endstance[-1, 1]
    unit_vector_along_circle = np.array([
        np.cos(param_controller['theta_end_thisStep'] + param_fixed['angleSlope']),
        -np.sin(param_controller['theta_end_thisStep'] + param_fixed['angleSlope'])
    ])
    v_endstance_body_pushoff_foot_frame = unit_vector_along_circle * d_angle_theta_end * param_fixed['leglength']
    unit_vector_trailing_leg = np.array([
        np.sin(param_controller['theta_end_thisStep'] + param_fixed['angleSlope']),
        np.cos(param_controller['theta_end_thisStep'] + param_fixed['angleSlope'])
    ])

    # Push off feedback calculation
    pushoff_impulse_magnitude_this_step = param_controller['PushoffImpulseMagnitude_nominal'] + \
        param_controller['pushoff_gain_ydot'] * delta_ydot_at_midstance_beltframe + \
        param_controller['pushoff_gain_y'] * delta_y_at_midstance + \
        param_controller['pushoff_gain_SUMy'] * delta_sum_y + \
        param_controller['pushoff_gain_BeltSpeed'] * delta_v_foot_lab_frame

    # In foot frame of reference
    v_after_pushoff_pushoff_foot_frame = v_endstance_body_pushoff_foot_frame + pushoff_impulse_magnitude_this_step * unit_vector_trailing_leg
    
    delta_e_pushoff = 0.5 * param_fixed['mbody'] * np.linalg.norm(v_after_pushoff_pushoff_foot_frame) ** 2 - 0.5 * param_fixed['mbody'] * np.linalg.norm(v_endstance_body_pushoff_foot_frame) ** 2

    # Interpolate to get the correct belt speed
    param_controller['PushoffFootSpeedNow'] = np.interp(tlist_till_endstance[-1], param_controller['tList_BeltSpeed'], param_controller['PushoffFootSpeedNowList'])

    # Converting to lab frame
    v_pushoff_foot_slopeframe = np.array([param_controller['PushoffFootSpeedNow'], 0])
    v_after_pushoff_labframe = v_after_pushoff_pushoff_foot_frame + v_pushoff_foot_slopeframe

    # Heel strike
    param_controller['HeelstrikeFootSpeedNow'] = np.interp(tlist_till_endstance[-1], param_controller['tList_BeltSpeed'], param_controller['HeelstrikeFootSpeedNowList'])
    v_heelstrike_foot_labframe = np.array([param_controller['HeelstrikeFootSpeedNow'], 0])
    v_after_pushoff_heelstrike_foot_frame = v_after_pushoff_labframe - v_heelstrike_foot_labframe

    # Step length and heel strike velocity
    step_length_next_step = 2 * param_fixed['leglength'] * np.sin(param_controller['theta_end_thisStep'] + param_fixed['angleSlope'])
    vector_leading_leg = np.array([-np.sin(param_controller['theta_end_thisStep'] + param_fixed['angleSlope']), np.cos(param_controller['theta_end_thisStep'] + param_fixed['angleSlope'])])
    unit_vector_leading_leg = vector_leading_leg / np.linalg.norm(vector_leading_leg)
    v_after_heelstrike_heelstrike_foot_frame = v_after_pushoff_heelstrike_foot_frame - np.dot(unit_vector_leading_leg, v_after_pushoff_heelstrike_foot_frame) * unit_vector_leading_leg

    delta_e_heelstrike = 0.5 * param_fixed['mbody'] * np.linalg.norm(v_after_heelstrike_heelstrike_foot_frame) ** 2 - 0.5 * param_fixed['mbody'] * np.linalg.norm(v_after_pushoff_heelstrike_foot_frame) ** 2

    # Initialize the second step simulation
    t_start = tlist_till_endstance[-1]
    t_stance = 30
    t_end = t_start + t_stance
    tspan = np.linspace(t_start, t_end, num_points_per_interval)

    # Get the new theta and thetaDot for the next step
    angle_theta0 = -param_controller['theta_end_thisStep'] - 2 * param_fixed['angleSlope']
    d_angle_theta0 = np.linalg.norm(v_after_heelstrike_heelstrike_foot_frame) / param_fixed['leglength']
    y_foot0 = statevarlist_till_endstance[-1, 2] + step_length_next_step

    state_var0 = np.array([angle_theta0, d_angle_theta0, y_foot0])

    # Second half step: endstance to midstance
    tlist_till_midstance, statevarlist_midstance = simulate_ip_until_midstance(state_var0, tspan, param_fixed, param_controller)

    # Update state variable
    state_var0 = statevarlist_midstance[-1, :]

    # Update SUMy
    y_foot0 = state_var0[2]
    sum_y0 += y_foot0 - param_controller['y_atMidstance_nominal_slopeframe']

    # Update controller input
    p_input0_out = np.append(state_var0, sum_y0)

    # Step to step cost
    ework_pushoff = abs(delta_e_pushoff)
    ework_heelstrike = abs(delta_e_heelstrike)
    emet_step2step = param_fixed['bPos'] * ework_pushoff + param_fixed['bNeg'] * ework_heelstrike

    # Swing cost
    thalfstance1 = tlist_till_endstance[-1] - tlist_till_endstance[0]
    thalfstance2 = tlist_till_midstance[-1] - tlist_till_midstance[0]
    v_swing_ato_c = (y_foot0 - statevarlist_till_endstance[0, 2]) / thalfstance1
    v_swing_bto_d = (statevarlist_midstance[-1, 2] - y_foot0) / thalfstance2

    # Remove work cost
    c1, c2, c3 = 0, 0, 0

    # Doke addition
    c1 += swing_cost_doke(thalfstance1, v_swing_initial_lab_frame, v_swing_ato_c, v_body_initial_lab_frame, param_fixed)
    c3 += swing_cost_doke(thalfstance2, param_controller['PushoffFootSpeedNow'], v_swing_bto_d, v_after_pushoff_labframe[0], param_fixed)

    # Total swing energy cost
    emet_swing_total = c1 + c2 + c3

    # Total energy cost
    emet_total = emet_step2step + emet_swing_total
    t_stance1 = tlist_till_endstance[-1] - tlist_till_endstance[0]
    t_stance2 = tlist_till_midstance[-1] - tlist_till_midstance[0]
    emet_per_time = emet_total / (t_stance1 + t_stance2)
    t_total = t_stance1 + t_stance2

    # Update controller input
    p_input0_out = np.append(p_input0_out, v_swing_bto_d)

    return p_input0_out, tlist_till_endstance, statevarlist_till_endstance, tlist_till_midstance, statevarlist_midstance, emet_total, emet_per_time, t_total, ework_pushoff, ework_heelstrike

