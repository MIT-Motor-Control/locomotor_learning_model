def load_initial_body_state(p_input_controller_asymmetric_nominal):
    """
    Load the initial conditions for the simulation.
    
    Args:
        p_input_controller_asymmetric_nominal (list): List of learnable parameters for the input controller.
    
    Returns:
        list: List representing the initial state variables for the model.
    """
    # Initial swing speed - shouldn't matter much, ideally equal to the nominal velocity
    v_swing_initial = 0.35

    # Initial conditions for the simulation
    state_var0_model = [
        0,  # angleTheta0 = stance leg angle
        p_input_controller_asymmetric_nominal[1],  # dAngleTheta0 = stance leg angular rate
        0,  # yFoot0 = 0; in lab frame
        0,  # sum of yFoot in lab frame (integral feedback for station keeping)
        v_swing_initial  # vSwing
    ]

    return state_var0_model