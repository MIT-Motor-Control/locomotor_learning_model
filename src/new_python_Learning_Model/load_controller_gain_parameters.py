def load_controller_gain_parameters(param_fixed):
    """
    Load parameters related to the feedback controller gains that keep the biped stable.
    
    Args:
        param_fixed (dict): Dictionary containing existing fixed parameters, including swing cost coefficient.
        
    Returns:
        dict: Dictionary containing controller gain parameters.
    """
    param_controller = {}
    
    # Controller parameters, gains, fixed during optimization
    if param_fixed['swingCost']['Coeff'] == 0.9:
        param_controller['pushoff_gain_ydot'] = -0.521088310893437
        param_controller['legAngle_gain_ydot'] = 0.279308152405233
        param_controller['pushoff_gain_y'] = -0.085939152240409
        param_controller['legAngle_gain_y'] = -0.028999999999996
        param_controller['pushoff_gain_SUMy'] = -0.007302228127922
        param_controller['legAngle_gain_SUMy'] = 0.002019507100848
        param_controller['legAngle_gain_BeltSpeed'] = -0.075044622818653
        param_controller['pushoff_gain_BeltSpeed'] = -1.055930819923483
    
    # Additional parameter
    param_controller['ControllerBeltFrameVsLabFrame'] = 1  # Default value
    
    return param_controller
