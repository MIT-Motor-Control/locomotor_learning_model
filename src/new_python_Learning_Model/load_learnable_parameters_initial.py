def load_learnable_parameters_initial(param_fixed):
    """
    Load the learnable parameters for asymmetric nominal conditions.

    Args:
        param_fixed (dict): Dictionary containing existing fixed parameters.

    Returns:
        list: List of learnable parameters for the input controller.
    """
    # Evaluate the objective function, asymmetric nominal
    if param_fixed['swingCost']['Coeff'] == 0.9:
        p_input_controller_asymmetric_nominal = [
            0.328221262798818,
            0.310751796902254,
            0.153556843539029,
            0,
            0,
            0.328221491356562,
            0.310751694570805,
            0.153557221281688,
            -0.000000038953735,
            -0.000000038953735
        ]
    else:
        p_input_controller_asymmetric_nominal = []  # Default if no match

    return p_input_controller_asymmetric_nominal