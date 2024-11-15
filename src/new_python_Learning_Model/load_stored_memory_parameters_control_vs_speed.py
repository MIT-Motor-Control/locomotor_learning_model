import numpy as np

def load_stored_memory_parameters_control_vs_speed(param_fixed):
    """
    Load parameters related to stored memory based on nominal conditions.
    
    Args:
        param_fixed (dict): Dictionary containing existing fixed parameters.
        
    Returns:
        dict: Updated dictionary with stored memory parameters.
    """
    # Memory stored based on nominal
    if param_fixed['swingCost']['Coeff'] == 0.9:
        param_fixed['storedmemory'] = {
            'nominalControl': [
                0.328221262798818,
                0.310751796902254,
                0.153556843539029,
                0.328221491356562,
                0.310751694570805,
                0.153557221281688,
                -0.000000038953735,
                -0.000000038953735
            ],
            'nominalContext': [-0.35, -0.35]
        }

    # How memory changes with context: slope
    param_fixed['storedmemory']['controlSlopeVsContext'] = np.zeros((8, 2))  # Start with a blank slate for memory accumulation

    return param_fixed
