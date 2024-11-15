import numpy as np

def swing_cost_doke(t_duration_of_change, v_initial, v_final, v_body, param_fixed):
    """
    Approximates the leg swing cost as in Doke and Kuo 2005, which posits a cost related to force rate.

    Args:
        t_duration_of_change (float): Duration of the change.
        v_initial (float): Initial velocity.
        v_final (float): Final velocity.
        v_body (float): Body velocity.
        param_fixed (dict): Fixed parameters for the simulation.

    Returns:
        float: The calculated swing cost.
    """
    # Calculate relative velocities
    v_initial_relative = v_initial - v_body
    v_final_relative = v_final - v_body

    # Change in velocity and force rate
    delta_v = v_final_relative - v_initial_relative
    force = param_fixed['mFoot'] * delta_v / t_duration_of_change
    force_rate = force / t_duration_of_change

    # Small constant to avoid singularity
    epsilon = 0.01

    # Calculate swing cost
    c = param_fixed['swingCost']['Coeff'] * (np.sqrt(force_rate ** 2 + epsilon ** 2) ** param_fixed['swingCost']['alpha'])

    return c
