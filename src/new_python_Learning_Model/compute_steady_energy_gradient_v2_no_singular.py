import numpy as np

def compute_steady_energy_gradient_v2_no_singular(Adynamics_model, Adynamics_energy, p_input_now, num_learning_dimensions, num_state_dimensions, include_internal_model_or_not):
    """
    Updates the gradient of the steady state energy with respect to the learnable control parameters by extrapolating.

    Args:
        Adynamics_model (ndarray): Dynamics model matrix.
        Adynamics_energy (ndarray): Energy dynamics matrix.
        p_input_now (ndarray): Current input parameters.
        num_learning_dimensions (int): Number of learning dimensions.
        num_state_dimensions (int): Number of state dimensions.
        include_internal_model_or_not (bool): Whether to include the internal model.

    Returns:
        ndarray: Gradient of the steady state energy with respect to control parameters.
    """
    # Adjust state dimensions to avoid singular matrices
    num_state_dimensions -= 1

    # Identity matrix for the current state dimensions
    Inow = np.eye(num_state_dimensions)

    # Extract matrices A and B from the dynamics model
    A = Adynamics_model[:num_state_dimensions, :num_state_dimensions]
    B = Adynamics_model[:num_state_dimensions, num_state_dimensions:num_state_dimensions + num_learning_dimensions]

    # Extract matrices G and H from the energy dynamics
    G = Adynamics_energy[0, :num_state_dimensions]
    H = Adynamics_energy[0, num_state_dimensions:num_state_dimensions + num_learning_dimensions]

    # Compute the gradient of steady state energy
    g_energy = (G @ np.linalg.solve(Inow - A, B) * include_internal_model_or_not + H).T

    return g_energy


