import numpy as np
import scipy.linalg as la

def computeSteadyEnergyGradientV2_NoSingular(AdynamicsModelNow, AdynamicsEnergyNow,
                                           pInputNow_ToTry, numLearningDimensions,
                                           numStateDimensions, includeInternalModelOrNot):
    """
    Compute the steady-state energy gradient with respect to controller parameters
    
    This function computes the gradient of steady-state energy consumption
    with respect to the controller parameters, using the identified dynamics
    and energy models.
    """
    if includeInternalModelOrNot == 0:
        # Return zero gradient if internal model is disabled
        return np.zeros(numLearningDimensions)
    
    # Extract the state-to-state transition matrix (A) and control-to-state matrix (B)
    # from the dynamics model
    A_matrix = AdynamicsModelNow[:, :numStateDimensions-1]
    B_matrix = AdynamicsModelNow[:, numStateDimensions-1:]
    
    # Extract energy model coefficients
    C_state = AdynamicsEnergyNow[:, :numStateDimensions-1]
    D_control = AdynamicsEnergyNow[:, numStateDimensions-1:]
    
    # Check if the system has a stable steady state
    eigvals = la.eigvals(A_matrix)
    if np.any(np.abs(eigvals) >= 1.0):
        # System is unstable or marginally stable, can't compute reliable gradient
        return np.zeros(numLearningDimensions)
    
    # Compute steady state response to a unit step in each control parameter
    # (I - A)^-1 * B gives the DC gain of the system
    I = np.eye(A_matrix.shape[0])
    try:
        dc_gain = la.solve(I - A_matrix, B_matrix)
    except:
        # Numerical issues, use a regularized version
        dc_gain = la.solve(I - A_matrix + 1e-6 * I, B_matrix)
    
    # The gradient of steady-state energy with respect to control is:
    # C * dc_gain + D
    gradient = (C_state @ dc_gain + D_control).flatten()
    
    # Ensure the gradient has the expected dimension (numLearningDimensions)
    if len(gradient) != numLearningDimensions:
        # In case of dimension mismatch, reshape or resize the gradient
        if len(gradient) > numLearningDimensions:
            gradient = gradient[:numLearningDimensions]
        else:
            # Pad with zeros if gradient is too short
            gradient = np.pad(gradient, (0, numLearningDimensions - len(gradient)))
    
    # Scale the gradient to avoid too large steps
    gradient_norm = np.linalg.norm(gradient)
    if gradient_norm > 1.0:
        gradient = gradient / gradient_norm
    
    return gradient