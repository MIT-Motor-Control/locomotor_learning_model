import numpy as np

def computeEnergyDynamicsModelRLS_NoSingular(AdynamicsEnergyNow, stateVarNowStore,
                                           EdotStore, pInputControllerStore_OnesTried,
                                           numStridesToUse):
    """
    Compute the energy dynamics model using Recursive Least Squares
    
    This function uses RLS to identify the model that maps:
    [stateNow; controlNow] -> energyRateNow
    """
    # Get recent data
    stateVarNow = stateVarNowStore[:, -numStridesToUse:]
    EdotNow = EdotStore[-numStridesToUse:]
    pInputController = pInputControllerStore_OnesTried[:, -numStridesToUse:]
    
    # Reshape Edot to a row vector
    EdotNow = EdotNow.reshape(1, -1)
    
    # Number of state and input dimensions
    num_state_dims = stateVarNow.shape[0]
    num_input_dims = pInputController.shape[0]
    num_samples = stateVarNow.shape[1]
    
    # Create augmented input [state; control]
    X = np.vstack((stateVarNow, pInputController))
    
    # Output is energy rate
    Y = EdotNow
    
    # We need the current model prediction to compute the old error
    Y_pred_old = AdynamicsEnergyNow @ X
    errorOldEnergyModel = np.mean((Y - Y_pred_old)**2)
    
    # Forgetting factor for RLS (gives more weight to recent data)
    lambda_factor = 0.95  # More aggressive forgetting for energy model
    
    # RLS update
    # Initialize P matrix (covariance-like matrix)
    if np.all(AdynamicsEnergyNow == 0):
        # For initial (zero) model, use a default P matrix with large diagonal values
        P = np.eye(X.shape[0]) * 1000
    else:
        # For subsequent updates, use a smaller P matrix
        P = np.eye(X.shape[0]) * 10
    
    # Update A matrix for each sample (sequential RLS)
    A = AdynamicsEnergyNow.copy()
    
    for t in range(num_samples):
        x_t = X[:, t:t+1]  # Input at time t
        y_t = Y[:, t:t+1]  # Output at time t
        
        # Predict
        y_hat_t = A @ x_t
        
        # Error
        e_t = y_t - y_hat_t
        
        # Gain
        k_t = P @ x_t / (lambda_factor + x_t.T @ P @ x_t)
        
        # Update model
        A = A + e_t @ k_t.T
        
        # Update P
        P = (P - k_t @ x_t.T @ P) / lambda_factor
    
    # Calculate new prediction error
    Y_pred_new = A @ X
    errorNewEnergyModel = np.mean((Y - Y_pred_new)**2)
    
    return A, errorOldEnergyModel, errorNewEnergyModel