import numpy as np

def computeModelDynamicsModelRLS_NoSingular(AdynamicsModelNow, stateVarNowStore,
                                          stateVarNextStore, pInputControllerStore_OnesTried,
                                          numStridesToUse):
    """
    Compute the dynamics model using Recursive Least Squares
    
    This function uses RLS to identify the dynamics model that maps:
    [stateNow; controlNow] -> stateNext
    """
    # Get recent data
    stateVarNow = stateVarNowStore[:, -numStridesToUse:]
    stateVarNext = stateVarNextStore[:, -numStridesToUse:]
    pInputController = pInputControllerStore_OnesTried[:, -numStridesToUse:]
    
    # Number of state and input dimensions
    num_state_dims = stateVarNow.shape[0]
    num_input_dims = pInputController.shape[0]
    num_samples = stateVarNow.shape[1]
    
    # Create augmented input [state; control]
    X = np.vstack((stateVarNow, pInputController))
    
    # Output is next state (without the last dimension to avoid singularity)
    Y = stateVarNext[:-1, :]
    
    # We need the current model prediction to compute the old error
    Y_pred_old = AdynamicsModelNow @ X
    errorOldDynamicsModel = np.mean(np.sum((Y - Y_pred_old)**2, axis=0))
    
    # Forgetting factor for RLS (gives more weight to recent data)
    lambda_factor = 0.97
    
    # RLS update
    # Initialize P matrix (covariance-like matrix)
    if np.all(AdynamicsModelNow == 0):
        # For initial (zero) model, use a default P matrix with large diagonal values
        P = np.eye(X.shape[0]) * 1000
    else:
        # For subsequent updates, use a smaller P matrix
        P = np.eye(X.shape[0]) * 10
    
    # Update A matrix for each sample (sequential RLS)
    A = AdynamicsModelNow.copy()
    
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
    errorNewDynamicsModel = np.mean(np.sum((Y - Y_pred_new)**2, axis=0))
    
    return A, errorOldDynamicsModel, errorNewDynamicsModel