import numpy as np

def errorFromMemoryCompute(controlSlopeVsContext, pInputNow_ConsideredGood,
                          paramFixed, contextNow):
    """
    Compute the error between the controller parameters and the memory prediction
    
    This function evaluates how well the current memory model predicts
    the current controller parameters for the given context.
    """
    # Predict controller parameters based on memory model
    pInputMemoryNow = paramFixed['storedmemory']['nominalControl'] + \
        controlSlopeVsContext @ (contextNow - paramFixed['storedmemory']['nominalContext'])
    
    # Compute the squared error between memory prediction and actual controller
    error = np.mean((pInputMemoryNow - pInputNow_ConsideredGood)**2)
    
    return error