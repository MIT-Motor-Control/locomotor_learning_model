import numpy as np

def loadStoredMemoryParameters_ControlVsSpeed(paramFixed):
    """
    Load stored memory parameters for control vs speed
    """
    
    ## memory stored based on nominal
    paramFixed['storedmemory'] = {}
    
    if paramFixed['swingCost']['Coeff'] == 0.9:
        paramFixed['storedmemory']['nominalControl'] = np.array([
            0.328221262798818,
            0.310751796902254,
            0.153556843539029,
            0.328221491356562,
            0.310751694570805,
            0.153557221281688,
            -0.000000038953735,
            -0.000000038953735
        ])
    
    paramFixed['storedmemory']['nominalContext'] = np.array([-0.35, -0.35])
    
    ## how memory changes with context: slope
    # learningParameterListLength = length(paramFixed.storedmemory.nominalControl)
    
    paramFixed['storedmemory']['controlSlopeVsContext'] = np.zeros((8, 2))  # start with a blank slate for memory accumulation
    
    return paramFixed