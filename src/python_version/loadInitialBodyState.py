import numpy as np

def loadInitialBodyState(pInputControllerAsymmetricNominal):
    """
    Initial conditions for the simulation
    NOTE: these are MID-STANCE initial conditions
    """
    
    vSwingInitial = 0.35  # initial swing speed - shouldn't matter much, ideally equal to the nominal velocity?
    
    stateVar0_Model = np.array([
        0,                               # angleTheta0 = stance leg angle
        pInputControllerAsymmetricNominal[1],  # dAngleTheta0 = stance leg angular rate
        0,                               # yFoot0 = 0; # in lab frame
        0,                               # sum of yFoot in lab frame (integral feedback for station keeping)
        vSwingInitial                    # vSwing 
    ])
    
    return stateVar0_Model  # checked and essential