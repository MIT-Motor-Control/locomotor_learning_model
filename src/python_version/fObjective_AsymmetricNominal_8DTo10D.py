import numpy as np
from fObjective_AsymmetricNominal import fObjective_AsymmetricNominal

def fObjective_AsymmetricNominal_8DTo10D(pInputControllerNominal, stateVar0_Model,
                                         paramController, paramFixed, tStart):
    """
    Converts 8D input to 10D input and calls fObjective_AsymmetricNominal
    """
    
    temp = np.zeros(10)
    temp[0:3] = pInputControllerNominal[0:3]
    temp[5:10] = pInputControllerNominal[3:8]
    pInputControllerNominal = temp
    
    f_objective, stateVar0, tEnd, f_energy, f_constraint = fObjective_AsymmetricNominal(
        pInputControllerNominal, stateVar0_Model,
        paramController, paramFixed, tStart)
    
    return f_objective, stateVar0, tEnd, f_energy, f_constraint  # checked and essential