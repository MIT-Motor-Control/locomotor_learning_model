import numpy as np

def loadLearnableParametersInitial(paramFixed):
    """
    What are the learnable parameters
    step - odd
    theta_end_nominal = ?
    ydot_atMidstance_nominal_beltframe = ?
    PushoffImpulseMagnitude_nominal = ?
    y_atMidstance_nominal_slopeframe = 0;
    SUMy_atMidstance_nominal_slopeframe = 0;
    
    step - even
    theta_end_nominal = ?
    ydot_atMidstance_nominal_beltframe = ?
    PushoffImpulseMagnitude_nominal = ?
    y_atMidstance_nominal_slopeframe = ?
    SUMy_atMidstance_nominal_slopeframe = ?
    
    asymmetric nominal means that the nominal for the two steps are not the same
    """
    
    ## evaluate the objective function, asymmetric nominal
    if paramFixed['swingCost']['Coeff'] == 0.9:
        pInputControllerAsymmetricNominal = np.array([
            0.328221262798818,
            0.310751796902254,
            0.153556843539029,
            0,
            0,
            0.328221491356562,
            0.310751694570805,
            0.153557221281688,
            -0.000000038953735,
            -0.000000038953735
        ])
    
    return pInputControllerAsymmetricNominal  # checked and essential