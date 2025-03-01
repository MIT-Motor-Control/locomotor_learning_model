import numpy as np
from simulateManySteps_AsymmetricControl import simulateManySteps_AsymmetricControl

def fObjective_AsymmetricNominal(pInputControllerNominal, stateVar0_Model,
                                paramController, paramFixed, tStart):
    """
    Objective function for asymmetric nominal controller
    """
    
    ## unwrap the controller
    if 'Odd' not in paramController:
        paramController['Odd'] = {}
    if 'Even' not in paramController:
        paramController['Even'] = {}
        
    paramController['Odd']['theta_end_nominal'] = pInputControllerNominal[0]
    paramController['Odd']['ydot_atMidstance_nominal_beltframe'] = pInputControllerNominal[1]
    paramController['Odd']['PushoffImpulseMagnitude_nominal'] = pInputControllerNominal[2]
    paramController['Odd']['y_atMidstance_nominal_slopeframe'] = pInputControllerNominal[3]  # needs to be fixed to zero
    paramController['Odd']['SUMy_atMidstance_nominal_slopeframe'] = pInputControllerNominal[4]  # needs to be fixed to zero
    
    paramController['Even']['theta_end_nominal'] = pInputControllerNominal[5]
    paramController['Even']['ydot_atMidstance_nominal_beltframe'] = pInputControllerNominal[6]
    paramController['Even']['PushoffImpulseMagnitude_nominal'] = pInputControllerNominal[7]
    paramController['Even']['y_atMidstance_nominal_slopeframe'] = pInputControllerNominal[8]
    paramController['Even']['SUMy_atMidstance_nominal_slopeframe'] = pInputControllerNominal[9]
    
    ## simulate 4 steps, so that we have some reasonable average to go off of,
    # just to be safe
    paramFixed['numSteps'] = paramFixed['Learner']['numStepsPerIteration']
    stateVar0, tStore, stateStore, EmetTotalStore, EmetPerTimeStore, tStepStore, EworkPushoffStore, EworkHeelstrikeStore = \
        simulateManySteps_AsymmetricControl(stateVar0_Model, paramController, paramFixed, tStart)
    
    f_energy = np.sum(EmetTotalStore) / np.sum(tStepStore)
    tEnd = tStore[-1][-1]
    
    f_constraint = (np.linalg.norm(stateVar0_Model[1:3] - stateVar0[1:3])) ** 2  # just thetaDot and yFoot is targeted to be zeroed
    
    # trading off energy and periodicity: seems not essential
    lambdaEnergyVsPeriodicity = paramFixed['lambdaEnergyVsPeriodicity']
    f_objective = lambdaEnergyVsPeriodicity * f_energy + (1 - lambdaEnergyVsPeriodicity) * f_constraint
    
    angleThetaEnd_1 = stateStore[0][-1, 0]  # Last row, first column of first step
    angleThetaEnd_2 = stateStore[1][-1, 0]  # Last row, first column of second step
    f_symmetry = (angleThetaEnd_1 - angleThetaEnd_2) ** 2
    
    # trading off energy and symmetry
    f_objective = paramFixed['lambdaEnergyVsSymmetry'] * f_objective + \
        (1 - paramFixed['lambdaEnergyVsSymmetry']) * f_symmetry * paramFixed['symmetryMultiplier']
    
    return f_objective, stateVar0, tEnd, f_energy, f_constraint  # checked and essential