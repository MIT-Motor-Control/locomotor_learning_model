import numpy as np
from simulateOneStep_MidstanceToMidstance_WithEnergy import simulateOneStep_MidstanceToMidstance_WithEnergy

def simulateManySteps_AsymmetricControl(stateVar0, paramController, paramFixed, tStart):
    """
    Simulate multiple steps with asymmetric control
    
    reduced state at mid-stance is:
    angleTheta0, dAngleTheta0, yFoot (lab frame), INTy (lab frame)
    """
    
    # Initialize storage variables
    tStore = [[] for _ in range(paramFixed['numSteps'] + 1)]
    stateStore = [[] for _ in range(paramFixed['numSteps'] + 1)]
    EmetTotalStore = np.zeros(paramFixed['numSteps'])
    EmetPerTimeStore = np.zeros(paramFixed['numSteps'])
    tStepStore = np.zeros(paramFixed['numSteps'])
    
    EworkPushoffStore = np.zeros(paramFixed['numSteps'])
    EworkHeelstrikeStore = np.zeros(paramFixed['numSteps'])
    
    ##
    for iStep in range(paramFixed['numSteps']):
        
        ## odd or even control
        if (iStep + 1) % 2 == 0:  # even step (adding 1 because Python is 0-indexed)
            paramController['theta_end_nominal'] = paramController['Even']['theta_end_nominal']
            paramController['ydot_atMidstance_nominal_beltframe'] = paramController['Even']['ydot_atMidstance_nominal_beltframe']
            paramController['PushoffImpulseMagnitude_nominal'] = paramController['Even']['PushoffImpulseMagnitude_nominal']
            paramController['y_atMidstance_nominal_slopeframe'] = paramController['Even']['y_atMidstance_nominal_slopeframe']
            paramController['SUMy_atMidstance_nominal_slopeframe'] = paramController['Even']['SUMy_atMidstance_nominal_slopeframe']
        else:  # odd step
            paramController['theta_end_nominal'] = paramController['Odd']['theta_end_nominal']
            paramController['ydot_atMidstance_nominal_beltframe'] = paramController['Odd']['ydot_atMidstance_nominal_beltframe']
            paramController['PushoffImpulseMagnitude_nominal'] = paramController['Odd']['PushoffImpulseMagnitude_nominal']
            paramController['y_atMidstance_nominal_slopeframe'] = 0  # fixed to zero by fiat
            paramController['SUMy_atMidstance_nominal_slopeframe'] = 0  # fixed to zero by fiat
        
        ## simulate the first step, belt 1
        results = simulateOneStep_MidstanceToMidstance_WithEnergy(
            stateVar0, iStep+1, tStart, paramController, paramFixed)
        
        stateVar0 = results[0]
        tlistTillEndstance = results[1]
        statevarlistTillEndstance = results[2]
        tlistTillMidstance = results[3]
        statevarlistMidstance = results[4]
        Emet_totalNow = results[5]
        Emet_perTime = results[6]
        tTotal = results[7]
        EworkPushoff = results[8]
        EworkHeelstrike = results[9]
        
        ## storing everything
        # the following weirdness is because a midstance to midstance
        # simulation straddles two steps
        
        tStore[iStep] = tlistTillEndstance if len(tStore[iStep]) == 0 else np.append(tStore[iStep], tlistTillEndstance)
        stateStore[iStep] = statevarlistTillEndstance if len(stateStore[iStep]) == 0 else np.vstack((stateStore[iStep], statevarlistTillEndstance))
        
        if iStep < paramFixed['numSteps'] - 1:
            tStore[iStep+1] = tlistTillMidstance[:-1] if len(tStore[iStep+1]) == 0 else np.append(tStore[iStep+1], tlistTillMidstance[:-1])
            stateStore[iStep+1] = statevarlistMidstance[:-1] if len(stateStore[iStep+1]) == 0 else np.vstack((stateStore[iStep+1], statevarlistMidstance[:-1]))
        else:
            tStore[iStep+1] = tlistTillMidstance if len(tStore[iStep+1]) == 0 else np.append(tStore[iStep+1], tlistTillMidstance)
            stateStore[iStep+1] = statevarlistMidstance if len(stateStore[iStep+1]) == 0 else np.vstack((stateStore[iStep+1], statevarlistMidstance))
        
        ## reset time, very important do not comment!
        tStart = tlistTillMidstance[-1]
        
        ## metabolic cost estimate
        EmetTotalStore[iStep] = Emet_totalNow
        EmetPerTimeStore[iStep] = Emet_perTime
        tStepStore[iStep] = tTotal
        
        EworkPushoffStore[iStep] = EworkPushoff
        EworkHeelstrikeStore[iStep] = EworkHeelstrike
    
    return stateVar0, tStore, stateStore, EmetTotalStore, EmetPerTimeStore, tStepStore, EworkPushoffStore, EworkHeelstrikeStore  # essential