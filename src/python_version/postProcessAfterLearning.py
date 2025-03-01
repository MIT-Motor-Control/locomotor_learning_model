import numpy as np
from simulateManySteps_AsymmetricControl import simulateManySteps_AsymmetricControl
from postProcessHelper_JustThePlots import postProcessHelper_JustThePlots

def postProcessAfterLearning(pInputStore, stateVar0_Model,
                            paramController, paramFixed, doAnimate):
    """
    Post-process the learning results and generate plots
    """
    
    numIterations = pInputStore.shape[1]
    
    ##
    tStore = []
    stateStore = []
    EmetTotalStore = []
    EmetPerTimeStore = []
    tStepStore = []
    EworkPushoffStore = []
    EworkHeelstrikeStore = []
    
    tStart = 0  # just the first time
    
    print('Post-processing all the walking data ...')
    
    # Initialize arrays for storing averages
    EdotStore_IterationAverage = np.zeros(numIterations)
    tTotalIterationStore = np.zeros(numIterations)
    
    ##
    for iStride in range(numIterations):
        
        if (iStride + 1) % 50 == 0:
            print(f"iStride = {iStride + 1}")
        
        ## get the stored current learned controller parameters
        pInputControllerAsymmetricNominal = pInputStore[:, iStride]
        
        ##
        if 'Odd' not in paramController:
            paramController['Odd'] = {}
        if 'Even' not in paramController:
            paramController['Even'] = {}
            
        paramController['Odd']['theta_end_nominal'] = pInputControllerAsymmetricNominal[0]
        paramController['Odd']['ydot_atMidstance_nominal_beltframe'] = pInputControllerAsymmetricNominal[1]
        paramController['Odd']['PushoffImpulseMagnitude_nominal'] = pInputControllerAsymmetricNominal[2]
        paramController['Odd']['y_atMidstance_nominal_slopeframe'] = pInputControllerAsymmetricNominal[3]  # needs to be fixed to zero, is it?
        paramController['Odd']['SUMy_atMidstance_nominal_slopeframe'] = pInputControllerAsymmetricNominal[4]  # needs to be fixed to zero, is it?
        
        paramController['Even']['theta_end_nominal'] = pInputControllerAsymmetricNominal[5]
        paramController['Even']['ydot_atMidstance_nominal_beltframe'] = pInputControllerAsymmetricNominal[6]
        paramController['Even']['PushoffImpulseMagnitude_nominal'] = pInputControllerAsymmetricNominal[7]
        paramController['Even']['y_atMidstance_nominal_slopeframe'] = pInputControllerAsymmetricNominal[8]
        paramController['Even']['SUMy_atMidstance_nominal_slopeframe'] = pInputControllerAsymmetricNominal[9]
        
        ##
        paramFixed['numSteps'] = paramFixed['Learner']['numStepsPerIteration']
        result = simulateManySteps_AsymmetricControl(stateVar0_Model, paramController, paramFixed, tStart)
        
        stateVar0_Model = result[0]
        tStore_Now = result[1]
        stateStore_Now = result[2]
        EmetTotalStore_Now = result[3]
        EmetPerTimeStore_Now = result[4]
        tStepStore_Now = result[5]
        EworkPushoffStore_Now = result[6]
        EworkHeelstrikeStore_Now = result[7]
        
        ## reset things
        tStart = tStore_Now[-1][-1]
        
        ##
        EdotStore_IterationAverage[iStride] = np.sum(EmetTotalStore_Now) / np.sum(tStepStore_Now)
        tTotalIterationStore[iStride] = np.sum(tStepStore_Now)
        
        ## assemble all the Store variables
        if iStride == 0:
            tStore = tStore_Now.copy()
            stateStore = stateStore_Now.copy()
            EmetTotalStore = EmetTotalStore_Now.copy()
            EmetPerTimeStore = EmetPerTimeStore_Now.copy()
            tStepStore = tStepStore_Now.copy()  # this is step time midstance to midstance, not useful
            
            EworkPushoffStore = EworkPushoffStore_Now.copy()
            EworkHeelstrikeStore = EworkHeelstrikeStore_Now.copy()
        
        else:
            # to merge the half steps at the end and the beginning to compile
            # things for steps defined as heel strike to heel strike
            tStore[-1] = np.append(tStore[-1], tStore_Now[0])
            stateStore[-1] = np.vstack((stateStore[-1], stateStore_Now[0])) if len(stateStore[-1]) > 0 else stateStore_Now[0]
            
            # the other steps
            tStore.extend(tStore_Now[1:])
            stateStore.extend(stateStore_Now[1:])
            
            # the rest are just arrays from midstance to midstance
            EmetTotalStore = np.append(EmetTotalStore, EmetTotalStore_Now)
            EmetPerTimeStore = np.append(EmetPerTimeStore, EmetPerTimeStore_Now)
            tStepStore = np.append(tStepStore, tStepStore_Now)
            
            EworkPushoffStore = np.append(EworkPushoffStore, EworkPushoffStore_Now)
            EworkHeelstrikeStore = np.append(EworkHeelstrikeStore, EworkHeelstrikeStore_Now)
    
    ## save if you wish to not run the whole simulation 
    # np.save('EverythingNeededForJustThePlots.npy', {
    #     'stateVar0_Model': stateVar0_Model,
    #     'tStore': tStore,
    #     'stateStore': stateStore,
    #     'EmetTotalStore': EmetTotalStore,
    #     'EmetPerTimeStore': EmetPerTimeStore,
    #     'tStepStore': tStepStore,
    #     'paramController': paramController,
    #     'paramFixed': paramFixed,
    #     'doAnimate': doAnimate,
    #     'EworkPushoffStore': EworkPushoffStore,
    #     'EworkHeelstrikeStore': EworkHeelstrikeStore,
    #     'EdotStore_IterationAverage': EdotStore_IterationAverage,
    #     'tTotalIterationStore': tTotalIterationStore
    # })
    
    ##
    postProcessHelper_JustThePlots(stateVar0_Model, tStore, stateStore,
        EmetTotalStore, EmetPerTimeStore, tStepStore, paramController, paramFixed, doAnimate, 
        EworkPushoffStore, EworkHeelstrikeStore, EdotStore_IterationAverage, tTotalIterationStore)