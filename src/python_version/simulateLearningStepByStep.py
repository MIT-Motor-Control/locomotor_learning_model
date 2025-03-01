import numpy as np
from fObjective_AsymmetricNominal_8DTo10D import fObjective_AsymmetricNominal_8DTo10D
from computeModelDynamicsModelRLS_NoSingular import computeModelDynamicsModelRLS_NoSingular
from computeEnergyDynamicsModelRLS_NoSingular import computeEnergyDynamicsModelRLS_NoSingular
from computeSteadyEnergyGradientV2_NoSingular import computeSteadyEnergyGradientV2_NoSingular
from gradientOfErrorFromMemoryCompute import gradientOfErrorFromMemoryCompute
from errorFromMemoryCompute import errorFromMemoryCompute
from getTreadmillSpeed import getTreadmillSpeed

def simulateLearningStepByStep(paramFixed, pInputControllerAsymmetricNominal, 
                              stateVar0_Model, contextNow, paramControllerGains):
    """
    Main function to simulate learning while walking, step by step
    """
    
    print(f"Simulating Learning While Walking ... ({paramFixed['numIterations']} strides)")
    
    ## key paramaters
    noiseSTD = paramFixed['Learner']['noiseSTDExploratory']
    numIterations = paramFixed['numIterations']
    includeInternalModelOrNot = paramFixed['Learner']['includeInternalModel']
    
    ##
    pInputControllerAsymmetricNominal = pInputControllerAsymmetricNominal[np.r_[0:3, 5:10]]
    # because the last two variablees of the odd step are zero without loss of
    # generality (origin setting)
    numLearningDimensions = len(pInputControllerAsymmetricNominal)
    numStateDimensions = len(stateVar0_Model)
    
    ## learning rates
    alphaEnergyLearningRate = paramFixed['Learner']['LearningRate']
    
    ## do the gradient descent
    
    # learning parameter input
    pInputNow = pInputControllerAsymmetricNominal.copy()
    
    # state initial condition
    # stateVar0_Model = stateVar0_Model
    
    # initial time
    tStart = 0  # re-setting this doesn't matter here
    
    ## initial internal model
    AdynamicsModelNow = np.zeros((numStateDimensions-1, numLearningDimensions+numStateDimensions))
    # no +1 for the constant term and -1 to avoid singularity?
    
    AdynamicsEnergyNow = np.zeros((1, numLearningDimensions+numStateDimensions))
    # just on output
    
    ## initializing gradient
    gEnergyGradientNow = np.zeros(pInputControllerAsymmetricNominal.shape)  # initial zero
    
    ## initializing various empty arrays
    pInputControllerStore_OnesTried = np.empty((numLearningDimensions, 0))
    stateVarNowStore = np.empty((numStateDimensions, 0))
    stateVarNextStore = np.empty((numStateDimensions, 0))
    EdotStore = np.array([])
    pInputControllerStore_OnesConsideredGood = np.empty((numLearningDimensions, 0))
    gradientEnergyEstimateStore = {}
    AdynamicsModelStore = {}
    AdynamicsEnergyStore = {}
    
    errorOldDynamicsModelStore = np.array([])
    errorNewDynamicsModelStore = np.array([])
    errorOldEnergyModelStore = np.array([])
    errorNewEnergyModelStore = np.array([])
    learningOnOrNotStore = np.array([])
    memoryToGradientDirectionCosineStore = np.zeros(numIterations)
    
    gGradientForMemoryUpdateStore = {}
    errorFromMemoryStore = np.array([])
    pInputMemoryNowStore = np.empty((numLearningDimensions, 0))
    
    ## copying some variables
    numStridesToUse = paramFixed['Learner']['numStepsToUseForEstimator']
    stateVar0_ModelNow = stateVar0_Model.copy()
    pInputNow_ConsideredGood = pInputNow.copy()
    
    ## looping over all strides
    for iStride in range(numIterations):  # 1 stride = 1 iteration
        
        if (iStride+1) % 50 == 0:
            print(f"iStride = {iStride+1}")  # display stride 
        
        ## exploratory noise
        delta_pInputNow1_Noise = noiseSTD * np.random.randn(numLearningDimensions)
        
        # for iDim in range(numLearningDimensions):  # making sure that the noise is not too small or too large
        #     a = 0.2*noiseSTD
        #     b = 2*noiseSTD
        #     if abs(delta_pInputNow1_Noise[iDim]) > a or abs(delta_pInputNow1_Noise[iDim]) < b:
        #         delta_pInputNow1_Noise[iDim] = np.sign(delta_pInputNow1_Noise[iDim])*(a+(b-a)*np.random.rand())
        #         # some uniform between a and b
        
        ## compute the gradient step
        # Ensure gradient has right dimension
        if len(gEnergyGradientNow) != numLearningDimensions:
            if len(gEnergyGradientNow) > numLearningDimensions:
                gEnergyGradientNow = gEnergyGradientNow[:numLearningDimensions]
            else:
                gEnergyGradientNow = np.pad(gEnergyGradientNow, (0, numLearningDimensions - len(gEnergyGradientNow)))
                
        delta_pInputNow2_Gradient = alphaEnergyLearningRate * (-gEnergyGradientNow)
        
        ## restriction gradient step size if using trust region
        if paramFixed['Learner']['shouldWeUseTrustRegion']:
            gradient_norm = np.linalg.norm(delta_pInputNow2_Gradient)
            if gradient_norm > paramFixed['Learner']['trustRegionSize'] * np.sqrt(numLearningDimensions):
                delta_pInputNow2_Gradient = delta_pInputNow2_Gradient / gradient_norm
                delta_pInputNow2_Gradient = delta_pInputNow2_Gradient * paramFixed['Learner']['trustRegionSize'] * np.sqrt(numLearningDimensions)
        
        ## use the linear memory model to get a memory to move toward
        # could be nonlinear, with a piecewise relu kind of network
        memory_prediction = paramFixed['storedmemory']['controlSlopeVsContext'] @ (contextNow - paramFixed['storedmemory']['nominalContext'])
        
        # Ensure memory prediction has right dimension
        if len(memory_prediction) != numLearningDimensions:
            if len(memory_prediction) > numLearningDimensions:
                memory_prediction = memory_prediction[:numLearningDimensions]
            else:
                memory_prediction = np.pad(memory_prediction, (0, numLearningDimensions - len(memory_prediction)))
        
        # Ensure nominal control has right dimension 
        nominal_control = paramFixed['storedmemory']['nominalControl']
        if len(nominal_control) != numLearningDimensions:
            if len(nominal_control) > numLearningDimensions:
                nominal_control = nominal_control[:numLearningDimensions]
            else:
                nominal_control = np.pad(nominal_control, (0, numLearningDimensions - len(nominal_control)))
        
        pInputMemoryNow = nominal_control + memory_prediction
        pInputMemoryNowStore = np.column_stack((pInputMemoryNowStore, pInputMemoryNow)) if pInputMemoryNowStore.size else pInputMemoryNow.reshape(-1, 1)
        
        ## take a step toward memory
        dirTowardMemory = (pInputMemoryNow - pInputNow_ConsideredGood)
        
        # modified cosine tuning: compute direction cosine between memory and gradient
        gradient_norm = np.linalg.norm(gEnergyGradientNow)
        dir_norm = np.linalg.norm(dirTowardMemory)
        
        if dir_norm > 0 and gradient_norm > 0:
            temp = np.dot(dirTowardMemory, -gEnergyGradientNow) / (dir_norm * gradient_norm)
        else:
            temp = np.nan
            
        memoryToGradientDirectionCosineStore[iStride] = temp
        
        powerMovetoMemory = paramFixed['Learner']['powerToTheMoveToMemory']
        
        if np.isnan(temp):  # this happens when we have divide by zero for the gradient say
            temp = 1 * paramFixed['Learner']['LearningRateTowardMemory']  # when gradient is bad, use memory <>
        else:
            temp = (1 + temp) / 2  # needs to be 1 when angle is zero and zero when 180 or -180 degrees
            temp = temp ** powerMovetoMemory  # square it to penalize opposing the gradient much more
            temp = temp * paramFixed['Learner']['LearningRateTowardMemory']  # adding a cosine law for the learning rate toward memory
        
        # compute step toward memory
        delta_pInputNow3_TowardMemory = temp * dirTowardMemory
        
        # add step toward memory
        pInputNow_ConsideredGood = pInputNow_ConsideredGood + delta_pInputNow3_TowardMemory  # step towards the stored memory
        
        ## add gradient and noise steps
        pInputNow_ConsideredGood = pInputNow_ConsideredGood + delta_pInputNow2_Gradient
        pInputNow_ToTry = pInputNow_ConsideredGood + delta_pInputNow1_Noise
        
        ## simulate walking
        f_objective, stateVar0_ModelNext, tEnd, f_energy, f_constraint = fObjective_AsymmetricNominal_8DTo10D(
            pInputNow_ToTry, stateVar0_ModelNow, 
            paramControllerGains, paramFixed, tStart)
            
        # f_energy is the EdotNow we need
        EdotNow = f_energy
        
        ## multiplicative measurement noise in energy estimates / measurements
        EdotNow = EdotNow * (1 + np.random.randn() * paramFixed['noiseEnergySensory'])
        
        ## store all the data so far
        stateVarNowStore = np.column_stack((stateVarNowStore, stateVar0_ModelNow)) if stateVarNowStore.size else stateVar0_ModelNow.reshape(-1, 1)
        stateVarNextStore = np.column_stack((stateVarNextStore, stateVar0_ModelNext)) if stateVarNextStore.size else stateVar0_ModelNext.reshape(-1, 1)
        pInputControllerStore_OnesTried = np.column_stack((pInputControllerStore_OnesTried, pInputNow_ToTry)) if pInputControllerStore_OnesTried.size else pInputNow_ToTry.reshape(-1, 1)
        EdotStore = np.append(EdotStore, EdotNow)
        pInputControllerStore_OnesConsideredGood = np.column_stack((pInputControllerStore_OnesConsideredGood, pInputNow_ConsideredGood)) if pInputControllerStore_OnesConsideredGood.size else pInputNow_ConsideredGood.reshape(-1, 1)
        
        ## update internal model dynamics (linear)
        if iStride > numStridesToUse:
            AdynamicsModelNow, errorOldDynamicsModel, errorNewDynamicsModel = computeModelDynamicsModelRLS_NoSingular(
                AdynamicsModelNow, stateVarNowStore,
                stateVarNextStore, pInputControllerStore_OnesTried, numStridesToUse)
        
        ## update energy model dynamics
        if iStride > numStridesToUse:
            AdynamicsEnergyNow, errorOldEnergyModel, errorNewEnergyModel = computeEnergyDynamicsModelRLS_NoSingular(
                AdynamicsEnergyNow, stateVarNowStore, EdotStore, 
                pInputControllerStore_OnesTried, numStridesToUse)
        
        ## compute the error in the linear models
        if iStride > numStridesToUse:
            # error in the Dynamics model
            errorOldDynamicsModelStore = np.append(errorOldDynamicsModelStore, errorOldDynamicsModel)
            errorNewDynamicsModelStore = np.append(errorNewDynamicsModelStore, errorNewDynamicsModel)
            
            # error in the Energy model
            errorOldEnergyModelStore = np.append(errorOldEnergyModelStore, errorOldEnergyModel)
            errorNewEnergyModelStore = np.append(errorNewEnergyModelStore, errorNewEnergyModel)
        
        ## get the gradient of the steady state energy    
        gEnergyGradientNow = computeSteadyEnergyGradientV2_NoSingular(
            AdynamicsModelNow, AdynamicsEnergyNow, pInputNow_ToTry,
            numLearningDimensions, numStateDimensions, includeInternalModelOrNot)
        
        gradientEnergyEstimateStore[iStride] = gEnergyGradientNow
        AdynamicsModelStore[iStride] = AdynamicsModelNow
        AdynamicsEnergyStore[iStride] = AdynamicsEnergyNow
        
        ## implements prediction error thresholding. 
        if paramFixed['Learner']['shouldWeThresholdPredictionError']:
            if iStride > numStridesToUse:
                # predictionErrorListNow = np.concatenate([
                #     errorOldDynamicsModel, errorNewDynamicsModel, 
                #     errorOldEnergyModel, errorNewEnergyModel])  # error control on both old and new
                
                predictionErrorListNow = np.concatenate([errorOldEnergyModel, errorNewEnergyModel])  # just with the new model
                
                if np.any(np.abs(predictionErrorListNow) > paramFixed['Learner']['predictionErrorThreshold']):
                    gEnergyGradientNow = 0 * gEnergyGradientNow
                    learningOnOrNotStore = np.append(learningOnOrNotStore, 0)
                    # if the gradient has high error, do not take the gradient step 
                else:
                    learningOnOrNotStore = np.append(learningOnOrNotStore, 1)
        
        ## reset for the next step
        tStart = tEnd
        stateVar0_ModelNow = stateVar0_ModelNext
        
        ## update the memory
        # the gradient that moves the memory toward the current experience
        gGradientForMemoryUpdate = gradientOfErrorFromMemoryCompute(
            paramFixed['storedmemory']['controlSlopeVsContext'],
            pInputNow_ConsideredGood, paramFixed, contextNow)
            
        # Make sure the gradient has the right shape for updating the slope matrix
        expected_shape = paramFixed['storedmemory']['controlSlopeVsContext'].shape
        if gGradientForMemoryUpdate.shape != expected_shape:
            # Reshape or recreate gradient with correct shape
            if len(gGradientForMemoryUpdate.flatten()) >= expected_shape[0] * expected_shape[1]:
                # Reshape if we have enough elements
                gGradientForMemoryUpdate = gGradientForMemoryUpdate.flatten()[:expected_shape[0] * expected_shape[1]].reshape(expected_shape)
            else:
                # Create a new gradient with zeros otherwise
                temp_gradient = np.zeros(expected_shape)
                # Fill what we can from the original gradient
                flat_gradient = gGradientForMemoryUpdate.flatten()
                temp_gradient.flat[:len(flat_gradient)] = flat_gradient
                gGradientForMemoryUpdate = temp_gradient
        
        # direction from current memory prediction to current controller experienced
        n_memorytocurrentcontroller = -dirTowardMemory  # dirTowardMemory is the direction from current controller to memory prediction
        n_memorytocurrentcontroller_norm = np.linalg.norm(n_memorytocurrentcontroller)
        
        if n_memorytocurrentcontroller_norm > 0:
            n_memorytocurrentcontroller = n_memorytocurrentcontroller / n_memorytocurrentcontroller_norm
        
            # current direction that the controller is moving toward
            n_vController = delta_pInputNow2_Gradient + delta_pInputNow3_TowardMemory
            n_vController_norm = np.linalg.norm(n_vController)
            
            if n_vController_norm > 0:
                n_vController = n_vController / n_vController_norm
                
                # cosine of angle between memory->current and v_currentController
                dot_MemoryToCurrent_vCurrent = np.dot(n_memorytocurrentcontroller, n_vController)  # this is the directionj cosine already
                
                # modified cosine tuning: scaling how memory should move
                temp = dot_MemoryToCurrent_vCurrent
                powerForMemoryFormation = paramFixed['Learner']['powerToTheMemoryFormation']
                
                if np.isnan(temp):  # this happens when we have divide by zero for the gradient say
                    temp = 0
                else:
                    temp = (1 + temp) / 2  # needs to be 1 when angle is zero and zero when 180 or -180 degrees
                    temp = temp ** powerForMemoryFormation  # square it to penalize opposing the gradient much more
                    temp = temp * paramFixed['Learner']['LearningRateForMemoryFormationUpdates']  # adding a cosine law for the learning rate toward memory
                
                # update the memory
                paramFixed['storedmemory']['controlSlopeVsContext'] = \
                    paramFixed['storedmemory']['controlSlopeVsContext'] + \
                    temp * (-gGradientForMemoryUpdate)
            else:
                temp = 0
        else:
            temp = 0
        
        gGradientForMemoryUpdateStore[iStride] = gGradientForMemoryUpdate
        errorFromMemoryStore = np.append(
            errorFromMemoryStore, 
            errorFromMemoryCompute(
                paramFixed['storedmemory']['controlSlopeVsContext'],
                pInputNow_ConsideredGood, paramFixed, contextNow))
        
        # update the memory context 
        vA, vB = getTreadmillSpeed(tStart, paramFixed['imposedFootSpeeds'])
        contextNow = np.array([vA, vB])
        
        ## add some noise in the context
        contextNow = contextNow + paramFixed['noiseEnergySensory'] * np.random.randn(2)
    
    return pInputControllerStore_OnesTried  # essential and checked.