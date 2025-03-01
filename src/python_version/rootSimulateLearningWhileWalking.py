import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import random
import os
import sys

# Import all the necessary functions
from loadBipedModelParameters import loadBipedModelParameters
from loadSensoryNoiseParameters import loadSensoryNoiseParameters
from loadControllerGainParameters import loadControllerGainParameters
from loadLearnerParameters import loadLearnerParameters
from loadProtocolParameters import loadProtocolParameters
from loadStoredMemoryParameters_ControlVsSpeed import loadStoredMemoryParameters_ControlVsSpeed
from loadLearnableParametersInitial import loadLearnableParametersInitial
from loadInitialBodyState import loadInitialBodyState
from getTreadmillSpeed import getTreadmillSpeed
from simulateLearningStepByStep import simulateLearningStepByStep
from postProcessAfterLearning import postProcessAfterLearning

# Clear figures
plt.close('all')

# This is the main program in this folder. Run this program, and it calls
# whatever functions are necessary to simulate learning while walking.

# -------------------------------------------------------------------------
# This initial section defines the various model parameters. See Methods
# for how these parameters are obtained or fixed.
# -------------------------------------------------------------------------

# warning('Learning and memory has been turned off right now')

## initialize empty parameter values
paramFixed = {}

## initialize the random number generator
np.random.seed(None)  # Equivalent to rng('shuffle','twister')

## biped model parameters
paramFixed = loadBipedModelParameters(paramFixed)

## sensory noise parameters
paramFixed = loadSensoryNoiseParameters(paramFixed)

## controller parameters
paramControllerGains = loadControllerGainParameters(paramFixed)
       
## set the integral controller to zero?
# paramController.pushoff_gain_SUMy =  0
# paramController.legAngle_gain_SUMy =  0

## learner parameters
paramFixed = loadLearnerParameters(paramFixed)

## adaptation protocol 
paramFixed = loadProtocolParameters(paramFixed)

## initial stored memory, default control
paramFixed = loadStoredMemoryParameters_ControlVsSpeed(paramFixed)

## how long to simulate parameters
# paramFixed = loadHowLongParameters(paramFixed)
# We get the simulation duration from the protocol, but you can override 
# this by uncommenting loadHowLongParameters.m function above and changing 
# the values in loadHowLongParameters

## load current learnable parameters
# look inside the function to what controller parameters are tuned by the learning algorithm
pInputControllerAsymmetricNominal = loadLearnableParametersInitial(paramFixed)

## load initial state and time
stateVar0_Model = loadInitialBodyState(pInputControllerAsymmetricNominal)
tStart = 0

# storing initial state for later use
stateVar0_Model_BeforeLearning = stateVar0_Model.copy()

## performance or objective function calculation. Just a demo, not necessary to do here.
# f = fObjective_AsymmetricNominal(pInputControllerAsymmetricNominal,stateVar0_Model, 
#     paramControllerGains,paramFixed,tStart) # function computes the
#     objective or performance function for one stride

## context for the gait
# TO DO: get this from 
vA, vB = getTreadmillSpeed(0, paramFixed['imposedFootSpeeds'])
contextNow = np.array([vA, vB])
contextLength = len(contextNow)

## Simulate learning step by step
pInputControllerStore_OnesTried = simulateLearningStepByStep(
    paramFixed, pInputControllerAsymmetricNominal, 
    stateVar0_Model, contextNow, paramControllerGains)

## convert the 8D back up to 10D to use the old functions
pInputControllerStore_8D = pInputControllerStore_OnesTried
pInputControllerStore_10D = np.vstack([
    pInputControllerStore_8D[0:3, :],
    np.zeros((2, pInputControllerStore_8D.shape[1])),
    pInputControllerStore_8D[3:8, :]
])

## post-process outputs and make some plots
doAnimate = 0  # leave on 0. changing this to 1 will not do anything in this version.
postProcessAfterLearning(
    pInputControllerStore_10D,
    stateVar0_Model_BeforeLearning,
    paramControllerGains, paramFixed, doAnimate)

if __name__ == "__main__":
    print("Simulation complete")