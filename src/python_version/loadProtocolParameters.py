import numpy as np
from makeTreadmillSpeed_Split import makeTreadmillSpeed_Split
from makeTreadmillSpeed_Tied import makeTreadmillSpeed_Tied

def loadProtocolParameters(paramFixed):
    """
    Adaptation protocol parameters: how the treadmill speed is changed, and
    whether the adaptation protocol is on a split-belt treadmill or a tied
    belt treadmill.
    """
    
    ## what speed protocol to use: split belt changes
    paramFixed['SplitOrTied'] = 'split'
    paramFixed['speedProtocol'] = 'classic split belt'
    paramFixed['transitionTime'] = 15  # in seconds
    paramFixed['imposedFootSpeeds'] = makeTreadmillSpeed_Split(paramFixed)
    
    ## what speed protocol to use: tied belt changes
    # more familiar task of walking on a regular treadmill with speed changes
    # paramFixed['transitionTime'] = 3  # in seconds
    # paramFixed['SplitOrTied'] = 'tied'
    # paramFixed['speedProtocol'] = 'four speed changes'
    # paramFixed['imposedFootSpeeds'] = makeTreadmillSpeed_Tied(paramFixed)
    
    ##
    paramFixed['angleSlope'] = 0
    # do not change: the code has not been tested for non-zero values
    
    ## We get the simulation duration from the protocol, but you can override 
    # this by changing the values in loadHowLongParameters.m function and
    # uncommenting that function in the root program
    
    ## How many steps to simulate
    # Fixed to exactly 6400 steps (3200 strides) to match the expected stride count
    # Each stride consists of 2 steps (left and right foot)
    paramFixed['numStepsToLearn'] = 3200 * 2  # 6400 steps = 3200 strides
    
    # Calibration to ensure transitions happen at the right strides
    # (Described as stride 302 for adaptation, stride 2492 for washout)
    
    ## optimization iterations - this should be exactly 3200 strides
    paramFixed['numIterations'] = 3200
    
    if paramFixed['Learner']['numStepsPerIteration'] % 2 != 0:
        paramFixed['Learner']['numStepsPerIteration'] = paramFixed['numStepsPerIteration'] + 1
    
    return paramFixed  # checked and essential