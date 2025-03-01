import numpy as np

def kneeGivenBodyFootScalar(yBody, zBody, yFoot, zFoot):
    """
    Calculate knee position given body and foot position (scalar version)
    """
    yMid = (yBody + yFoot) / 2
    zMid = (zBody + zFoot) / 2
    
    legLengthNow = np.sqrt((yBody - yFoot)**2 + (zBody - zFoot)**2)
    legLengthNominal = 1
    
    # Make sure the value is within the valid range for arccos (-1 to 1)
    cos_value = np.clip(legLengthNow / legLengthNominal, -1.0, 1.0)
    alpha = np.arccos(cos_value)
    alpha = abs(alpha)
    
    R = np.array([
        [np.cos(alpha), -np.sin(alpha)], 
        [np.sin(alpha), np.cos(alpha)]
    ])
    
    vec = np.array([yMid - yBody, zMid - zBody])
    temp = np.array([yBody, zBody]) + R @ vec
    
    yKnee = temp[0]
    zKnee = temp[1]
    
    return yKnee, zKnee

def kneeGivenBodyFoot(yBodyList, zBodyList, yFootList, zFootList):
    """
    Calculate knee positions given body and foot positions (vectorized)
    """
    yKneeList = np.zeros_like(yBodyList)
    zKneeList = np.zeros_like(zBodyList)
    
    for jFrame in range(len(yBodyList)):
        yKneeList[jFrame], zKneeList[jFrame] = kneeGivenBodyFootScalar(
            yBodyList[jFrame], zBodyList[jFrame],
            yFootList[jFrame], zFootList[jFrame])
    
    return yKneeList, zKneeList

# checked and not essential.
# this is just for visualization purposes suppressed in this version.