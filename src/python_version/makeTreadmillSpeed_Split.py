import numpy as np
import matplotlib.pyplot as plt

def makeTreadmillSpeed_Split(paramFixed):
    """
    Create split-belt treadmill speed profiles with precisely controlled transitions
    """
    L = 0.95
    g = 9.81
    timeScaling = np.sqrt(L/g)
    
    # Exactly match the specified speeds
    vNormal = -0.3   # Normal speed (absolute value is 0.3)
    vFast = -0.49    # Fast belt speed (absolute value is 0.49)
    vSlow = -0.1636  # Slow belt speed (absolute value is 0.1636)
    
    # Create exact time points for transitions
    tList = np.array([0, 1156, 9798, 9920, 12500])
    
    # Set speeds for each phase
    footSpeed1List = np.array([
        vNormal,  # Start at normal
        vFast,    # First transition (split condition)
        vFast,    # Stay fast
        vNormal,  # Back to normal
        vNormal   # End at normal
    ])
    
    footSpeed2List = np.array([
        vNormal,  # Start at normal
        vSlow,    # First transition (split condition)
        vSlow,    # Stay slow
        vNormal,  # Back to normal
        vNormal   # End at normal
    ])
    
    # Add short transitions to avoid discontinuities
    tListWithTransitions = []
    footSpeed1WithTransitions = []
    footSpeed2WithTransitions = []
    
    # Short transition time to create sharp but not instantaneous changes
    transitionTime = 10
    
    for i in range(len(tList) - 1):
        if i > 0:  # Add transition points
            # Add point just before transition
            tListWithTransitions.append(tList[i] - transitionTime/2)
            footSpeed1WithTransitions.append(footSpeed1List[i-1])
            footSpeed2WithTransitions.append(footSpeed2List[i-1])
            
            # Add point just after transition
            tListWithTransitions.append(tList[i] + transitionTime/2)
            footSpeed1WithTransitions.append(footSpeed1List[i])
            footSpeed2WithTransitions.append(footSpeed2List[i])
        else:
            # First point
            tListWithTransitions.append(tList[i])
            footSpeed1WithTransitions.append(footSpeed1List[i])
            footSpeed2WithTransitions.append(footSpeed2List[i])
    
    # Add final point
    tListWithTransitions.append(tList[-1])
    footSpeed1WithTransitions.append(footSpeed1List[-1])
    footSpeed2WithTransitions.append(footSpeed2List[-1])
    
    # Convert to numpy arrays
    tListWithTransitions = np.array(tListWithTransitions)
    footSpeed1WithTransitions = np.array(footSpeed1WithTransitions)
    footSpeed2WithTransitions = np.array(footSpeed2WithTransitions)
    
    # Calculate accelerations
    a1List = np.zeros_like(tListWithTransitions)
    a2List = np.zeros_like(tListWithTransitions)
    
    # Finite-difference for non-edge points
    for i in range(1, len(tListWithTransitions)-1):
        dt_left = tListWithTransitions[i] - tListWithTransitions[i-1]
        dt_right = tListWithTransitions[i+1] - tListWithTransitions[i]
        
        # Central difference for acceleration
        if dt_left > 0 and dt_right > 0:
            a1List[i] = (footSpeed1WithTransitions[i+1] - footSpeed1WithTransitions[i-1]) / (dt_left + dt_right)
            a2List[i] = (footSpeed2WithTransitions[i+1] - footSpeed2WithTransitions[i-1]) / (dt_left + dt_right)
    
    # Forward difference for first point
    if len(tListWithTransitions) > 1:
        dt = tListWithTransitions[1] - tListWithTransitions[0]
        if dt > 0:
            a1List[0] = (footSpeed1WithTransitions[1] - footSpeed1WithTransitions[0]) / dt
            a2List[0] = (footSpeed2WithTransitions[1] - footSpeed2WithTransitions[0]) / dt
    
    # Backward difference for last point
    if len(tListWithTransitions) > 1:
        dt = tListWithTransitions[-1] - tListWithTransitions[-2]
        if dt > 0:
            a1List[-1] = (footSpeed1WithTransitions[-1] - footSpeed1WithTransitions[-2]) / dt
            a2List[-1] = (footSpeed2WithTransitions[-1] - footSpeed2WithTransitions[-2]) / dt
    
    # Create return structure
    beltSpeedsImposed = {}
    beltSpeedsImposed['tList'] = tListWithTransitions
    beltSpeedsImposed['footSpeed1List'] = footSpeed1WithTransitions
    beltSpeedsImposed['footSpeed2List'] = footSpeed2WithTransitions
    beltSpeedsImposed['footAcc1List'] = a1List
    beltSpeedsImposed['footAcc2List'] = a2List
    
    # Plot the speed profiles for verification
    plt.figure(2555)
    plt.plot(tListWithTransitions, np.abs(footSpeed1WithTransitions), linewidth=2, label='Fast belt')
    plt.plot(tListWithTransitions, np.abs(footSpeed2WithTransitions), linewidth=2, label='Slow belt')
    plt.xlabel('t')
    plt.ylabel('treadmill speeds (non-dimensional)')
    plt.legend()
    plt.ylim([0, 0.6])
    plt.title('Split belt speed change protocol')
    
    return beltSpeedsImposed