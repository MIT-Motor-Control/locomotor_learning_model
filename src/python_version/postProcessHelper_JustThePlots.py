import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from kneeGivenBodyFoot import kneeGivenBodyFoot
from convertMetToVO2 import convertMetToVO2
from getTreadmillSpeed import getTreadmillSpeed

def postProcessHelper_JustThePlots(stateVar0, tStore, stateStore,
                                 EmetStore, EmetPerTimeStore, tStepStore, 
                                 paramController, paramFixed, doAnimate,
                                 EworkPushoffStore, EworkHeelstrikeStore, 
                                 EdotStore_IterationAverage, tTotalIterationStore):
    """
    This function makes the plots/figures for the split-belt adaptation.
    
    Uses dynamic calculations based on simulation data.
    """
    paramFixed['numSteps'] = len(EmetStore)
    
    # Calculate step lengths and asymmetry
    tStanceList = np.zeros(paramFixed['numSteps'])
    stepLengthList = np.zeros(paramFixed['numSteps'])
    
    for iStep in range(paramFixed['numSteps']):
        if len(tStore[iStep]) > 0:
            tStanceList[iStep] = tStore[iStep][-1] - tStore[iStep][0]
        
        if len(stateStore[iStep]) > 0:
            theta_end = stateStore[iStep][-1, 0]
            stepLengthList[iStep] = abs(2 * paramFixed['leglength'] * np.sin(theta_end))
    
    # Group steps by odd/even for asymmetry calculation
    tStance_fast = tStanceList[0::2]
    tStance_slow = tStanceList[1::2]
    
    # Make sure arrays have same length for element-wise operations
    min_len = min(len(tStance_fast), len(tStance_slow))
    tStance_fast = tStance_fast[:min_len]
    tStance_slow = tStance_slow[:min_len]
    
    # Calculate time asymmetry
    stepTimeAsymmetryList = (tStance_slow - tStance_fast) / (tStance_slow + tStance_fast)
    
    # Calculate step length asymmetry
    stepLength_slow = stepLengthList[0::2]
    stepLength_fast = stepLengthList[1::2]
    
    # Make sure arrays have same length
    stepLength_slow = stepLength_slow[:min_len]
    stepLength_fast = stepLength_fast[:min_len]
    
    # Calculate step length asymmetry
    stepLengthAsymmetryList = (stepLength_slow - stepLength_fast) / (stepLength_fast + stepLength_slow)
    
    # Create stride indices array
    strideCountList = np.arange(1, min_len + 1)
    
    # Calculate metabolic data
    params = {}
    params['tList'] = np.cumsum(tTotalIterationStore)
    params['EmetRateList'] = EdotStore_IterationAverage
    
    # Convert to VO2 and smooth the metabolic rate
    if len(params['tList']) > 1 and len(params['EmetRateList']) > 1:
        tSpan_Smoothed, EmetSList_Smoothed = convertMetToVO2(params)
    else:
        tSpan_Smoothed = params['tList']
        EmetSList_Smoothed = params['EmetRateList']
    
    # Find experiment transition points from treadmill speeds
    # These are critical time points for our experiment
    transitionPointsFound = False
    adaptationStart = 0
    postAdaptationStart = 0
    
    # Calculate total simulation time
    totalTime = np.sum(tTotalIterationStore)
    
    # Generate time points for checking treadmill speed transitions
    checkTimes = np.linspace(0, totalTime, 1000)
    foot1SpeedList, foot2SpeedList = getTreadmillSpeed(checkTimes, paramFixed['imposedFootSpeeds'])
    
    # Find transition points based on when speeds change
    speedDiff = np.abs(foot1SpeedList) - np.abs(foot2SpeedList)
    transitions = np.where(np.abs(np.diff(speedDiff)) > 0.01)[0]
    
    if len(transitions) >= 2:
        transitionPointsFound = True
        adaptationTimePoint = checkTimes[transitions[0]]
        postAdaptationTimePoint = checkTimes[transitions[-1]]
        
        # Find corresponding stride indices
        cumulativeTime = np.cumsum(tTotalIterationStore)
        adaptationStart = np.searchsorted(cumulativeTime, adaptationTimePoint)
        postAdaptationStart = np.searchsorted(cumulativeTime, postAdaptationTimePoint)
    
    if not transitionPointsFound:
        # Fallback to approximations if transitions not found
        if len(strideCountList) >= 300:
            adaptationStart = 300
        else:
            adaptationStart = int(len(strideCountList) * 0.1)
            
        if len(strideCountList) >= 2500:
            postAdaptationStart = 2500
        else:
            postAdaptationStart = int(len(strideCountList) * 0.8)
    
    # Create main figure
    plt.figure(200, figsize=(18, 6))
    plt.suptitle('Split-belt Treadmill: Adaptation', fontsize=16)
    
    # =============================================
    # 1. TREADMILL SPEED SUBPLOT
    # =============================================
    plt.subplot(1, 3, 1)
    
    # Create precise time points for plotting treadmill speeds
    plot_time = np.linspace(0, totalTime, 500)
    foot1SpeedList, foot2SpeedList = getTreadmillSpeed(plot_time, paramFixed['imposedFootSpeeds'])
    
    # Plot treadmill speeds
    plt.plot(plot_time, np.abs(foot1SpeedList), 'b-', linewidth=2, label='Fast belt')
    plt.plot(plot_time, np.abs(foot2SpeedList), 'r-', linewidth=2, label='Slow belt')
    
    # Set axis limits
    maxSpeed = max(np.max(np.abs(foot1SpeedList)), np.max(np.abs(foot2SpeedList)))
    plt.ylim([0, maxSpeed * 1.1])
    plt.xlim([0, totalTime])
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Treadmill belt speeds', fontsize=12)
    plt.legend(fontsize=10)
    
    # =============================================
    # 2. STEP LENGTH SYMMETRY SUBPLOT
    # =============================================
    plt.subplot(1, 3, 2)
    
    # Plot the actual calculated asymmetry
    if len(strideCountList) > 0 and len(stepLengthAsymmetryList) > 0:
        # Apply Savitzky-Golay filter to smooth the data while preserving trends
        if len(stepLengthAsymmetryList) > 11:  # Need enough points for filtering
            stepLengthAsymmetry_smoothed = savgol_filter(stepLengthAsymmetryList, 11, 3)
            plt.plot(strideCountList, stepLengthAsymmetry_smoothed, 'b-', linewidth=2)
        else:
            plt.plot(strideCountList, stepLengthAsymmetryList, 'b-', linewidth=2)
    
    # Set axis limits and labels
    if len(strideCountList) > 0:
        plt.xlim([0, max(strideCountList)])
    plt.ylim([-0.5, 0.5])
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Stride index', fontsize=12)
    plt.ylabel('Step length symmetry', fontsize=12)
    
    # Add vertical lines at transition points
    if transitionPointsFound:
        plt.axvline(x=adaptationStart, color='k', linestyle=':', alpha=0.5)
        plt.axvline(x=postAdaptationStart, color='k', linestyle=':', alpha=0.5)
    
    # =============================================
    # 3. METABOLIC RATE SUBPLOT
    # =============================================
    plt.subplot(1, 3, 3)
    
    # Calculate 2-step averaged metabolic rate
    if len(EdotStore_IterationAverage) > 1:
        # Raw metabolic rate (2-step average)
        stride_indices = np.arange(len(EdotStore_IterationAverage))
        plt.plot(stride_indices, EdotStore_IterationAverage, 'b-', linewidth=1, label="2 step average")
        
        # Smoothed metabolic rate using VO2 conversion
        if len(tSpan_Smoothed) > 0 and len(EmetSList_Smoothed) > 0:
            # Map the smoothed VO2 data to stride indices
            stride_time_cumsum = np.cumsum(tTotalIterationStore)
            
            # Make sure we have enough points for proper mapping
            if len(stride_time_cumsum) > 1 and len(tSpan_Smoothed) > 1:
                # Create interpolation function
                stride_indices = np.arange(len(stride_time_cumsum))
                
                # Create mapping from time to stride index using interpolation
                time_to_stride = interp1d(stride_time_cumsum, stride_indices, 
                                         bounds_error=False, fill_value="extrapolate")
                
                # Convert smoothed times to stride indices
                smoothed_stride_indices = time_to_stride(tSpan_Smoothed)
                
                # Plot the smoothed data
                plt.plot(smoothed_stride_indices, EmetSList_Smoothed, 'r-', 
                        linewidth=2, label="Edot smoothed by VO2")
    
    # Set axis limits and labels
    if len(EdotStore_IterationAverage) > 0:
        plt.xlim([0, len(EdotStore_IterationAverage)])
        max_met_rate = max(np.max(EdotStore_IterationAverage), 
                          np.max(EmetSList_Smoothed) if len(EmetSList_Smoothed) > 0 else 0)
        plt.ylim([0, max_met_rate * 1.1])
    else:
        plt.ylim([0, 0.4])  # Default if no data
    
    plt.xlabel('Stride index', fontsize=12)
    plt.ylabel('Edot, met rate', fontsize=12)
    plt.legend(fontsize=10)
    
    # Add vertical lines at transition points
    if transitionPointsFound:
        plt.axvline(x=adaptationStart, color='k', linestyle=':', alpha=0.5)
        plt.axvline(x=postAdaptationStart, color='k', linestyle=':', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # If this is a tied-belt simulation, also generate those plots
    if paramFixed['SplitOrTied'] == 'tied':
        plt.figure(201, figsize=(15, 6))
        plt.suptitle('Tied Treadmill: Step frequency changes in response to speed changes')
        
        # Treadmill speed subplot
        plt.subplot(1, 2, 1)
        foot1SpeedList, foot2SpeedList = getTreadmillSpeed(params['tList'], paramFixed['imposedFootSpeeds'])
        plt.plot(params['tList'], np.abs(foot1SpeedList), 'b-', linewidth=2, label='Left belt')
        plt.plot(params['tList'], np.abs(foot2SpeedList), 'r-', linewidth=2, label='Right belt')
        plt.xlabel('Time')
        plt.ylabel('Treadmill belt speeds')
        plt.legend()
        
        max_speed = max(np.max(np.abs(foot1SpeedList)), np.max(np.abs(foot2SpeedList)))
        plt.ylim([0, max_speed * 1.1])
        
        # Step frequency subplot
        plt.subplot(1, 2, 2)
        if len(tStance_fast) > 0 and len(tStance_slow) > 0:
            tStancePerStride = (tStance_fast + tStance_slow) / 2
            tList_stepBegin = np.cumsum(tStanceList)
            plt.plot(tList_stepBegin[::2], 1. / tStancePerStride, 'g-', linewidth=2)
            
            # Set y-axis limits based on calculated frequencies
            max_freq = np.max(1. / tStancePerStride)
            plt.ylim([0, max_freq * 1.2])
        else:
            plt.ylim([0, 0.8])  # Default if no data
            
        plt.xlabel('Time (non dim)')
        plt.ylabel('Step freq, averaged over 2 steps')
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()