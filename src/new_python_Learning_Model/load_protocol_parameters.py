import matplotlib.pyplot as plt
import numpy as np

def load_protocol_parameters(param_fixed):
    """
    Load parameters related to the adaptation protocol, such as treadmill speed changes and split-belt vs tied-belt settings.
    
    Args:
        param_fixed (dict): Dictionary containing existing fixed parameters.
        
    Returns:
        dict: Updated dictionary with protocol parameters.
    """
    # Ensure 'Learner' dictionary and 'numStepsPerIteration' key exist
    if 'Learner' not in param_fixed:
        param_fixed['Learner'] = {}
    if 'numStepsPerIteration' not in param_fixed['Learner']:
        param_fixed['Learner']['numStepsPerIteration'] = 100  # Default value

    # Speed protocol: split belt changes
    param_fixed['SplitOrTied'] = 'split'
    param_fixed['speedProtocol'] = 'classic split belt'
    param_fixed['transitionTime'] = 15  # in seconds
    param_fixed['imposedFootSpeeds'] = make_treadmill_speed_split(param_fixed)
    
    # Additional protocol parameters
    param_fixed['angleSlope'] = 0  # Do not change: code has not been tested for non-zero values

    # How many steps to simulate
    nominal_step_time = 1.7
    param_fixed['numStepsToLearn'] = param_fixed['imposedFootSpeeds']['tList'][-1] / nominal_step_time
    param_fixed['numStepsToLearn'] = round(param_fixed['numStepsToLearn'] / 100) * 100  # Round to the nearest hundred

    # Optimization iterations
    param_fixed['numIterations'] = param_fixed['numStepsToLearn'] // param_fixed['Learner']['numStepsPerIteration']

    if param_fixed['Learner']['numStepsPerIteration'] % 2 != 0:
        param_fixed['Learner']['numStepsPerIteration'] += 1
    
    return param_fixed

def make_treadmill_speed_split(param_fixed):
    """
    Define foot speeds for the split-belt treadmill protocol and plot the speed profile.
    
    Args:
        param_fixed (dict): Dictionary containing existing fixed parameters.
        
    Returns:
        dict: Dictionary containing imposed foot speeds for different phases of the split-belt treadmill protocol.
    """
    L = 0.95  # Length parameter
    g = 9.81  # Acceleration due to gravity
    time_scaling = np.sqrt(L / g)

    # Speed parameters
    delta = 0.0328  # Half of normal delta
    v_normal = -0.3276
    v_fast = v_normal - 5 * delta
    v_slow = v_normal + 5 * delta

    # Define treadmill speed profiles for different phases based on speed protocol
    t_store = []
    foot_speed1_store = []
    foot_speed2_store = []

    if param_fixed['speedProtocol'] == 'single speed':
        # Phase 1: Warmup
        t_duration1 = 9 * 60 / time_scaling
        foot_speed1_phase1 = v_normal
        foot_speed2_phase1 = v_normal

        # Phase 2: Baseline
        t_duration2 = 1 * 60 / time_scaling
        foot_speed1_phase2 = v_normal
        foot_speed2_phase2 = v_normal

        # Phase 3: Adaptation
        t_duration3 = 1 * 60 / time_scaling
        foot_speed1_phase3 = v_normal
        foot_speed2_phase3 = v_normal

        # Phase 4: Post-adaptation
        t_duration4 = 1 * 60 / time_scaling
        foot_speed1_phase4 = v_normal
        foot_speed2_phase4 = v_normal

        # Store values
        t_store = [0, t_duration1, t_duration2, t_duration3, t_duration4]
        t_store = np.cumsum(t_store)

        foot_speed1_store = [foot_speed1_phase1, foot_speed1_phase1, foot_speed1_phase2, foot_speed1_phase3, foot_speed1_phase4]
        foot_speed2_store = [foot_speed2_phase1, foot_speed2_phase1, foot_speed2_phase2, foot_speed2_phase3, foot_speed2_phase4]

    elif param_fixed['speedProtocol'] == 'classic split belt':
        # Phase 1: Warmup (tied)
        t_duration1 = 1 * 60 / time_scaling
        foot_speed1_phase1 = v_normal
        foot_speed2_phase1 = v_normal

        # Phase 2: Baseline (tied)
        t_duration2 = 5 * 60 / time_scaling
        foot_speed1_phase2 = v_normal
        foot_speed2_phase2 = v_normal

        # Phase 3: Split adaptation
        t_duration3 = 45 * 60 / time_scaling
        foot_speed1_phase3 = v_fast
        foot_speed2_phase3 = v_slow

        # Phase 4: Second adaptation (back to tied)
        t_duration4 = 5 * 60 / time_scaling
        foot_speed1_phase4 = v_normal
        foot_speed2_phase4 = v_normal

        # Store values
        t_store = [0, t_duration1, t_duration2, t_duration3, t_duration4]
        t_store = np.cumsum(t_store)

        foot_speed1_store = [foot_speed1_phase1, foot_speed1_phase1, foot_speed1_phase2, foot_speed1_phase3, foot_speed1_phase4]
        foot_speed2_store = [foot_speed2_phase1, foot_speed2_phase1, foot_speed2_phase2, foot_speed2_phase3, foot_speed2_phase4]

    # Adding transition phases
    t_store_new = [0]
    foot_speed1_store_new = [foot_speed1_store[0]]
    foot_speed2_store_new = [foot_speed2_store[0]]

    for i_tran in range(1, len(t_store)):
        if i_tran < len(t_store) - 1:
            t_store_new.extend([t_store[i_tran], t_store[i_tran] + param_fixed['transitionTime'] / time_scaling])
            foot_speed1_store_new.extend([foot_speed1_store[i_tran], foot_speed1_store[i_tran + 1]])
            foot_speed2_store_new.extend([foot_speed2_store[i_tran], foot_speed2_store[i_tran + 1]])
        else:
            t_store_new.append(t_store[i_tran])
            foot_speed1_store_new.append(foot_speed1_store[i_tran])
            foot_speed2_store_new.append(foot_speed2_store[i_tran])

    # Plotting the treadmill speed profile
    plt.figure()
    plt.plot(t_store_new, np.abs(foot_speed1_store_new), linewidth=2, label='(abs) fast belt')
    plt.plot(t_store_new, np.abs(foot_speed2_store_new), linewidth=2, label='(abs) slow belt')
    plt.xlabel('Time (s)')
    plt.ylabel('Treadmill Speeds (non-dimensional)')
    plt.legend()
    plt.ylim([0, abs(v_fast) * 1.25])
    plt.title('Split belt speed change protocol')
    plt.grid(True)
    plt.show()

    # Store in a structure
    belt_speeds_imposed = {
        'tList': t_store_new,
        'footSpeed1List': foot_speed1_store_new,
        'footSpeed2List': foot_speed2_store_new
    }

    # Calculate foot accelerations
    a1_list = np.diff(belt_speeds_imposed['footSpeed1List']) / np.diff(belt_speeds_imposed['tList'])
    a2_list = np.diff(belt_speeds_imposed['footSpeed2List']) / np.diff(belt_speeds_imposed['tList'])
    a1_list = np.append(a1_list, 0)
    a2_list = np.append(a2_list, 0)

    belt_speeds_imposed['footAcc1List'] = a1_list
    belt_speeds_imposed['footAcc2List'] = a2_list

    return belt_speeds_imposed
