import numpy as np
import matplotlib.pyplot as plt
 
def make_treadmill_speed_split(paramFixed):
    # ------------------------------------------------------------------
    # exactly as in MATLAB:
    L = 0.95
    g = 9.81
    timeScaling = np.sqrt(L / g)

    # +5 delta
    delta = 0.0328  # half of normal delta
    vNormal = -0.3276
    vFast   = vNormal - 5 * delta
    vSlow   = vNormal + 5 * delta

    # ------------------------------------------------------------------
    # branch on protocol
    speedProtocol = paramFixed['speedProtocol']
    if speedProtocol == 'single speed':
        # transitioning from one speed to next takes
        tDurationTransition = paramFixed['transitionTime'] / timeScaling

        # phase durations (non-dimensionalized)
        tDuration1 = 9 * 60 / timeScaling
        tDuration2 = 1 * 60 / timeScaling
        tDuration3 = 1 * 60 / timeScaling
        tDuration4 = 1 * 60 / timeScaling

        # tied‐belt speeds
        f1_p1 = f2_p1 = vNormal
        f1_p2 = f2_p2 = vNormal
        f1_p3 = f2_p3 = vNormal
        f1_p4 = f2_p4 = vNormal

    elif speedProtocol == 'classic split belt':
        tDurationTransition = paramFixed['transitionTime'] / timeScaling

        tDuration1 = 1 * 60 / timeScaling
        tDuration2 = 5 * 60 / timeScaling
        tDuration3 = 45 * 60 / timeScaling
        tDuration4 = 5 * 60 / timeScaling

        f1_p1 = f2_p1 = vNormal
        f1_p2 = f2_p2 = vNormal
        f1_p3, f2_p3 = vFast, vSlow
        f1_p4 = f2_p4 = vNormal

    else:
        raise ValueError(f"Unknown speedProtocol: {speedProtocol}")

    # build the “phase” arrays exactly as MATLAB’s tStore and speed stores
    tStore = np.cumsum([0, tDuration1, tDuration2, tDuration3, tDuration4])
    footSpeed1Store = np.array([f1_p1, f1_p1, f1_p2, f1_p3, f1_p4])
    footSpeed2Store = np.array([f2_p1, f2_p1, f2_p2, f2_p3, f2_p4])

    # ------------------------------------------------------------------
    # insert transition segments
    tStore_new       = [tStore[0]]
    foot1_new        = [footSpeed1Store[0]]
    foot2_new        = [footSpeed2Store[0]]

    for i in range(1, len(tStore)):
        if i < len(tStore) - 1:
            # start of phase i
            tStore_new.append(tStore[i])
            foot1_new.append(footSpeed1Store[i])
            foot2_new.append(footSpeed2Store[i])
            # end of transition into phase i+1
            tStore_new.append(tStore[i] + tDurationTransition)
            foot1_new.append(footSpeed1Store[i+1])
            foot2_new.append(footSpeed2Store[i+1])
        else:
            # last phase just append once
            tStore_new.append(tStore[i])
            foot1_new.append(footSpeed1Store[i])
            foot2_new.append(footSpeed2Store[i])

    # convert lists back to arrays
    tList            = np.array(tStore_new)
    footSpeed1List   = np.array(foot1_new)
    footSpeed2List   = np.array(foot2_new)

    # ------------------------------------------------------------------
    # compute accelerations
    a1List = np.diff(footSpeed1List) / np.diff(tList)
    a2List = np.diff(footSpeed2List) / np.diff(tList)
    # pad with zero at end to match lengths
    a1List = np.append(a1List, 0.0)
    a2List = np.append(a2List, 0.0)

    # optional: plot like MATLAB
    plt.figure(2555)
    plt.plot(tList, np.abs(footSpeed1List), linewidth=2, label='(abs) fast belt')
    plt.plot(tList, np.abs(footSpeed2List), linewidth=2, label='(abs) slow belt')
    plt.xlabel('t')
    plt.ylabel('treadmill speeds (non-dimensional)')
    plt.ylim([0, abs(vFast) * 1.25])
    plt.title('Split belt speed change protocol')
    plt.legend()
    plt.show()

    # ------------------------------------------------------------------
    # pack into the same‐named structure/dict
    beltSpeedsImposed = {
        'tList':            tList,
        'footSpeed1List':   footSpeed1List,
        'footSpeed2List':   footSpeed2List,
        'footAcc1List':     a1List,
        'footAcc2List':     a2List
    }

    return beltSpeedsImposed
