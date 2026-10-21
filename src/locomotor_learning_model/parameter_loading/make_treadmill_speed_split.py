import numpy as np


def make_treadmill_speed_split(paramFixed):
    """Build the baseline split-belt treadmill profile."""
    # Model constants used to nondimensionalize treadmill timings.
    L = 0.95
    g = 9.81
    timeScaling = np.sqrt(L / g)
    paramFixed['timeScaling'] = timeScaling

    # Scale the speed difference about the fixed mean belt speed.
    delta = 0.0328
    vNormal = -0.3276
    if 'splitSpeedDiffMS' in paramFixed:
        speed_diff_ms = float(paramFixed['splitSpeedDiffMS'])
        if not np.isfinite(speed_diff_ms) or speed_diff_ms < 0:
            raise ValueError("splitSpeedDiffMS must be finite and non-negative")
        half_delta = 5 * delta * speed_diff_ms
    else:
        speed_diff_ms = 1.0
        half_delta = 5 * delta
    vFast = vNormal - half_delta
    vSlow = vNormal + half_delta
    paramFixed['splitSpeedDiffMS'] = speed_diff_ms

    speedProtocol = paramFixed['speedProtocol']
    if speedProtocol == 'single speed':
        tDurationTransition = paramFixed['transitionTime'] / timeScaling

        tDuration1 = 9 * 60 / timeScaling
        tDuration2 = 1 * 60 / timeScaling
        tDuration3 = 1 * 60 / timeScaling
        tDuration4 = 1 * 60 / timeScaling

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

    tStore = np.cumsum([0, tDuration1, tDuration2, tDuration3, tDuration4])
    footSpeed1Store = np.array([f1_p1, f1_p1, f1_p2, f1_p3, f1_p4])
    footSpeed2Store = np.array([f2_p1, f2_p1, f2_p2, f2_p3, f2_p4])

    # Insert finite-duration transitions between phases.
    tStore_new       = [tStore[0]]
    foot1_new        = [footSpeed1Store[0]]
    foot2_new        = [footSpeed2Store[0]]

    for i in range(1, len(tStore)):
        if i < len(tStore) - 1:
            tStore_new.append(tStore[i])
            foot1_new.append(footSpeed1Store[i])
            foot2_new.append(footSpeed2Store[i])
            tStore_new.append(tStore[i] + tDurationTransition)
            foot1_new.append(footSpeed1Store[i+1])
            foot2_new.append(footSpeed2Store[i+1])
        else:
            tStore_new.append(tStore[i])
            foot1_new.append(footSpeed1Store[i])
            foot2_new.append(footSpeed2Store[i])

    tList            = np.array(tStore_new)
    footSpeed1List   = np.array(foot1_new)
    footSpeed2List   = np.array(foot2_new)

    a1List = np.diff(footSpeed1List) / np.diff(tList)
    a2List = np.diff(footSpeed2List) / np.diff(tList)
    a1List = np.append(a1List, 0.0)
    a2List = np.append(a2List, 0.0)

    beltSpeedsImposed = {
        'tList':            tList,
        'footSpeed1List':   footSpeed1List,
        'footSpeed2List':   footSpeed2List,
        'footAcc1List':     a1List,
        'footAcc2List':     a2List
    }

    return beltSpeedsImposed
