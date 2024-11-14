import numpy as np

def get_treadmill_speed(t, belt_speeds_imposed):
    """
    Get the treadmill speeds for foot 1 and foot 2 at a specific time.

    Args:
        t (float): The time at which to evaluate the treadmill speeds.
        belt_speeds_imposed (dict): Dictionary containing the imposed treadmill speeds, with keys 'tList', 'footSpeed1List', and 'footSpeed2List'.

    Returns:
        tuple: Tuple containing the foot speeds for foot 1 and foot 2 at time t.
    """
    foot_speed1 = np.interp(t, belt_speeds_imposed['tList'], belt_speeds_imposed['footSpeed1List'], left=None, right=None)
    foot_speed2 = np.interp(t, belt_speeds_imposed['tList'], belt_speeds_imposed['footSpeed2List'], left=None, right=None)

    return foot_speed1, foot_speed2