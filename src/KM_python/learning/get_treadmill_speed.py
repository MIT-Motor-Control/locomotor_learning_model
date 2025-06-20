from typing import Tuple
import numpy as np

def get_treadmill_speed(t: float, belt_speeds_imposed) -> Tuple[float, float]:
    """Interpolate the treadmill belt speeds at time ``t``.
    This follows ``getTreadmillSpeed.m`` line by line, supporting either a
    dictionary with ``t_list``/``foot_speed*_list`` entries or an object with
    ``tList``/``footSpeed*List`` attributes.
    """

    try:  # object with MATLAB-style attributes
        t_list = belt_speeds_imposed.tList
        foot_speed1_list = belt_speeds_imposed.footSpeed1List
        foot_speed2_list = belt_speeds_imposed.footSpeed2List
    except AttributeError:  # mapping with dictionary keys (as produced by make_treadmill_speed_split)
        t_list = belt_speeds_imposed["tList"]
        foot_speed1_list = belt_speeds_imposed["footSpeed1List"]
        foot_speed2_list = belt_speeds_imposed["footSpeed2List"]

    foot_speed1 = np.interp(t, t_list, foot_speed1_list)
    foot_speed2 = np.interp(t, t_list, foot_speed2_list)

    return float(foot_speed1), float(foot_speed2)