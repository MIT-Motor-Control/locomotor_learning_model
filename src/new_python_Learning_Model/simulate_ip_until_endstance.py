from scipy.integrate import solve_ivp
import numpy as np


def simulate_ip_until_endstance(statevar0, tspan, paramFixed, paramController):
    """
    Simulate inverted pendulum until endstance.

    Parameters:
    - statevar0: array-like, initial state variables.
    - tspan: tuple, time span for the simulation (t0, tf).
    - paramFixed: dict or any, fixed parameters for the ODE.
    - paramController: dict or any, controller parameters for the ODE.

    Returns:
    - tlistTillEndstance: array, time points where the solution was evaluated.
    - statevarlistTillEndstance: array, solution values at the time points.
    """
    

    # Define relative and absolute tolerances
    tol = 1e-10

    # Combine parameters into args tuple
    args = (paramFixed, paramController)

    # Ensure tspan is a numpy array
    tspan = np.asarray(tspan)

    # Extract t0 and tf
    t0 = tspan[0]
    tf = tspan[-1]

    # Determine if t_eval should be used
    if len(tspan) > 2:
        t_eval = tspan
    else:
        t_eval = None  # Let the solver choose time points

    # Solve ODE with event detection
    sol = solve_ivp(
        singlePendulumODE,
        (t0, tf),
        statevar0,
        method='RK45',
        atol=tol,
        rtol=tol,
        t_eval=t_eval,
        args=args,
        events=DetectEndstance
    )

    # Extract time points and state variables
    tlistTillEndstance = sol.t
    statevarlistTillEndstance = sol.y.T  # Transpose to match MATLAB output

    return tlistTillEndstance, statevarlistTillEndstance

def singlePendulumODE(t, y, paramFixed, paramController):
    """
    Compute the derivatives for the single pendulum ODE.
    """

    # Ensure y is a flat array
    y = np.asarray(y).flatten()

    # Unpack state variables
    theta = y[0]
    omega = y[1]

    # Extract fixed parameters
    g = paramFixed.get('g', 9.81)  # Gravity
    L = paramFixed.get('L', 1.0)   # Pendulum length
    m = paramFixed.get('m', 1.0)   # Mass

    # Controller torque
    if callable(paramController):
        torque = paramController(t, y)
    else:
        torque = paramController.get('torque', 0)

    # Equations of motion
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta) + torque / (m * L**2)

    return [dtheta_dt, domega_dt]

def DetectEndstance(t, y, paramFixed, paramController):
    """
    Event function to detect endstance.
    """

    # Ensure y is a flat array
    y = np.asarray(y).flatten()
    theta = y[0]

    # Event occurs when theta crosses zero
    value = theta
    return value

