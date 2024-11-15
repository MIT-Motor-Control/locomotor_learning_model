from scipy.integrate import solve_ivp
from singlePendulumODE import singlePendulumODE
from DetectEndstance import DetectEndstance

def simulate_ip_until_endstance(statevar0, tspan_2, paramFixed, paramController):
    """
    Simulate inverted pendulum until endstance.

    Parameters:
    - statevar0: array-like, initial state variables.
    - tspan: tuple, time span for the simulation (t0, tf).
    - paramFixed: dict, fixed parameters for the ODE.
    - paramController: dict, controller parameters for the ODE.

    Returns:
    - tlistTillEndstance: array, time points where the solution was evaluated.
    - statevarlistTillEndstance: array, solution values at the time points.
    """
    

    # Define relative and absolute tolerances
    tol = 1e-10

    # Combine parameters into args tuple
    args = (paramFixed, paramController)
    print(tspan_2)

    # Solve ODE with event detection
    sol = solve_ivp(
        fun=singlePendulumODE,
        t_span=tspan_2,
        y0=statevar0,
        method='RK45',
        atol=tol,
        rtol=tol,
        args=args,
        events=DetectEndstance
    )

    # Extract time points and state variables
    tlistTillEndstance = sol.t
    statevarlistTillEndstance = sol.y.T  # Transpose to match MATLAB output

    return tlistTillEndstance, statevarlistTillEndstance

