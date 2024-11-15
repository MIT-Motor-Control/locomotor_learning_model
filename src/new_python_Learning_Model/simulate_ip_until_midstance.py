import numpy as np
from scipy.integrate import solve_ivp

def simulate_ip_until_midstance(statevar0, tspan, param_fixed, param_controller):
    """
    Simulate inverted pendulum until midstance.

    Args:
        statevar0 (ndarray): Initial state variables.
        tspan (ndarray): Time span for simulation.
        param_fixed (dict): Fixed parameters for the simulation.
        param_controller (dict): Controller parameters.

    Returns:
        tuple: Time list until midstance and state variable list until midstance.
    """
    # Define event function to detect midstance
    def detect_midstance(t, y):
        return y[0]  # Example event condition: angle theta = 0
    detect_midstance.terminal = True
    detect_midstance.direction = 0

    # Set options for the ODE solver
    options = {
        'rtol': 1e-10,
        'atol': 1e-10,
        'events': detect_midstance
    }

    # Define the single pendulum ODE
    def single_pendulum_ode(t, y):
        # Placeholder for the actual pendulum ODE equations
        dydt = [y[1], -param_fixed['leglength'] * np.sin(y[0])]  # Example ODE
        return dydt

    # Solve the ODE until midstance
    sol = solve_ivp(single_pendulum_ode, [tspan[0], tspan[-1]], statevar0, t_eval=tspan, **options)

    tlist_till_midstance = sol.t
    statevarlist_till_midstance = sol.y.T

    return tlist_till_midstance, statevarlist_till_midstance

