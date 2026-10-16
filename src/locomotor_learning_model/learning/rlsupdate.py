from __future__ import annotations

import numpy as np
from scipy.linalg import qr, solve_triangular


def _pivoted_basic_lstsq(
    input_now_store: np.ndarray,
    output_next_store: np.ndarray,
) -> np.ndarray:
    """Solve a least-squares system with a pivoted basic solution.

    The rolling estimator can become rank deficient, so pivoted QR is used to
    select a consistent subset of independent columns before solving.
    """

    q_factor, r_factor, pivot = qr(input_now_store, mode="economic", pivoting=True)
    tolerance = max(input_now_store.shape) * np.finfo(r_factor.dtype).eps * np.abs(r_factor).max()
    rank = int(np.sum(np.abs(np.diag(r_factor)) > tolerance))

    q_transpose_y = q_factor.T @ output_next_store
    coefficients = np.zeros((input_now_store.shape[1], output_next_store.shape[1]))
    if rank > 0:
        coefficients[:rank, :] = solve_triangular(
            r_factor[:rank, :rank],
            q_transpose_y[:rank, :],
            lower=False,
        )

    basic_solution = np.zeros_like(coefficients)
    basic_solution[pivot, :] = coefficients
    return basic_solution


def rlsupdate(
    aold: np.ndarray,
    beta: float,
    input_now_store: np.ndarray,
    output_next_store: np.ndarray,
    mu_measurement: float,
    num_steps_to_use: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Update a rolling linear model from recent input/output histories.

    ``beta`` and ``mu_measurement`` are unused compatibility arguments from
    the MATLAB signature. This estimator fits a finite window without
    exponential forgetting.
    """

    # Pick only the most recent ``num_steps_to_use`` strides. Each column is a
    # single step.
    input_now_store = input_now_store[:, -num_steps_to_use:]
    
    if output_next_store.ndim == 1:
        output_next_store = output_next_store[-num_steps_to_use:]
        output_next_store = output_next_store.reshape(1, -1)
    else:
        output_next_store = output_next_store[:, -num_steps_to_use:]

    # Transpose so that each row represents one step.
    input_now_store = input_now_store.T
    output_next_store = output_next_store.T

    # Add the column of ones for the constant term.
    input_now_store = np.hstack(
        [input_now_store, np.ones((input_now_store.shape[0], 1))]
    )

    # Use pivoted QR so rank-deficient windows choose a consistent solution.
    anew = _pivoted_basic_lstsq(input_now_store, output_next_store)

    # Compute errors using the old model.
    aold_t = aold.T
    error_old_all_steps = output_next_store - input_now_store @ aold_t
    error_old_most_recent_step = error_old_all_steps[-1, :].reshape(-1, 1)

    # Compute errors using the new model.
    error_new_all_steps = output_next_store - input_now_store @ anew
    error_new_most_recent_step = error_new_all_steps[-1, :].reshape(-1, 1)

    # Transpose back so that rows correspond to outputs.
    anew = anew.T

    return anew, error_old_most_recent_step, error_new_most_recent_step
