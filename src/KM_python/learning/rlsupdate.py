from __future__ import annotations

from typing import Tuple

import numpy as np


def rlsupdate(
    aold: np.ndarray,
    beta: float,
    input_now_store: np.ndarray,
    output_next_store: np.ndarray,
    mu_measurement: float,
    num_steps_to_use: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform the simple least-squares model update.

    This is a direct line-by-line translation of the MATLAB function
    ``RLSupdate.m``.  The optional parameters ``beta`` and ``mu_measurement``
    are included for API compatibility but are not used.
    """

    # Pick only the most recent ``num_steps_to_use`` strides. Each column is a
    # single step.
    input_now_store = input_now_store[:, -num_steps_to_use:]
    
    # Handle both 1D and 2D output arrays
    if output_next_store.ndim == 1:
        # For 1D arrays, convert to row vector and slice
        output_next_store = output_next_store[-num_steps_to_use:]
        output_next_store = output_next_store.reshape(1, -1)
    else:
        # For 2D arrays, slice columns normally
        output_next_store = output_next_store[:, -num_steps_to_use:]

    # Transpose so that each row represents one step.
    input_now_store = input_now_store.T
    output_next_store = output_next_store.T

    # Add the column of ones for the constant term.
    input_now_store = np.hstack(
        [input_now_store, np.ones((input_now_store.shape[0], 1))]
    )

    # Solve the least-squares problem.
    anew, *_ = np.linalg.lstsq(input_now_store, output_next_store, rcond=None)

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